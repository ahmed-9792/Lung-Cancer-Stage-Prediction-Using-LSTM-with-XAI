from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
import cv2
import pytesseract
from PIL import Image
import re
import os
from werkzeug.utils import secure_filename
import base64
import io
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Global variables to store model and preprocessors
lstm_model = None
scaler = None
feature_names = None
sequence_length = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_float_conversion(value, default=0.0):
    """Safely convert a value to float, return default if conversion fails"""
    if value is None or value == '' or value == 'None':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def load_lstm_model():
    """Load the LSTM model and preprocessors"""
    global lstm_model, scaler, feature_names, sequence_length
    
    try:
        # Load the trained LSTM model
        lstm_model = load_model('lung_cancer_lstm_model.h5')
        
        # Load preprocessors
        with open('scaler.pkl', 'rb') as f:
            data = pickle.load(f)
            scaler = data['scaler']
            feature_names = data['feature_names']
            sequence_length = data['sequence_length']
        
        print("LSTM model and preprocessors loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        return False

def preprocess_image(image_path):
    """Preprocess image for better OCR results"""
    try:
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply thresholding to get binary image
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def extract_biomarker_data(image_path):
    """Extract biomarker data from medical report image using OCR"""
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)
        
        if processed_img is None:
            return None
        
        # Configure tesseract for better number recognition
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()/:.,- '
        
        # Extract text from image
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        print(f"Extracted text: {text}")
        
        # Initialize data dictionary with default values
        extracted_data = {
            'Age': None,
            'ctDNA_EGFR': None,
            'ctDNA_KRAS': None,
            'miRNA_21': None,
            'miRNA_155': None,
            'Protein_CEA': None,
            'Protein_CYFRA': None,
            'Gender': None,
            'Smoking_History': None
        }
        
        # Extract age
        age_patterns = [
            r'Age[:\s]*(\d+)',
            r'age[:\s]*(\d+)',
            r'AGE[:\s]*(\d+)'
        ]
        for pattern in age_patterns:
            age_match = re.search(pattern, text, re.IGNORECASE)
            if age_match:
                extracted_data['Age'] = safe_float_conversion(age_match.group(1))
                break
        
        # Extract gender
        gender_patterns = [
            r'Gender[:\s]*(Male|Female)',
            r'gender[:\s]*(male|female)',
            r'GENDER[:\s]*(MALE|FEMALE)'
        ]
        for pattern in gender_patterns:
            gender_match = re.search(pattern, text, re.IGNORECASE)
            if gender_match:
                extracted_data['Gender'] = gender_match.group(1).capitalize()
                break
        
        # Extract smoking history
        smoking_patterns = [
    r'Smoking\s+Habits\s*[:\s]\s*(Never|Current|Former)', # Handles "Smoking Habits: Never"
    r'Smoking\s*Habits?\s*[:\s]\s*(Never|Current|Former)',
    r'Smoking\s+History\s*[:\s]\s*(Never|Current|Former)', # Alternative wording
    r'SMOKING\s*:\s*(Never|Current|Former)' # Simpler, if only "SMOKING: value"
]
        for pattern in smoking_patterns:
            smoking_match = re.search(pattern, text, re.IGNORECASE)
            if smoking_match:
                extracted_data['Smoking_History'] = smoking_match.group(1).capitalize()
                break
        
        # Extract biomarker values with improved patterns
        biomarker_patterns = {
    'ctDNA_EGFR': [
        r'ctDNA\s+EGFR\s*\(ng/mL\)\s*[:\s]*\s*(\d+\.?\d*)', # Handles "(ng/mL)" and colon/spaces
        r'ctDNA\s+EGFR[^0-9\n]*?(\d+\.?\d*)',         # General: skips non-digits/newlines until number
        r'EGFR[^0-9\n]*?(\d+\.?\d*)'                   # Fallback: only biomarker name
    ],
    'ctDNA_KRAS': [
        r'ctDNA\s+KRAS\s*\(ng/mL\)\s*[:\s]*\s*(\d+\.?\d*)',
        r'ctDNA\s+KRAS[^0-9\n]*?(\d+\.?\d*)',
        r'KRAS[^0-9\n]*?(\d+\.?\d*)'
    ],
    'miRNA_21': [ # Was working, but add robust version
        r'miRNA[_\s]*21\s*[:\s]*\s*(\d+\.?\d*)',
        r'miRNA[_\s]*21[^0-9\n]*?(\d+\.?\d*)'
    ],
    'miRNA_155': [
        r'miRNA[_\s]*155\s*[:\s]*\s*(\d+\.?\d*)',
        r'miRNA[_\s]*155[^0-9\n]*?(\d+\.?\d*)' # This should capture "miRNA_155 155.79"
    ],
    'Protein_CEA': [
        r'Protein[_\s]*CEA\s*\(ng/mL\)\s*[:\s]*\s*(\d+\.?\d*)',
        r'Protein[_\s]*CEA[^0-9\n]*?(\d+\.?\d*)',
        r'CEA[^0-9\n]*?(\d+\.?\d*)'
    ],
    'Protein_CYFRA': [
        r'Protein[_\s]*CYFRA\s*\(ng/mL\)\s*[:\s]*\s*(\d+\.?\d*)', # Matches "Protein_CYFRA (ng/mL): VALUE"
        r'Protein[_\s]*CYFRA[^0-9\n]*?(\d+\.?\d*)',
        r'CYFRA[^0-9\n]*?(\d+\.?\d*)'
    ]
}
        
        # Extract biomarker values
        for biomarker, patterns in biomarker_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = safe_float_conversion(match.group(1))
                    if value != 0.0:  # Only set if we got a valid non-zero value
                        extracted_data[biomarker] = value
                        break
        
        print(f"Extracted data: {extracted_data}")
        return extracted_data
        
    except Exception as e:
        print(f"Error extracting biomarker data: {e}")
        return None

def predict_with_lstm(input_data):
    """Make prediction using the LSTM model"""
    global lstm_model, scaler, feature_names, sequence_length
    
    if lstm_model is None or scaler is None:
        return None
    
    try:
        # Ensure input is in correct format
        if isinstance(input_data, dict):
            # Convert dictionary to array based on feature names
            input_array = []
            for feature in feature_names:
                if feature in input_data:
                    input_array.append(input_data[feature])
                else:
                    print(f"Missing feature: {feature}")
                    # Use default value for missing features
                    input_array.append(0.0)
            input_data = np.array(input_array)
        
        # Scale input
        input_scaled = scaler.transform(input_data.reshape(1, -1))
        
        # Create sequence for LSTM
        input_seq = []
        for i in range(sequence_length):
            if i == 0:
                input_seq.append(input_scaled[0])
            else:
                # Add slight variation as done in the original code
                variation = input_scaled[0] * (1 + np.random.normal(0, 0.01, len(input_scaled[0])))
                input_seq.append(variation)
        
        input_seq = np.array(input_seq).reshape(1, sequence_length, -1)
        
        # Make prediction
        prediction = lstm_model.predict(input_seq, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Map prediction to label
        class_labels = ['No Cancer', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
        predicted_label = class_labels[predicted_class]
        
        return {
            'prediction': predicted_label,
            'confidence': float(confidence),
            'probabilities': {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}
        }
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

# Load model on startup
load_lstm_model()

@app.route('/')
def index():
    """Renders the homepage"""
    return render_template('index.html')

@app.route('/indexproj')
def indexproj():
    """Renders the prediction result page"""
    return render_template('indexproj.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data from the user input with safe conversion
        age = safe_float_conversion(request.form.get('Age', ''), 50.0)  # Default age 50
        gender = request.form.get('Gender', 'Male')
        smoking_history = request.form.get('Smoking_History', 'Never')
        
        # Biomarker data with safe conversion
        ctDNA_EGFR = safe_float_conversion(request.form.get('ctDNA_EGFR', ''), 0.0)
        ctDNA_KRAS = safe_float_conversion(request.form.get('ctDNA_KRAS', ''), 0.0)
        miRNA_21 = safe_float_conversion(request.form.get('miRNA_21', ''), 0.0)
        miRNA_155 = safe_float_conversion(request.form.get('miRNA_155', ''), 0.0)
        Protein_CEA = safe_float_conversion(request.form.get('Protein_CEA', ''), 0.0)
        Protein_CYFRA = safe_float_conversion(request.form.get('Protein_CYFRA', ''), 0.0)
        
        # Process categorical variables to match LSTM preprocessing
        is_smoker = 1 if smoking_history.lower() != 'never' else 0
        is_female = 1 if gender.lower() == 'female' else 0
        
        # Create input dictionary based on the feature names from LSTM model
        input_data = {}
        
        # Map form inputs to LSTM feature names
        feature_mapping = {
            'Age': age,
            'is_smoker': is_smoker,
            'is_female': is_female,
            'ctDNA_EGF': ctDNA_EGFR,
            'ctDNA_KRAS': ctDNA_KRAS,
            'miRNA_21': miRNA_21,
            'miRNA_155': miRNA_155,
            'Protein_CEA': Protein_CEA,
            'Protein_CYFRA': Protein_CYFRA
        }
        
        # If we have the actual feature names from the model, use those
        if feature_names:
            for i, feature_name in enumerate(feature_names):
                # Try to map the feature name to our form data
                if feature_name in feature_mapping:
                    input_data[feature_name] = feature_mapping[feature_name]
                elif 'age' in feature_name.lower():
                    input_data[feature_name] = age
                elif 'smoker' in feature_name.lower():
                    input_data[feature_name] = is_smoker
                elif 'female' in feature_name.lower() or 'gender' in feature_name.lower():
                    input_data[feature_name] = is_female
                elif 'ctdna' in feature_name.lower() and 'egf' in feature_name.lower():
                    input_data[feature_name] = ctDNA_EGFR
                elif 'ctdna' in feature_name.lower() and 'kras' in feature_name.lower():
                    input_data[feature_name] = ctDNA_KRAS
                elif 'mirna' in feature_name.lower() and '21' in feature_name:
                    input_data[feature_name] = miRNA_21
                elif 'mirna' in feature_name.lower() and '15' in feature_name:
                    input_data[feature_name] = miRNA_155
                elif 'protein' in feature_name.lower() and 'cea' in feature_name.lower():
                    input_data[feature_name] = Protein_CEA
                elif 'protein' in feature_name.lower() and ('cyfra' in feature_name.lower() or 'cy' in feature_name.lower()):
                    input_data[feature_name] = Protein_CYFRA
                else:
                    # Default value for unmatched features
                    input_data[feature_name] = 0.0
        else:
            # Fallback if feature names not available
            input_data = feature_mapping
        
        # Make prediction using LSTM model
        result = predict_with_lstm(input_data)
        
        if result:
            prediction_text = f"Predicted Stage: {result['prediction']}"
            confidence_text = f"Confidence: {result['confidence']:.3f}"
            
            # Create detailed results text
            probabilities_text = "Stage Probabilities:\n"
            for stage, prob in result['probabilities'].items():
                probabilities_text += f"  {stage}: {prob:.3f}\n"
            
            full_result = f"{prediction_text}\n{confidence_text}\n\n{probabilities_text}"
            
            return render_template('indexproj.html', 
                                 prediction_text=full_result,
                                 predicted_stage=result['prediction'],
                                 confidence=result['confidence'],
                                 probabilities=result['probabilities'])
        else:
            return render_template('indexproj.html', 
                                 prediction_text="Error: Could not make prediction. Please check your inputs.")
    
    except Exception as e:
        error_message = f"Error processing prediction: {str(e)}"
        print(error_message)
        return render_template('indexproj.html', prediction_text=error_message)

@app.route('/predict_from_image', methods=['POST'])
def predict_from_image():
    """Handle image-based prediction requests"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            flash('No file selected')
            return redirect(url_for('indexproj'))
        
        file = request.files['image']
        
        # Check if file is valid
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('indexproj'))
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract data from image
            extracted_data = extract_biomarker_data(file_path)
            
            if extracted_data is None:
                return render_template('indexproj.html', 
                                     prediction_text="Error: Could not extract data from image. Please ensure the image is clear and contains biomarker data.",
                                     show_extracted_data=True,
                                     extracted_data={})
            
            # Fill in missing values with defaults or form data if provided - using safe conversion
            age = extracted_data.get('Age') if extracted_data.get('Age') is not None else safe_float_conversion(request.form.get('Age', ''), 50.0)
            gender = extracted_data.get('Gender') if extracted_data.get('Gender') is not None else request.form.get('Gender', 'Male')
            smoking_history = extracted_data.get('Smoking_History') if extracted_data.get('Smoking_History') is not None else request.form.get('Smoking_History', 'Never')
            
            # Biomarker data (prioritize extracted data, fallback to form data) - using safe conversion
            ctDNA_EGFR = extracted_data.get('ctDNA_EGFR') if extracted_data.get('ctDNA_EGFR') is not None else safe_float_conversion(request.form.get('ctDNA_EGFR', ''), 0.0)
            ctDNA_KRAS = extracted_data.get('ctDNA_KRAS') if extracted_data.get('ctDNA_KRAS') is not None else safe_float_conversion(request.form.get('ctDNA_KRAS', ''), 0.0)
            miRNA_21 = extracted_data.get('miRNA_21') if extracted_data.get('miRNA_21') is not None else safe_float_conversion(request.form.get('miRNA_21', ''), 0.0)
            miRNA_155 = extracted_data.get('miRNA_155') if extracted_data.get('miRNA_155') is not None else safe_float_conversion(request.form.get('miRNA_155', ''), 0.0)
            Protein_CEA = extracted_data.get('Protein_CEA') if extracted_data.get('Protein_CEA') is not None else safe_float_conversion(request.form.get('Protein_CEA', ''), 0.0)
            Protein_CYFRA = extracted_data.get('Protein_CYFRA') if extracted_data.get('Protein_CYFRA') is not None else safe_float_conversion(request.form.get('Protein_CYFRA', ''), 0.0)
            
            # Process categorical variables
            is_smoker = 1 if smoking_history.lower() != 'never' else 0
            is_female = 1 if gender.lower() == 'female' else 0
            
            # Prepare input data for prediction
            input_data = {
                'Age': age,
                'ctDNA_EGFR': ctDNA_EGFR,
                'ctDNA_KRAS': ctDNA_KRAS,
                'miRNA_21': miRNA_21,
                'miRNA_155': miRNA_155,
                'Protein_CEA': Protein_CEA,
                'Protein_CYFRA': Protein_CYFRA,
                'is_smoker': is_smoker,
                'is_female': is_female
            }
            
            # Make prediction
            result = predict_with_lstm(input_data)
            
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            if result:
                prediction_text = f"Predicted Stage: {result['prediction']}"
                confidence_text = f"Confidence: {result['confidence']:.3f}"
                
                # Create detailed results text
                probabilities_text = "Stage Probabilities:\n"
                for stage, prob in result['probabilities'].items():
                    probabilities_text += f"  {stage}: {prob:.3f}\n"
                
                full_result = f"{prediction_text}\n{confidence_text}\n\n{probabilities_text}"
                
                return render_template('indexproj.html', 
                                     prediction_text=full_result,
                                     predicted_stage=result['prediction'],
                                     confidence=result['confidence'],
                                     probabilities=result['probabilities'],
                                     show_extracted_data=True,
                                     extracted_data=extracted_data,
                                     used_data=input_data)
            else:
                return render_template('indexproj.html', 
                                     prediction_text="Error: Could not make prediction from extracted data.",
                                     show_extracted_data=True,
                                     extracted_data=extracted_data)
        else:
            flash('Invalid file format. Please upload an image file.')
            return redirect(url_for('indexproj'))
    
    except Exception as e:
        error_message = f"Error processing image prediction: {str(e)}"
        print(error_message)
        return render_template('indexproj.html', 
                             prediction_text=error_message,
                             show_extracted_data=True,
                             extracted_data={})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (returns JSON)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract data from JSON with safe conversion
        input_data = {
            'Age': safe_float_conversion(data.get('age', ''), 50.0),
            'is_smoker': 1 if data.get('smoking_history', 'never').lower() != 'never' else 0,
            'is_female': 1 if data.get('gender', 'male').lower() == 'female' else 0,
            'ctDNA_EGF': safe_float_conversion(data.get('ctDNA_EGFR', ''), 0.0),
            'ctDNA_KRAS': safe_float_conversion(data.get('ctDNA_KRAS', ''), 0.0),
            'miRNA_21': safe_float_conversion(data.get('miRNA_21', ''), 0.0),
            'miRNA_155': safe_float_conversion(data.get('miRNA_155', ''), 0.0),
            'Protein_CEA': safe_float_conversion(data.get('Protein_CEA', ''), 0.0),
            'Protein_CYFRA': safe_float_conversion(data.get('Protein_CYFRA', ''), 0.0)
        }
        
        result = predict_with_lstm(input_data)
        
        if result:
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities']
            })
        else:
            return jsonify({'success': False, 'error': 'Prediction failed'}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Get information about the loaded model"""
    global feature_names, sequence_length
    
    info = {
        'model_loaded': lstm_model is not None,
        'feature_names': feature_names if feature_names else [],
        'sequence_length': sequence_length if sequence_length else 0,
        'model_type': 'LSTM'
    }
    
    return jsonify(info)

if __name__ == '__main__':
    # Check if model is loaded
    if lstm_model is None:
        print("Warning: LSTM model not loaded. Please ensure 'lung_cancer_lstm_model.h5' and 'scaler.pkl' exist.")
        print("Run the LSTM training script first to generate these files.")
    else:
        print("LSTM model loaded successfully!")
        print(f"Feature names: {feature_names}")
        print(f"Sequence length: {sequence_length}")
    
    # Run the Flask app
    app.run(debug=True)