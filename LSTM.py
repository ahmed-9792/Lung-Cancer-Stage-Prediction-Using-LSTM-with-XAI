import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Deep Learning and ML libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# XAI libraries
import lime
import lime.lime_tabular
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class LungCancerLSTMPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.lime_explainer = None
        self.sequence_length = 3
        
    def auto_detect_columns(self, df):
        """Automatically detect column names from the dataset"""
        column_mapping = {}
        
        # Map common variations of column names
        for col in df.columns:
            col_lower = col.lower()
            
            # ctDNA columns
            if 'ctdna' in col_lower and 'egf' in col_lower:
                column_mapping['ctDNA_EGF'] = col
            elif 'ctdna' in col_lower and 'kras' in col_lower:
                column_mapping['ctDNA_KRAS'] = col
            
            # miRNA columns
            elif 'mirna' in col_lower and '21' in col:
                column_mapping['miRNA_21'] = col
            elif 'mirna' in col_lower and '15' in col:
                column_mapping['miRNA_155'] = col
            
            # Protein columns
            elif 'protein' in col_lower and 'cea' in col_lower:
                column_mapping['Protein_CEA'] = col
            elif 'protein' in col_lower and ('cyfra' in col_lower or 'cy' in col_lower):
                column_mapping['Protein_CYFRA'] = col
            
            # Basic columns
            elif col_lower == 'age':
                column_mapping['Age'] = col
            elif 'gender' in col_lower:
                column_mapping['Gender'] = col
            elif 'smoking' in col_lower:
                column_mapping['Smoking'] = col
            elif 'cancer' in col_lower and 'status' in col_lower:
                column_mapping['Cancer_Status'] = col
        
        return column_mapping
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Print column names to debug
        print("Available columns:", df.columns.tolist())
        
        # Auto-detect column names
        column_mapping = self.auto_detect_columns(df)
        print("Detected column mapping:", column_mapping)
        
        # Find target column
        target_col = None
        for col in df.columns:
            if 'cancer' in col.lower() and ('status' in col.lower() or 'stage' in col.lower()):
                target_col = col
                break
        
        if target_col is None:
            # Try to find any column that might be the target
            possible_targets = [col for col in df.columns if any(keyword in col.lower() 
                              for keyword in ['cancer', 'stage', 'diagnosis', 'class', 'label'])]
            if possible_targets:
                target_col = possible_targets[0]
            else:
                raise ValueError("Could not find target column. Please specify manually.")
        
        print(f"Using target column: {target_col}")
        
        # Handle missing values and clean data
        df = df.dropna(subset=[target_col])  # Remove rows without target
        
        # Find numeric columns (biomarker data)
        numeric_cols = []
        for col in df.columns:
            if col != target_col and df[col].dtype in ['float64', 'int64', 'object']:
                # Check if column contains numeric data
                try:
                    pd.to_numeric(df[col].replace('?', np.nan), errors='raise')
                    numeric_cols.append(col)
                except:
                    pass
        
        print("Using numeric columns:", numeric_cols)
        
        # Replace '?' with NaN and fill with median for numeric columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace('?', np.nan), errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # Create binary features - check for correct column names
        smoking_cols = [col for col in df.columns if 'smoking' in col.lower()]
        gender_cols = [col for col in df.columns if 'gender' in col.lower()]
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        
        feature_cols = numeric_cols.copy()
        
        if smoking_cols:
            smoking_col = smoking_cols[0]
            df['is_smoker'] = (~df[smoking_col].str.contains('Never', case=False, na=False)).astype(int)
            feature_cols.append('is_smoker')
        
        if gender_cols:
            gender_col = gender_cols[0]
            df['is_female'] = (df[gender_col].str.contains('Female', case=False, na=False)).astype(int)
            feature_cols.append('is_female')
        
        if age_cols:
            age_col = age_cols[0]
            df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
            df[age_col] = df[age_col].fillna(df[age_col].median())
            if age_col not in feature_cols:
                feature_cols.append(age_col)
        
        # Filter only existing columns
        existing_feature_cols = [col for col in feature_cols if col in df.columns]
        print("Final feature columns:", existing_feature_cols)
        
        X = df[existing_feature_cols].values
        
        # Prepare target variable
        # Map cancer stages to numeric categories
        unique_targets = df[target_col].unique()
        print(f"Unique values in {target_col}:", unique_targets)
        
        cancer_mapping = {}
        for i, target in enumerate(sorted(unique_targets)):
            target_str = str(target).lower()
            if any(keyword in target_str for keyword in ['no', 'negative', 'normal', 'benign']):
                cancer_mapping[target] = 0
            elif 'i' in target_str and 'ii' not in target_str and 'iv' not in target_str:
                cancer_mapping[target] = 1
            elif 'ii' in target_str and 'iii' not in target_str:
                cancer_mapping[target] = 2
            elif 'iii' in target_str and 'iv' not in target_str:
                cancer_mapping[target] = 3
            elif 'iv' in target_str:
                cancer_mapping[target] = 4
            else:
                # For any other positive cases, assign to Stage I
                cancer_mapping[target] = 1 if i > 0 else 0
        
        print("Cancer mapping:", cancer_mapping)
        
        df['cancer_category'] = df[target_col].map(cancer_mapping)
        
        # Handle unmapped values
        unmapped_mask = df['cancer_category'].isna()
        if unmapped_mask.sum() > 0:
            print(f"Warning: {unmapped_mask.sum()} unmapped cancer status values found")
            print("Unmapped values:", df[unmapped_mask][target_col].unique())
            # Remove unmapped values
            df = df.dropna(subset=['cancer_category'])
        
        y = df['cancer_category'].values.astype(int)
        
        self.feature_names = existing_feature_cols
        print(f"Dataset loaded: {len(X)} samples, {len(existing_feature_cols)} features")
        print(f"Cancer distribution: {np.bincount(y)}")
        print(f"Class labels: {['No Cancer', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']}")
        
        return X, y, df
    
    def add_noise_augmentation(self, X, y, noise_factor=0.1, augment_factor=2):
        """Add noise to training data to improve model robustness and accuracy"""
        print(f"Adding noise augmentation with factor {noise_factor}...")
        
        X_augmented = []
        y_augmented = []
        
        # Original data
        X_augmented.append(X)
        y_augmented.append(y)
        
        # Add augmented data
        for i in range(augment_factor):
            # Add Gaussian noise
            noise = np.random.normal(0, noise_factor * np.std(X, axis=0), X.shape)
            X_noisy = X + noise
            
            # Add some systematic variations
            if i == 1:
                # Scale variation
                scale_factor = np.random.uniform(0.95, 1.05, X.shape[1])
                X_noisy = X * scale_factor
            
            X_augmented.append(X_noisy)
            y_augmented.append(y)
        
        X_final = np.vstack(X_augmented)
        y_final = np.hstack(y_augmented)
        
        print(f"Data augmented from {len(X)} to {len(X_final)} samples")
        return X_final, y_final
    
    def create_sequences(self, X, y):
        """Create sequences for LSTM input"""
        print("Creating sequences for LSTM...")
        
        # For each sample, create a sequence by duplicating and adding slight variations
        X_seq = []
        y_seq = []
        
        for i in range(len(X)):
            sequence = []
            base_sample = X[i]
            
            # Create sequence by adding temporal variations
            for j in range(self.sequence_length):
                if j == 0:
                    sequence.append(base_sample)
                else:
                    # Add slight temporal variation
                    variation = base_sample * (1 + np.random.normal(0, 0.02, len(base_sample)))
                    sequence.append(variation)
            
            X_seq.append(sequence)
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape, num_classes):
        """Build LSTM model architecture"""
        print("Building LSTM model...")
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape, dropout=0.2),
            BatchNormalization(),
            LSTM(64, return_sequences=False, dropout=0.2),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, file_path):
        """Complete training pipeline"""
        print("Starting training pipeline...")
        
        # Load and preprocess data
        X, y, df = self.load_and_preprocess_data(file_path)
        
        # Apply noise augmentation
        X_aug, y_aug = self.add_noise_augmentation(X, y, noise_factor=0.08, augment_factor=3)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_aug)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_aug)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
        )
        
        # Build model
        num_classes = len(np.unique(y_seq))
        input_shape = (self.sequence_length, X_scaled.shape[1])
        self.model = self.build_model(input_shape, num_classes)
        
        print("Model architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7)
        ]
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=16,
            epochs=25,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, np.argmax(train_pred, axis=1))
        test_accuracy = accuracy_score(y_test, np.argmax(test_pred, axis=1))
        
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, np.argmax(test_pred, axis=1)))
        
        # Setup LIME explainer
        self.setup_lime_explainer(X_scaled[:1000])  # Use subset for explainer
        
        return history, test_accuracy
    
    def setup_lime_explainer(self, X_sample):
        """Setup LIME explainer for XAI"""
        print("Setting up LIME explainer...")
        
        # Reshape for LIME (it expects 2D input)
        X_reshaped = X_sample.reshape(X_sample.shape[0], -1)
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_reshaped,
            feature_names=[f"{feat}_{i}" for feat in self.feature_names for i in range(self.sequence_length)],
            class_names=['No Cancer', 'Stage I', 'Stage II', 'Stage III', 'Stage IV'],
            mode='classification'
        )
    
    def explain_prediction(self, X_sample):
        """Generate LIME explanation for a prediction"""
        if self.lime_explainer is None:
            print("LIME explainer not initialized. Please train the model first.")
            return None
        
        # Reshape sample for prediction
        X_seq_sample = X_sample.reshape(1, self.sequence_length, -1)
        
        # Make prediction
        prediction = self.model.predict(X_seq_sample)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Prepare for LIME
        X_flat = X_sample.reshape(1, -1)
        
        def predict_fn(X_flat_batch):
            # Reshape back to sequence format
            batch_size = X_flat_batch.shape[0]
            X_seq_batch = X_flat_batch.reshape(batch_size, self.sequence_length, -1)
            return self.model.predict(X_seq_batch)
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            X_flat[0], predict_fn, num_features=10
        )
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'explanation': explanation
        }
    
    def save_model(self, model_path='lung_cancer_lstm_model.h5', scaler_path='scaler.pkl'):
        """Save trained model and preprocessors"""
        print("Saving model and preprocessors...")
        
        # Save model
        self.model.save(model_path)
        
        # Save preprocessors
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'sequence_length': self.sequence_length
            }, f)
        
        print(f"Model saved to {model_path}")
        print(f"Preprocessors saved to {scaler_path}")
    
    def load_model(self, model_path='lung_cancer_lstm_model.h5', scaler_path='scaler.pkl'):
        """Load trained model and preprocessors"""
        print("Loading model and preprocessors...")
        
        # Load model
        self.model = load_model(model_path)
        
        # Load preprocessors
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.sequence_length = data['sequence_length']
        
        print("Model and preprocessors loaded successfully!")
    
    def predict_new_sample(self, input_data):
        """Predict cancer stage for new patient data"""
        if self.model is None:
            print("Model not loaded. Please load or train a model first.")
            return None
        
        # Ensure input is in correct format
        if isinstance(input_data, dict):
            # Convert dictionary to array based on feature names
            input_array = []
            for feature in self.feature_names:
                if feature in input_data:
                    input_array.append(input_data[feature])
                else:
                    print(f"Missing feature: {feature}")
                    return None
            input_data = np.array(input_array)
        
        # Scale input
        input_scaled = self.scaler.transform(input_data.reshape(1, -1))
        
        # Create sequence
        input_seq = []
        for i in range(self.sequence_length):
            if i == 0:
                input_seq.append(input_scaled[0])
            else:
                # Add slight variation
                variation = input_scaled[0] * (1 + np.random.normal(0, 0.01, len(input_scaled[0])))
                input_seq.append(variation)
        
        input_seq = np.array(input_seq).reshape(1, self.sequence_length, -1)
        
        # Make prediction
        prediction = self.model.predict(input_seq)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Map prediction to label
        class_labels = ['No Cancer', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
        predicted_label = class_labels[predicted_class]
        
        return {
            'prediction': predicted_label,
            'confidence': confidence,
            'probabilities': {class_labels[i]: prediction[0][i] for i in range(len(class_labels))}
        }

def interactive_prediction_interface():
    """Interactive interface for making predictions"""
    print("\n" + "="*60)
    print("LUNG CANCER STAGE PREDICTION SYSTEM")
    print("="*60)
    
    # Initialize predictor
    predictor = LungCancerLSTMPredictor()
    
    # Try to load existing model
    try:
        predictor.load_model()
        print("Existing model loaded successfully!")
    except:
        print("No existing model found. Please train a new model first.")
        return
    
    while True:
        print("\nOptions:")
        print("1. Predict cancer stage for new patient")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            print("\nEnter patient information:")
            print("(Enter values for available features)")
            
            try:
                # Collect patient data dynamically based on feature names
                patient_data = {}
                print(f"Required features: {predictor.feature_names}")
                
                for feature in predictor.feature_names:
                    value = input(f"{feature}: ")
                    try:
                        patient_data[feature] = float(value)
                    except ValueError:
                        print(f"Invalid value for {feature}. Using 0.")
                        patient_data[feature] = 0.0
                
                # Make prediction
                result = predictor.predict_new_sample(patient_data)
                
                if result:
                    print(f"\n{'='*50}")
                    print("PREDICTION RESULTS")
                    print(f"{'='*50}")
                    print(f"Predicted Stage: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.3f}")
                    print("\nAll Probabilities:")
                    for stage, prob in result['probabilities'].items():
                        print(f"  {stage}: {prob:.3f}")
                    print(f"{'='*50}")
                
            except Exception as e:
                print(f"Error making prediction: {e}")
        
        elif choice == '2':
            print("Thank you for using the Lung Cancer Prediction System!")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

def main():
    """Main function to train model and start interface"""
    print("LUNG CANCER LSTM PREDICTION SYSTEM")
    print("="*50)
    
    # File path
    file_path = "C://Users//ahmed//Desktop//LSTM//NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22.csv"
    
    # Initialize predictor
    predictor = LungCancerLSTMPredictor()
    
    # Check if model exists
    model_exists = False
    try:
        predictor.load_model()
        print("Existing model found and loaded!")
        model_exists = True
        
        choice = input("Do you want to retrain the model? (y/n): ").strip().lower()
        if choice != 'y':
            # Start interactive interface with existing model
            interactive_prediction_interface()
            return
    except Exception as e:
        print(f"No existing model found or error loading: {e}")
        print("Training new model...")
    
    # Train model
    try:
        history, accuracy = predictor.train_model(file_path)
        
        # Save model
        predictor.save_model()
        print(f"Model trained and saved with accuracy: {accuracy:.4f}")
        
        # Start interactive interface
        interactive_prediction_interface()
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Please check your data file and column names.")

if __name__ == "__main__":
    # Run the main function  
    main()