import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

text = pytesseract.image_to_string(Image.open("C://Users//ahmed//Desktop//LSTM//DATASETT//dataset (1).jpg"))
print(text)
