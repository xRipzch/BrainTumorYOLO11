# app.py
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\ander\Desktop\Dat23A Kea\Valgfag\Machine Learning\BrainTumorYOLO11\BrainTumorYOLO11\BrainTumor\brain-tumor-detector.keras')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read and preprocess the image
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).resize((224, 224))  # Adjust size as needed
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run model inference
    predictions = model.predict(img_array)
    
    # Process predictions (this depends on your model)
    result = predictions.tolist()  # Or any processing to extract detection info
    return jsonify({'predictions': result})

if __name__ == '__main__':
    app.run(debug=True)