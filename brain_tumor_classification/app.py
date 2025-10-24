from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Author information
AUTHOR_INFO = {
    'name': 'Parthiv Meduri',
    'title': 'AI/ML Developer & Researcher',
    'bio': 'Passionate about leveraging deep learning and computer vision to solve real-world healthcare challenges. Currently focusing on medical image analysis and diagnostic AI systems.',
    'skills': ['Deep Learning', 'Computer Vision', 'TensorFlow', 'Python', 'Medical AI'],
    'contact': {
        'email': 'parthiv.meduri@example.com',
        'github': 'https://github.com/Parthiv19M',
        'linkedin': 'https://linkedin.com/in/parthivmeduri'
    }
}

# Model configuration
MODEL_CLASSES = ['Glioma', 'Meningioma', 'Pituitary Tumor', 'No Tumor']
MODEL_PATH = 'models/brain_tumor_classifier.h5'

@app.route('/')
def home():
    return render_template('index.html', author=AUTHOR_INFO, classes=MODEL_CLASSES)

@app.route('/about')
def about():
    return render_template('about.html', author=AUTHOR_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Load and preprocess image
            img = Image.open(filepath).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Load model and make prediction
            model = tf.keras.models.load_model(MODEL_PATH)
            predictions = model.predict(img_array)
            predicted_class = MODEL_CLASSES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))

            # Clean up
            os.remove(filepath)

            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': {MODEL_CLASSES[i]: float(predictions[0][i])
                                for i in range(len(MODEL_CLASSES))}
            })

        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
