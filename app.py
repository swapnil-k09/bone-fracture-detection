"""
Flask Web Application for X-ray Bone Fracture Detection
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime

from utils.preprocess import preprocess_single_image
from utils.gradcam import GradCAM

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

model = None
gradcam = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_ai_model():
    global model, gradcam
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path):
        return False
    try:
        model = load_model(model_path)
        gradcam = GradCAM(model)
        print("✅ Model and Grad-CAM loaded!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def process_xray(image_path):
    try:
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            return {'error': 'Could not load image'}
        
        preprocessed = preprocess_single_image(image_path, target_size=(224, 224))
        if preprocessed is None:
            return {'error': 'Could not preprocess image'}
        
        gradcam_url = None
        
        if model is not None:
            # Fix shape
            if len(preprocessed.shape) == 4:
                preprocessed = preprocessed.squeeze(axis=0)
            if preprocessed.shape == (1, 224, 224):
                preprocessed = np.transpose(preprocessed, (1, 2, 0))
            elif len(preprocessed.shape) == 2:
                preprocessed = np.expand_dims(preprocessed, axis=-1)
            
            # Predict
            pred_input = np.expand_dims(preprocessed, axis=0)
            prediction = model.predict(pred_input, verbose=0)[0][0]
            
            # Generate Grad-CAM for fractures
            if prediction < 0.5:
                try:
                    heatmap, overlay = gradcam.generate_visualization(preprocessed, original_image)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    gradcam_filename = f'gradcam_{timestamp}.png'
                    gradcam_path = os.path.join('static/results', gradcam_filename)
                    
                    cv2.imwrite(gradcam_path, overlay)
                    gradcam_url = f'/static/results/{gradcam_filename}'
                    print(f"✅ Grad-CAM saved: {gradcam_url}")
                except Exception as e:
                    print(f"⚠️ Grad-CAM failed: {e}")
                    gradcam_url = None
        else:
            import random
            prediction = random.uniform(0.3, 0.95)
        
        is_fractured = prediction < 0.5
        confidence = (1 - prediction) if is_fractured else prediction
        
        return {
            'prediction': 'Fracture Detected' if is_fractured else 'Normal',
            'confidence': float(confidence * 100),
            'probability': float(prediction),
            'is_fractured': bool(is_fractured),
            'gradcam_url': gradcam_url,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'demo_mode': model is None
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        result = process_xray(filepath)
        if 'error' in result:
            return jsonify(result), 500
        result['image_url'] = f'/uploads/{unique_filename}'
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/api/status')
def api_status():
    return jsonify({'status': 'online', 'model_loaded': model is not None, 'version': '1.0.0', 'timestamp': datetime.now().isoformat()})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

with app.app_context():
    load_ai_model()

if __name__ == '__main__':
    print("\n🏥 X-RAY FRACTURE DETECTION WITH GRAD-CAM")
    print("="*70)
    print("http://localhost:5000")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)