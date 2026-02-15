"""
Flask Web Application for X-ray Bone Fracture Detection
Main application file with routes and prediction logic
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
import json
import base64
from io import BytesIO
from PIL import Image

# Import our utilities
from utils.preprocess import preprocess_single_image
from utils.gradcam import GradCAM

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Global variables for model (loaded once)
model = None
gradcam = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_ai_model():
    """Load the trained model (called once at startup)"""
    global model, gradcam
    
    model_path = 'models/best_model.h5'
    
    if not os.path.exists(model_path):
        print("⚠️  WARNING: Model not found!")
        print(f"   Expected location: {model_path}")
        print("   The app will run in DEMO mode (random predictions)")
        return False
    
    try:
        print("📦 Loading AI model...")
        model = load_model(model_path)
        gradcam = GradCAM(model)
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def demo_prediction():
    """Generate demo prediction when model isn't available"""
    import random
    prediction = random.uniform(0.3, 0.95)
    return prediction

def process_xray(image_path):
    """
    Process X-ray image and generate prediction with Grad-CAM
    
    Args:
        image_path: Path to uploaded image
        
    Returns:
        Dictionary with results
    """
    try:
        # Load original image
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if original_image is None:
            return {'error': 'Could not load image'}
        
        # Preprocess for model
        preprocessed = preprocess_single_image(image_path, target_size=(224, 224))
        
        if preprocessed is None:
            return {'error': 'Could not preprocess image'}
        
        # Make prediction
        if model is not None:
            # Real prediction
            pred_input = np.expand_dims(preprocessed, axis=0)
            prediction = model.predict(pred_input, verbose=0)[0][0]
            
            # Generate Grad-CAM if fracture detected
            if prediction > 0.5:
                heatmap, overlay = gradcam.generate_visualization(
                    preprocessed,
                    original_image
                )
                
                # Save Grad-CAM visualization
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                gradcam_filename = f'gradcam_{timestamp}.png'
                gradcam_path = os.path.join('static/results', gradcam_filename)
                
                cv2.imwrite(gradcam_path, overlay)
                gradcam_url = f'/static/results/{gradcam_filename}'
            else:
                gradcam_url = None
        else:
            # Demo mode
            prediction = demo_prediction()
            gradcam_url = None
        
        # Determine result
        is_fractured = prediction > 0.5
        confidence = prediction if is_fractured else (1 - prediction)
        
        # Prepare result
        result = {
            'prediction': 'Fracture Detected' if is_fractured else 'Normal',
            'confidence': float(confidence * 100),
            'probability': float(prediction),
            'is_fractured': is_fractured,
            'gradcam_url': gradcam_url,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'demo_mode': model is None
        }
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process image
        result = process_xray(filepath)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Add file URL
        result['image_url'] = f'/uploads/{unique_filename}'
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/privacy')
def privacy():
    """Privacy Policy page"""
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    """Terms of Service page"""
    return render_template('terms.html')

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# Initialize model on startup
with app.app_context():
    model_loaded = load_ai_model()
    if not model_loaded:
        print("\n" + "="*70)
        print("⚠️  RUNNING IN DEMO MODE")
        print("="*70)
        print("Model not found. The app will generate random predictions.")
        print("To use real AI predictions:")
        print("  1. Train a model: python train.py")
        print("  2. Ensure best_model.h5 exists in models/")
        print("  3. Restart the app")
        print("="*70 + "\n")

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏥 X-RAY BONE FRACTURE DETECTION SYSTEM")
    print("="*70)
    print("Starting web server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
