"""
Configuration settings for Bone Fracture Detection System
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.h5')

# Upload paths
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Image settings
IMAGE_SIZE = (224, 224)  # Input size for the model
IMAGE_CHANNELS = 1  # Grayscale
COLOR_MODE = 'grayscale'

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Data augmentation parameters
AUGMENTATION_PARAMS = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'fill_mode': 'nearest'
}

# Model architecture
MODEL_TYPE = 'densenet121'  # Options: 'custom', 'vgg16', 'resnet50', 'densenet121'
DROPOUT_RATE = 0.5
DENSE_UNITS = 512

# Training callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Grad-CAM settings
GRADCAM_LAYER_NAME = 'conv5_block16_concat'  # For DenseNet121
GRADCAM_ALPHA = 0.4  # Overlay transparency

# Web application settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# File upload settings
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Prediction threshold
FRACTURE_THRESHOLD = 0.5  # Confidence threshold for fracture detection

# Class names
CLASS_NAMES = ['Normal', 'Fractured']

# Logging
LOG_DIR = os.path.join(BASE_DIR, 'logs')
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, 'tensorboard')

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# GPU settings
import tensorflow as tf

# Configure GPU memory growth to avoid OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU(s) detected: {len(gpus)}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("ℹ️ No GPU detected, using CPU")

# Environment check
def check_environment():
    """Check if the environment is properly set up"""
    print("\n" + "="*50)
    print("ENVIRONMENT CHECK")
    print("="*50)
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version}")
    
    # Check TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    
    # Check GPU
    print(f"GPU Available: {len(gpus) > 0}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    
    # Check directories
    print(f"\nData directory exists: {os.path.exists(DATA_DIR)}")
    print(f"Train directory exists: {os.path.exists(TRAIN_DIR)}")
    print(f"Validation directory exists: {os.path.exists(VALIDATION_DIR)}")
    print(f"Test directory exists: {os.path.exists(TEST_DIR)}")
    
    print("="*50 + "\n")

if __name__ == '__main__':
    check_environment()
