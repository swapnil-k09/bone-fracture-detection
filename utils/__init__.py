"""
Utilities Package for X-ray Bone Fracture Detection

This package contains various utility modules:
- preprocess: Image preprocessing with OpenCV
- data_loader: Dataset loading and organization
- augmentation: Data augmentation techniques
- visualization: Data visualization tools
- model_builder: CNN model architectures
- gradcam: Grad-CAM visualization for model interpretability
"""

from .preprocess import XRayPreprocessor, preprocess_single_image, preprocess_directory
from .data_loader import DatasetLoader, organize_mura_dataset
from .augmentation import XRayAugmenter, get_keras_augmentation_generator
from .visualization import XRayVisualizer, create_data_exploration_report
from .model_builder import FractureDetectionModel, create_callbacks, print_model_summary
from .gradcam import GradCAM, GradCAMPlusPlus, batch_visualize, compare_gradcam_methods

__all__ = [
    # Preprocessing
    'XRayPreprocessor',
    'preprocess_single_image',
    'preprocess_directory',
    
    # Data Loading
    'DatasetLoader',
    'organize_mura_dataset',
    
    # Augmentation
    'XRayAugmenter',
    'get_keras_augmentation_generator',
    
    # Visualization
    'XRayVisualizer',
    'create_data_exploration_report',
    
    # Model Building
    'FractureDetectionModel',
    'create_callbacks',
    'print_model_summary',
    
    # Grad-CAM
    'GradCAM',
    'GradCAMPlusPlus',
    'batch_visualize',
    'compare_gradcam_methods',
]

__version__ = '1.0.0'
