"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Visualizes which regions of X-ray images the CNN focuses on for predictions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import os


class GradCAM:
    """
    Grad-CAM visualization for CNN models
    Highlights important regions in images that influence model predictions
    """
    
    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of convolutional layer to visualize (auto-detected if None)
        """
        self.model = model
        self.layer_name = layer_name
        
        if self.layer_name is None:
            self.layer_name = self._find_target_layer()
        
        print(f"✅ Grad-CAM initialized with layer: {self.layer_name}")
    
    def _find_target_layer(self) -> str:
        """
        Automatically find the last convolutional layer
        
        Returns:
            Name of the last conv layer
        """
        # Search for last convolutional layer
        for layer in reversed(self.model.layers):
            # Check if layer has 4D output (conv layer)
            if len(layer.output.shape) == 4:
                return layer.name
        
        raise ValueError("Could not find convolutional layer in model")
    
    def compute_heatmap(self, 
                       image: np.ndarray,
                       class_idx: Optional[int] = None,
                       eps: float = 1e-8) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for an image
        
        Args:
            image: Input image (preprocessed)
            class_idx: Target class index (None for predicted class)
            eps: Small constant for numerical stability
            
        Returns:
            Heatmap as numpy array
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Create gradient model
        grad_model = keras.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            
            # If class_idx not specified, use predicted class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get output for target class
            if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                # Binary classification with sigmoid
                class_output = predictions[:, 0]
            else:
                # Multi-class with softmax
                class_output = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight conv outputs by gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap = heatmap / (np.max(heatmap) + eps)  # Normalize to [0, 1]
        
        return heatmap
    
    def overlay_heatmap(self,
                       heatmap: np.ndarray,
                       image: np.ndarray,
                       alpha: float = 0.4,
                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap
            image: Original image
            alpha: Transparency of overlay (0-1)
            colormap: OpenCV colormap to use
            
        Returns:
            Overlaid image
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Convert grayscale image to RGB if needed
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = np.uint8(255 * image)
        
        # Overlay heatmap
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlaid
    
    def generate_visualization(self,
                             image: np.ndarray,
                             original_image: Optional[np.ndarray] = None,
                             class_idx: Optional[int] = None,
                             alpha: float = 0.4,
                             colormap: int = cv2.COLORMAP_JET) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete Grad-CAM visualization
        
        Args:
            image: Preprocessed image for model
            original_image: Original image for overlay (if different from preprocessed)
            class_idx: Target class index
            alpha: Overlay transparency
            colormap: Colormap for heatmap
            
        Returns:
            Tuple of (heatmap, overlaid_image)
        """
        # Compute heatmap
        heatmap = self.compute_heatmap(image, class_idx)
        
        # Use original image if provided, otherwise use preprocessed
        if original_image is None:
            original_image = image.squeeze() if len(image.shape) == 4 else image
        
        # Overlay heatmap
        overlaid = self.overlay_heatmap(heatmap, original_image, alpha, colormap)
        
        return heatmap, overlaid
    
    def visualize(self,
                 image: np.ndarray,
                 original_image: Optional[np.ndarray] = None,
                 prediction: Optional[float] = None,
                 class_names: list = ['Normal', 'Fractured'],
                 save_path: Optional[str] = None,
                 show: bool = True):
        """
        Create and display complete visualization with predictions
        
        Args:
            image: Preprocessed image
            original_image: Original image
            prediction: Model prediction probability
            class_names: List of class names
            save_path: Path to save visualization
            show: Whether to display visualization
        """
        # Generate Grad-CAM
        heatmap, overlaid = self.generate_visualization(image, original_image)
        
        # Get prediction if not provided
        if prediction is None:
            if len(image.shape) == 3:
                image_input = np.expand_dims(image, axis=0)
            else:
                image_input = image
            prediction = self.model.predict(image_input, verbose=0)[0][0]
        
        # Determine predicted class
        predicted_class = 1 if prediction > 0.5 else 0
        confidence = prediction if predicted_class == 1 else (1 - prediction)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if original_image is not None:
            display_img = original_image
        else:
            display_img = image.squeeze() if len(image.shape) == 4 else image
        
        axes[0].imshow(display_img, cmap='gray')
        axes[0].set_title('Original X-ray', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlaid)
        axes[2].set_title('Highlighted Regions', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Add prediction info
        pred_text = f"Prediction: {class_names[predicted_class]}\n"
        pred_text += f"Confidence: {confidence*100:.1f}%"
        
        color = 'red' if predicted_class == 1 else 'green'
        fig.suptitle(pred_text, fontsize=14, fontweight='bold', color=color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ - Improved version with weighted combination of gradients
    Better localization for multiple objects
    """
    
    def compute_heatmap(self,
                       image: np.ndarray,
                       class_idx: Optional[int] = None,
                       eps: float = 1e-8) -> np.ndarray:
        """
        Compute Grad-CAM++ heatmap
        
        Args:
            image: Input image
            class_idx: Target class index
            eps: Small constant
            
        Returns:
            Heatmap
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Create gradient model
        grad_model = keras.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                class_output = predictions[:, 0]
            else:
                class_output = predictions[:, class_idx]
        
        # First order gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        # Second order gradients
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_outputs_2, predictions_2 = grad_model(image)
                if len(predictions_2.shape) == 2 and predictions_2.shape[1] == 1:
                    class_output_2 = predictions_2[:, 0]
                else:
                    class_output_2 = predictions_2[:, class_idx]
            grads_2 = tape3.gradient(class_output_2, conv_outputs_2)
        grads_3 = tape2.gradient(grads_2, conv_outputs_2)
        
        # Calculate alpha weights
        grads = grads[0]
        grads_2 = grads_2[0] if grads_2 is not None else grads
        grads_3 = grads_3[0] if grads_3 is not None else grads_2
        
        # Global sum pooling
        alpha_denom = grads_2 * 2.0 + grads_3 * conv_outputs[0]
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, eps)
        
        alphas = grads_2 / alpha_denom
        
        # Weight importance
        weights = np.maximum(grads, 0.0)
        alpha_norm = alphas / (np.sum(alphas, axis=(0, 1), keepdims=True) + eps)
        
        # Deep feature importance
        deep_linearization = np.sum(weights * alpha_norm, axis=(0, 1))
        
        # Create heatmap
        conv_outputs = conv_outputs[0]
        heatmap = np.sum(deep_linearization * conv_outputs, axis=-1)
        
        # Normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + eps)
        
        return heatmap


def batch_visualize(model: keras.Model,
                   images: list,
                   original_images: Optional[list] = None,
                   layer_name: Optional[str] = None,
                   output_dir: str = 'gradcam_outputs',
                   class_names: list = ['Normal', 'Fractured']):
    """
    Generate Grad-CAM visualizations for multiple images
    
    Args:
        model: Trained model
        images: List of preprocessed images
        original_images: List of original images
        layer_name: Target layer name
        output_dir: Output directory
        class_names: Class names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    gradcam = GradCAM(model, layer_name)
    
    print(f"\n🎨 Generating Grad-CAM visualizations for {len(images)} images...")
    
    for i, image in enumerate(images):
        # Get original image
        orig_img = original_images[i] if original_images else None
        
        # Get prediction
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0]
        
        # Generate visualization
        save_path = os.path.join(output_dir, f'gradcam_{i+1}.png')
        gradcam.visualize(
            image,
            original_image=orig_img,
            prediction=pred,
            class_names=class_names,
            save_path=save_path,
            show=False
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(images)}")
    
    print(f"✅ All visualizations saved to: {output_dir}/")


def compare_gradcam_methods(model: keras.Model,
                           image: np.ndarray,
                           original_image: Optional[np.ndarray] = None,
                           layer_name: Optional[str] = None,
                           save_path: Optional[str] = None):
    """
    Compare Grad-CAM and Grad-CAM++ side by side
    
    Args:
        model: Trained model
        image: Preprocessed image
        original_image: Original image
        layer_name: Target layer
        save_path: Save path
    """
    # Generate both heatmaps
    gradcam = GradCAM(model, layer_name)
    gradcam_pp = GradCAMPlusPlus(model, layer_name)
    
    heatmap_cam, overlay_cam = gradcam.generate_visualization(image, original_image)
    heatmap_pp, overlay_pp = gradcam_pp.generate_visualization(image, original_image)
    
    # Display
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    if original_image is not None:
        display_img = original_image
    else:
        display_img = image.squeeze()
    
    # Row 1: Grad-CAM
    axes[0, 0].imshow(heatmap_cam, cmap='jet')
    axes[0, 0].set_title('Grad-CAM Heatmap', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(overlay_cam)
    axes[0, 1].set_title('Grad-CAM Overlay', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Row 2: Grad-CAM++
    axes[1, 0].imshow(heatmap_pp, cmap='jet')
    axes[1, 0].set_title('Grad-CAM++ Heatmap', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(overlay_pp)
    axes[1, 1].set_title('Grad-CAM++ Overlay', fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle('Grad-CAM vs Grad-CAM++ Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comparison saved: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    print("="*70)
    print("GRAD-CAM VISUALIZATION MODULE")
    print("="*70 + "\n")
    
    print("This module provides Grad-CAM visualization for CNN models.")
    print("\nFeatures:")
    print("  ✓ Grad-CAM - Standard gradient-based visualization")
    print("  ✓ Grad-CAM++ - Improved localization")
    print("  ✓ Automatic layer detection")
    print("  ✓ Batch processing")
    print("  ✓ Customizable overlays")
    
    print("\nUsage:")
    print("  from utils.gradcam import GradCAM")
    print("  gradcam = GradCAM(model)")
    print("  gradcam.visualize(image)")
    
    print("\n" + "="*70)
