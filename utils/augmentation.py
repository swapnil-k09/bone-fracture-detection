"""
Data Augmentation for X-ray Images
Increases dataset size and improves model generalization
"""

import cv2
import numpy as np
from typing import Tuple, List
import random


class XRayAugmenter:
    """
    Data augmentation specifically designed for X-ray images
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize augmenter
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
    
    def rotate(self, image: np.ndarray, angle: float = None,
               angle_range: Tuple[float, float] = (-20, 20)) -> np.ndarray:
        """
        Rotate image
        
        Args:
            image: Input image
            angle: Specific angle to rotate (if None, random within range)
            angle_range: Range for random rotation
            
        Returns:
            Rotated image
        """
        if angle is None:
            angle = np.random.uniform(angle_range[0], angle_range[1])
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, matrix, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        
        return rotated
    
    def flip(self, image: np.ndarray, 
            horizontal: bool = True,
            vertical: bool = False) -> np.ndarray:
        """
        Flip image
        
        Args:
            image: Input image
            horizontal: Apply horizontal flip
            vertical: Apply vertical flip
            
        Returns:
            Flipped image
        """
        if horizontal and np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        if vertical and np.random.random() > 0.5:
            image = cv2.flip(image, 0)
        
        return image
    
    def shift(self, image: np.ndarray,
             width_shift: float = None,
             height_shift: float = None,
             shift_range: float = 0.2) -> np.ndarray:
        """
        Shift image horizontally and/or vertically
        
        Args:
            image: Input image
            width_shift: Horizontal shift (fraction of width)
            height_shift: Vertical shift (fraction of height)
            shift_range: Maximum shift range if specific shifts not provided
            
        Returns:
            Shifted image
        """
        h, w = image.shape[:2]
        
        if width_shift is None:
            width_shift = np.random.uniform(-shift_range, shift_range)
        if height_shift is None:
            height_shift = np.random.uniform(-shift_range, shift_range)
        
        # Calculate pixel shifts
        tx = int(width_shift * w)
        ty = int(height_shift * h)
        
        # Create translation matrix
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply translation
        shifted = cv2.warpAffine(image, matrix, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        
        return shifted
    
    def zoom(self, image: np.ndarray,
            zoom_factor: float = None,
            zoom_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Zoom in/out on image
        
        Args:
            image: Input image
            zoom_factor: Specific zoom factor (if None, random within range)
            zoom_range: Range for random zoom
            
        Returns:
            Zoomed image
        """
        if zoom_factor is None:
            zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
        
        h, w = image.shape[:2]
        
        # Calculate new dimensions
        new_h = int(h * zoom_factor)
        new_w = int(w * zoom_factor)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Crop or pad to original size
        if zoom_factor > 1:  # Zoom in (crop)
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            cropped = resized[start_h:start_h+h, start_w:start_w+w]
            return cropped
        else:  # Zoom out (pad)
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            padded = cv2.copyMakeBorder(resized, pad_h, h-new_h-pad_h,
                                       pad_w, w-new_w-pad_w,
                                       cv2.BORDER_CONSTANT, value=0)
            return padded
    
    def adjust_brightness(self, image: np.ndarray,
                         factor: float = None,
                         factor_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Adjust image brightness
        
        Args:
            image: Input image
            factor: Brightness factor (if None, random within range)
            factor_range: Range for random brightness
            
        Returns:
            Brightness-adjusted image
        """
        if factor is None:
            factor = np.random.uniform(factor_range[0], factor_range[1])
        
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted
    
    def adjust_contrast(self, image: np.ndarray,
                       factor: float = None,
                       factor_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Adjust image contrast
        
        Args:
            image: Input image
            factor: Contrast factor (if None, random within range)
            factor_range: Range for random contrast
            
        Returns:
            Contrast-adjusted image
        """
        if factor is None:
            factor = np.random.uniform(factor_range[0], factor_range[1])
        
        # Adjust contrast
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted
    
    def add_noise(self, image: np.ndarray,
                 noise_type: str = 'gaussian',
                 intensity: float = 0.01) -> np.ndarray:
        """
        Add noise to image
        
        Args:
            image: Input image
            noise_type: Type of noise ('gaussian' or 'salt_pepper')
            intensity: Noise intensity
            
        Returns:
            Noisy image
        """
        if noise_type == 'gaussian':
            # Gaussian noise
            noise = np.random.normal(0, intensity * 255, image.shape)
            noisy = image + noise
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            noisy = image.copy()
            # Salt
            salt = np.random.random(image.shape) < intensity
            noisy[salt] = 255
            # Pepper
            pepper = np.random.random(image.shape) < intensity
            noisy[pepper] = 0
            
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return noisy
    
    def elastic_transform(self, image: np.ndarray,
                         alpha: float = 34,
                         sigma: float = 4) -> np.ndarray:
        """
        Apply elastic deformation
        Useful for simulating tissue deformation
        
        Args:
            image: Input image
            alpha: Deformation intensity
            sigma: Smoothness of deformation
            
        Returns:
            Elastically transformed image
        """
        h, w = image.shape[:2]
        
        # Random displacement fields
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        
        # Create mesh grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Add displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Remap
        transformed = cv2.remap(image, map_x, map_y, 
                              interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
        
        return transformed
    
    def augment_pipeline(self, image: np.ndarray,
                        augmentation_config: dict = None) -> np.ndarray:
        """
        Apply a pipeline of augmentations
        
        Args:
            image: Input image
            augmentation_config: Configuration for augmentations
            
        Returns:
            Augmented image
        """
        if augmentation_config is None:
            # Default configuration
            augmentation_config = {
                'rotate': True,
                'flip': True,
                'shift': True,
                'zoom': True,
                'brightness': False,  # Be careful with medical images
                'contrast': False,    # Be careful with medical images
                'noise': False,
                'elastic': False
            }
        
        augmented = image.copy()
        
        # Apply augmentations
        if augmentation_config.get('rotate'):
            augmented = self.rotate(augmented)
        
        if augmentation_config.get('flip'):
            augmented = self.flip(augmented, horizontal=True, vertical=False)
        
        if augmentation_config.get('shift'):
            augmented = self.shift(augmented)
        
        if augmentation_config.get('zoom'):
            augmented = self.zoom(augmented)
        
        if augmentation_config.get('brightness'):
            augmented = self.adjust_brightness(augmented)
        
        if augmentation_config.get('contrast'):
            augmented = self.adjust_contrast(augmented)
        
        if augmentation_config.get('noise'):
            augmented = self.add_noise(augmented, intensity=0.005)
        
        if augmentation_config.get('elastic'):
            augmented = self.elastic_transform(augmented)
        
        return augmented
    
    def generate_augmented_dataset(self, images: List[np.ndarray],
                                  labels: List[int],
                                  augmentations_per_image: int = 3) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate augmented dataset
        
        Args:
            images: List of original images
            labels: List of labels
            augmentations_per_image: Number of augmentations per image
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []
        
        print(f"Generating {augmentations_per_image} augmentations per image...")
        
        for i, (img, label) in enumerate(zip(images, labels)):
            if i % 100 == 0:
                print(f"Processing: {i}/{len(images)}")
            
            # Add original
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Generate augmentations
            for _ in range(augmentations_per_image):
                aug_img = self.augment_pipeline(img)
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        print(f"✅ Generated {len(augmented_images)} images from {len(images)} originals")
        
        return augmented_images, augmented_labels
    
    def visualize_augmentations(self, image: np.ndarray, 
                               num_examples: int = 6,
                               save_path: str = None):
        """
        Visualize different augmentations
        
        Args:
            image: Original image
            num_examples: Number of augmentation examples to show
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Generate augmentations
        augmented = [image]  # Original first
        for _ in range(num_examples - 1):
            aug = self.augment_pipeline(image)
            augmented.append(aug)
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('X-ray Data Augmentation Examples', fontsize=14, fontweight='bold')
        
        axes = axes.ravel()
        titles = ['Original'] + [f'Augmented {i+1}' for i in range(num_examples-1)]
        
        for i, (img, title) in enumerate(zip(augmented, titles)):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        
        plt.show()


# TensorFlow/Keras ImageDataGenerator equivalent
def get_keras_augmentation_generator():
    """
    Get TensorFlow/Keras ImageDataGenerator for augmentation
    
    Returns:
        ImageDataGenerator configured for X-ray images
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=20,           # Rotate up to 20 degrees
        width_shift_range=0.2,       # Shift horizontally up to 20%
        height_shift_range=0.2,      # Shift vertically up to 20%
        horizontal_flip=True,        # Random horizontal flip
        zoom_range=0.2,              # Zoom in/out up to 20%
        fill_mode='constant',        # Fill with black for medical images
        cval=0,                      # Fill value
        # Note: Be careful with brightness/contrast for medical images
    )
    
    return datagen


if __name__ == '__main__':
    # Example usage
    print("X-ray Augmentation Module")
    print("=" * 60)
    print("\nUsage examples:")
    print("1. Single augmentation:")
    print("   augmenter = XRayAugmenter()")
    print("   aug_img = augmenter.rotate(image, angle=15)")
    print("\n2. Full pipeline:")
    print("   aug_img = augmenter.augment_pipeline(image)")
    print("\n3. Generate dataset:")
    print("   aug_images, aug_labels = augmenter.generate_augmented_dataset(images, labels, 3)")
    print("\n4. Visualize:")
    print("   augmenter.visualize_augmentations(image)")
