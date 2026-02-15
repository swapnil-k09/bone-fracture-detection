"""
Image Preprocessing Utilities for X-ray Bone Fracture Detection
Uses OpenCV for various preprocessing techniques
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import os


class XRayPreprocessor:
    """
    Preprocessor for X-ray images using OpenCV
    Handles loading, resizing, denoising, contrast enhancement, and normalization
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the preprocessor
        
        Args:
            target_size: Target size for resized images (width, height)
        """
        self.target_size = target_size
        
    def load_image(self, image_path: str, grayscale: bool = True) -> Optional[np.ndarray]:
        """
        Load an image from file
        
        Args:
            image_path: Path to the image file
            grayscale: Whether to load as grayscale
            
        Returns:
            Loaded image or None if failed
        """
        try:
            if grayscale:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"❌ Failed to load image: {image_path}")
                return None
                
            return img
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None
    
    def resize_image(self, image: np.ndarray, 
                    target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            target_size: Target size (width, height), uses self.target_size if None
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
            
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        return resized
    
    def denoise_gaussian(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur for noise reduction
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (must be odd)
            
        Returns:
            Denoised image
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return denoised
    
    def denoise_bilateral(self, image: np.ndarray, d: int = 9, 
                         sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
        """
        Apply bilateral filter (preserves edges while reducing noise)
        
        Args:
            image: Input image
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
            
        Returns:
            Denoised image
        """
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return denoised
    
    def denoise_nlm(self, image: np.ndarray, h: int = 10, 
                   template_window_size: int = 7, 
                   search_window_size: int = 21) -> np.ndarray:
        """
        Apply Non-Local Means Denoising (advanced denoising)
        
        Args:
            image: Input image
            h: Filter strength
            template_window_size: Size of template patch
            search_window_size: Size of search area
            
        Returns:
            Denoised image
        """
        denoised = cv2.fastNlMeansDenoising(image, None, h, 
                                           template_window_size, 
                                           search_window_size)
        return denoised
    
    def enhance_contrast_clahe(self, image: np.ndarray, 
                               clip_limit: float = 2.0,
                               tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Very effective for medical images like X-rays
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        return enhanced
    
    def enhance_contrast_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Apply standard histogram equalization
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        enhanced = cv2.equalizeHist(image)
        return enhanced
    
    def normalize_image(self, image: np.ndarray, 
                       method: str = 'minmax') -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            image: Input image
            method: Normalization method ('minmax' or 'standard')
            
        Returns:
            Normalized image (float32, range 0-1)
        """
        image = image.astype(np.float32)
        
        if method == 'minmax':
            # Normalize to [0, 1]
            normalized = (image - image.min()) / (image.max() - image.min() + 1e-7)
        elif method == 'standard':
            # Standardize to mean=0, std=1
            normalized = (image - image.mean()) / (image.std() + 1e-7)
            # Clip to reasonable range
            normalized = np.clip(normalized, -3, 3)
            # Scale to [0, 1]
            normalized = (normalized + 3) / 6
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return normalized
    
    def sharpen_image(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Sharpen image using unsharp masking
        
        Args:
            image: Input image
            strength: Sharpening strength (0-2, 1 is normal)
            
        Returns:
            Sharpened image
        """
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        
        return sharpened
    
    def remove_borders(self, image: np.ndarray, threshold: int = 10) -> np.ndarray:
        """
        Remove black borders from X-ray images
        
        Args:
            image: Input image
            threshold: Threshold for detecting borders
            
        Returns:
            Cropped image
        """
        # Find non-zero pixels
        coords = cv2.findNonZero((image > threshold).astype(np.uint8))
        
        if coords is not None:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(coords)
            # Crop image
            cropped = image[y:y+h, x:x+w]
            return cropped
        else:
            return image
    
    def preprocess_full_pipeline(self, image_path: str, 
                                 denoise_method: str = 'gaussian',
                                 enhance_contrast: bool = True,
                                 sharpen: bool = False,
                                 remove_borders: bool = True) -> Optional[np.ndarray]:
        """
        Complete preprocessing pipeline for X-ray images
        
        Args:
            image_path: Path to input image
            denoise_method: Method for denoising ('gaussian', 'bilateral', 'nlm', or None)
            enhance_contrast: Whether to apply CLAHE
            sharpen: Whether to sharpen the image
            remove_borders: Whether to remove black borders
            
        Returns:
            Preprocessed image ready for model input
        """
        # Load image
        img = self.load_image(image_path, grayscale=True)
        if img is None:
            return None
        
        # Remove borders if requested
        if remove_borders:
            img = self.remove_borders(img)
        
        # Resize to target size
        img = self.resize_image(img)
        
        # Denoise
        if denoise_method == 'gaussian':
            img = self.denoise_gaussian(img, kernel_size=5)
        elif denoise_method == 'bilateral':
            img = self.denoise_bilateral(img)
        elif denoise_method == 'nlm':
            img = self.denoise_nlm(img)
        
        # Enhance contrast
        if enhance_contrast:
            img = self.enhance_contrast_clahe(img)
        
        # Sharpen
        if sharpen:
            img = self.sharpen_image(img, strength=0.5)
        
        # Normalize to [0, 1]
        img = self.normalize_image(img, method='minmax')
        
        return img
    
    def preprocess_batch(self, image_paths: list, 
                        save_dir: Optional[str] = None,
                        show_progress: bool = True) -> list:
        """
        Preprocess a batch of images
        
        Args:
            image_paths: List of image paths
            save_dir: Directory to save preprocessed images (optional)
            show_progress: Whether to show progress
            
        Returns:
            List of preprocessed images
        """
        processed_images = []
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        total = len(image_paths)
        for i, img_path in enumerate(image_paths):
            if show_progress and i % 100 == 0:
                print(f"Processing: {i}/{total} ({100*i/total:.1f}%)")
            
            # Preprocess
            img = self.preprocess_full_pipeline(img_path)
            
            if img is not None:
                processed_images.append(img)
                
                # Save if directory provided
                if save_dir:
                    filename = os.path.basename(img_path)
                    save_path = os.path.join(save_dir, filename)
                    # Convert back to uint8 for saving
                    img_save = (img * 255).astype(np.uint8)
                    cv2.imwrite(save_path, img_save)
        
        if show_progress:
            print(f"✅ Processed {len(processed_images)}/{total} images")
        
        return processed_images
    
    def visualize_preprocessing_steps(self, image_path: str, 
                                     save_path: Optional[str] = None):
        """
        Visualize each preprocessing step
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization (optional)
        """
        import matplotlib.pyplot as plt
        
        # Load original
        original = self.load_image(image_path, grayscale=True)
        if original is None:
            return
        
        # Apply each step
        resized = self.resize_image(original)
        denoised = self.denoise_gaussian(resized)
        enhanced = self.enhance_contrast_clahe(denoised)
        normalized = self.normalize_image(enhanced)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('X-ray Preprocessing Pipeline', fontsize=16, fontweight='bold')
        
        # Original
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('1. Original')
        axes[0, 0].axis('off')
        
        # Resized
        axes[0, 1].imshow(resized, cmap='gray')
        axes[0, 1].set_title(f'2. Resized to {self.target_size}')
        axes[0, 1].axis('off')
        
        # Denoised
        axes[0, 2].imshow(denoised, cmap='gray')
        axes[0, 2].set_title('3. Denoised (Gaussian)')
        axes[0, 2].axis('off')
        
        # Enhanced
        axes[1, 0].imshow(enhanced, cmap='gray')
        axes[1, 0].set_title('4. Enhanced (CLAHE)')
        axes[1, 0].axis('off')
        
        # Normalized
        axes[1, 1].imshow(normalized, cmap='gray')
        axes[1, 1].set_title('5. Normalized')
        axes[1, 1].axis('off')
        
        # Histogram comparison
        axes[1, 2].hist(original.ravel(), bins=256, alpha=0.5, label='Original')
        axes[1, 2].hist((enhanced).ravel(), bins=256, alpha=0.5, label='Enhanced')
        axes[1, 2].set_title('6. Histogram Comparison')
        axes[1, 2].legend()
        axes[1, 2].set_xlabel('Pixel Value')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        
        plt.show()


# Convenience functions for quick use
def preprocess_single_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Quick preprocessing of a single image
    
    Args:
        image_path: Path to image
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image
    """
    preprocessor = XRayPreprocessor(target_size=target_size)
    return preprocessor.preprocess_full_pipeline(image_path)


def preprocess_directory(input_dir: str, output_dir: str, 
                        target_size: Tuple[int, int] = (224, 224)):
    """
    Preprocess all images in a directory
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for preprocessed images
        target_size: Target size for resizing
    """
    import glob
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    
    # Preprocess
    preprocessor = XRayPreprocessor(target_size=target_size)
    preprocessor.preprocess_batch(image_paths, save_dir=output_dir, show_progress=True)
    
    print(f"✅ Preprocessed images saved to: {output_dir}")


if __name__ == '__main__':
    # Example usage
    print("X-ray Preprocessor Module")
    print("=" * 50)
    print("\nUsage examples:")
    print("1. Single image:")
    print("   img = preprocess_single_image('path/to/xray.png')")
    print("\n2. Directory:")
    print("   preprocess_directory('data/raw/', 'data/processed/')")
    print("\n3. Custom pipeline:")
    print("   preprocessor = XRayPreprocessor(target_size=(256, 256))")
    print("   img = preprocessor.preprocess_full_pipeline('xray.png')")
