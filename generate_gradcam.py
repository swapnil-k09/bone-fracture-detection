"""
Generate Grad-CAM Visualizations
Standalone script to create heatmap visualizations for X-ray predictions
"""

import os
import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utils.gradcam import GradCAM, GradCAMPlusPlus, batch_visualize, compare_gradcam_methods
from utils.preprocess import preprocess_single_image


def generate_single_gradcam(model_path: str,
                           image_path: str,
                           output_path: str = 'gradcam_result.png',
                           layer_name: str = None,
                           use_gradcam_pp: bool = False):
    """
    Generate Grad-CAM for a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to X-ray image
        output_path: Output path for visualization
        layer_name: Target layer (auto-detected if None)
        use_gradcam_pp: Use Grad-CAM++ instead of Grad-CAM
    """
    print("\n" + "="*70)
    print("GRAD-CAM VISUALIZATION GENERATION")
    print("="*70 + "\n")
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded!\n")
    
    # Load and preprocess image
    print(f"📸 Loading image: {image_path}")
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return
    
    # Preprocess for model
    preprocessed = preprocess_single_image(image_path, target_size=(224, 224))
    
    if preprocessed is None:
        print(f"❌ Error: Could not preprocess image")
        return
    
    print("✅ Image loaded and preprocessed!\n")
    
    # Get prediction
    print("🔮 Making prediction...")
    prediction = model.predict(np.expand_dims(preprocessed, axis=0), verbose=0)[0][0]
    predicted_class = "Fractured" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else (1 - prediction)
    
    print(f"   Prediction: {predicted_class}")
    print(f"   Confidence: {confidence*100:.1f}%\n")
    
    # Generate Grad-CAM
    print("🎨 Generating Grad-CAM visualization...")
    
    if use_gradcam_pp:
        gradcam = GradCAMPlusPlus(model, layer_name)
        method = "Grad-CAM++"
    else:
        gradcam = GradCAM(model, layer_name)
        method = "Grad-CAM"
    
    # Create visualization
    gradcam.visualize(
        preprocessed,
        original_image=original_image,
        prediction=prediction,
        save_path=output_path,
        show=True
    )
    
    print(f"\n✅ {method} visualization complete!")
    print(f"📁 Saved to: {output_path}")


def generate_batch_gradcam(model_path: str,
                          image_dir: str,
                          output_dir: str = 'gradcam_batch',
                          layer_name: str = None,
                          max_images: int = None):
    """
    Generate Grad-CAM for multiple images
    
    Args:
        model_path: Path to trained model
        image_dir: Directory containing X-ray images
        output_dir: Output directory
        layer_name: Target layer
        max_images: Maximum number of images to process
    """
    print("\n" + "="*70)
    print("BATCH GRAD-CAM GENERATION")
    print("="*70 + "\n")
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded!\n")
    
    # Get image files
    import glob
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    # Limit number of images
    if max_images and len(image_files) > max_images:
        image_files = image_files[:max_images]
    
    print(f"📸 Found {len(image_files)} images")
    
    # Load and preprocess images
    print("⚙️  Preprocessing images...")
    preprocessed_images = []
    original_images = []
    
    for img_path in image_files:
        # Load original
        orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if orig is None:
            continue
        
        # Preprocess
        prep = preprocess_single_image(img_path)
        if prep is None:
            continue
        
        preprocessed_images.append(prep)
        original_images.append(orig)
    
    print(f"✅ Preprocessed {len(preprocessed_images)} images\n")
    
    # Generate visualizations
    batch_visualize(
        model=model,
        images=preprocessed_images,
        original_images=original_images,
        layer_name=layer_name,
        output_dir=output_dir
    )
    
    print(f"\n✅ Batch processing complete!")
    print(f"📁 All visualizations saved to: {output_dir}/")


def compare_methods(model_path: str,
                   image_path: str,
                   output_path: str = 'gradcam_comparison.png',
                   layer_name: str = None):
    """
    Compare Grad-CAM and Grad-CAM++ on the same image
    
    Args:
        model_path: Path to model
        image_path: Path to image
        output_path: Output path
        layer_name: Target layer
    """
    print("\n" + "="*70)
    print("GRAD-CAM METHOD COMPARISON")
    print("="*70 + "\n")
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded!\n")
    
    # Load image
    print(f"📸 Loading image: {image_path}")
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    preprocessed = preprocess_single_image(image_path)
    print("✅ Image loaded!\n")
    
    # Compare methods
    print("🎨 Generating comparison...")
    compare_gradcam_methods(
        model=model,
        image=preprocessed,
        original_image=original,
        layer_name=layer_name,
        save_path=output_path
    )
    
    print(f"\n✅ Comparison complete!")
    print(f"📁 Saved to: {output_path}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM visualizations for X-ray fracture detection'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Single image command
    single_parser = subparsers.add_parser('single', help='Generate Grad-CAM for single image')
    single_parser.add_argument('--model', required=True, help='Path to trained model')
    single_parser.add_argument('--image', required=True, help='Path to X-ray image')
    single_parser.add_argument('--output', default='gradcam_result.png', help='Output path')
    single_parser.add_argument('--layer', default=None, help='Target layer name')
    single_parser.add_argument('--gradcam_pp', action='store_true', help='Use Grad-CAM++')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Generate Grad-CAM for multiple images')
    batch_parser.add_argument('--model', required=True, help='Path to trained model')
    batch_parser.add_argument('--image_dir', required=True, help='Directory with images')
    batch_parser.add_argument('--output_dir', default='gradcam_batch', help='Output directory')
    batch_parser.add_argument('--layer', default=None, help='Target layer name')
    batch_parser.add_argument('--max_images', type=int, default=None, help='Max images to process')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare Grad-CAM methods')
    compare_parser.add_argument('--model', required=True, help='Path to trained model')
    compare_parser.add_argument('--image', required=True, help='Path to X-ray image')
    compare_parser.add_argument('--output', default='gradcam_comparison.png', help='Output path')
    compare_parser.add_argument('--layer', default=None, help='Target layer name')
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'single':
        generate_single_gradcam(
            model_path=args.model,
            image_path=args.image,
            output_path=args.output,
            layer_name=args.layer,
            use_gradcam_pp=args.gradcam_pp
        )
    
    elif args.command == 'batch':
        generate_batch_gradcam(
            model_path=args.model,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            layer_name=args.layer,
            max_images=args.max_images
        )
    
    elif args.command == 'compare':
        compare_methods(
            model_path=args.model,
            image_path=args.image,
            output_path=args.output,
            layer_name=args.layer
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
