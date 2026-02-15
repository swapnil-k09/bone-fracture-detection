"""
Visualization Utilities for X-ray Analysis
Helps explore and understand the dataset
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os


class XRayVisualizer:
    """
    Visualization tools for X-ray images and model outputs
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def show_image(self, image: np.ndarray, title: str = "X-ray Image",
                  save_path: Optional[str] = None):
        """
        Display a single X-ray image
        
        Args:
            image: Image to display
            title: Title for the image
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def show_grid(self, images: List[np.ndarray], 
                 titles: Optional[List[str]] = None,
                 rows: int = 3, cols: int = 3,
                 save_path: Optional[str] = None):
        """
        Display multiple images in a grid
        
        Args:
            images: List of images to display
            titles: Optional list of titles
            rows: Number of rows
            cols: Number of columns
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.ravel()
        
        for i, ax in enumerate(axes):
            if i < len(images):
                ax.imshow(images[i], cmap='gray')
                if titles and i < len(titles):
                    ax.set_title(titles[i])
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def compare_images(self, images: List[np.ndarray],
                      titles: List[str],
                      save_path: Optional[str] = None):
        """
        Compare multiple images side by side
        
        Args:
            images: List of images to compare
            titles: List of titles
            save_path: Optional path to save figure
        """
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
        
        if n_images == 1:
            axes = [axes]
        
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def show_histogram(self, image: np.ndarray, title: str = "Histogram",
                      save_path: Optional[str] = None):
        """
        Show histogram of pixel intensities
        
        Args:
            image: Input image
            title: Title for plot
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(10, 6))
        
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        plt.plot(hist, color='black')
        plt.fill_between(range(256), hist.ravel(), alpha=0.3)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def compare_histograms(self, images: List[np.ndarray],
                          labels: List[str],
                          title: str = "Histogram Comparison",
                          save_path: Optional[str] = None):
        """
        Compare histograms of multiple images
        
        Args:
            images: List of images
            labels: List of labels for each image
            title: Title for plot
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(12, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(images)))
        
        for img, label, color in zip(images, labels, colors):
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            plt.plot(hist, label=label, color=color, alpha=0.7, linewidth=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def show_class_distribution(self, labels: List[int],
                               class_names: List[str] = ['Normal', 'Fractured'],
                               title: str = "Class Distribution",
                               save_path: Optional[str] = None):
        """
        Show distribution of classes in dataset
        
        Args:
            labels: List of labels
            class_names: Names of classes
            title: Title for plot
            save_path: Optional path to save figure
        """
        # Count each class
        unique, counts = np.unique(labels, return_counts=True)
        
        # Create bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        colors = ['#2ecc71', '#e74c3c']
        ax1.bar([class_names[i] for i in unique], counts, color=colors, alpha=0.7)
        ax1.set_title(f'{title} - Bar Chart', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (label, count) in enumerate(zip(unique, counts)):
            ax1.text(i, count, f'{count:,}\n({100*count/len(labels):.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(counts, labels=[class_names[i] for i in unique],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title(f'{title} - Pie Chart', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        # Print statistics
        print("\n" + "=" * 50)
        print("CLASS DISTRIBUTION")
        print("=" * 50)
        for i, count in zip(unique, counts):
            pct = 100 * count / len(labels)
            print(f"{class_names[i]:10s}: {count:6,} ({pct:5.1f}%)")
        print(f"{'Total':10s}: {len(labels):6,}")
        print("=" * 50 + "\n")
    
    def show_sample_images(self, image_paths: List[str],
                          labels: List[int],
                          class_names: List[str] = ['Normal', 'Fractured'],
                          samples_per_class: int = 5,
                          save_path: Optional[str] = None):
        """
        Show sample images from each class
        
        Args:
            image_paths: List of image paths
            labels: List of labels
            class_names: Names of classes
            samples_per_class: Number of samples per class
            save_path: Optional path to save figure
        """
        # Separate by class
        class_samples = {i: [] for i in range(len(class_names))}
        
        for path, label in zip(image_paths, labels):
            if len(class_samples[label]) < samples_per_class:
                class_samples[label].append(path)
        
        # Create plot
        fig, axes = plt.subplots(len(class_names), samples_per_class,
                                figsize=(3*samples_per_class, 3*len(class_names)))
        
        if len(class_names) == 1:
            axes = [axes]
        
        for class_idx, class_name in enumerate(class_names):
            for sample_idx, img_path in enumerate(class_samples[class_idx]):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                ax = axes[class_idx][sample_idx]
                ax.imshow(img, cmap='gray')
                
                if sample_idx == 0:
                    ax.set_ylabel(class_name, fontsize=12, fontweight='bold')
                
                ax.axis('off')
        
        plt.suptitle('Sample X-ray Images by Class', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def show_image_statistics(self, images: List[np.ndarray],
                             save_path: Optional[str] = None):
        """
        Show statistical analysis of images
        
        Args:
            images: List of images
            save_path: Optional path to save figure
        """
        # Calculate statistics
        shapes = [img.shape for img in images]
        heights = [s[0] for s in shapes]
        widths = [s[1] for s in shapes]
        means = [img.mean() for img in images]
        stds = [img.std() for img in images]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Height distribution
        axes[0, 0].hist(heights, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Image Heights', fontweight='bold')
        axes[0, 0].set_xlabel('Height (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(heights), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(heights):.0f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Width distribution
        axes[0, 1].hist(widths, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Image Widths', fontweight='bold')
        axes[0, 1].set_xlabel('Width (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(widths), color='red', linestyle='--',
                          label=f'Mean: {np.mean(widths):.0f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mean intensity distribution
        axes[1, 0].hist(means, bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Mean Pixel Intensity', fontweight='bold')
        axes[1, 0].set_xlabel('Mean Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Standard deviation distribution
        axes[1, 1].hist(stds, bins=30, color='plum', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Pixel Intensity Standard Deviation', fontweight='bold')
        axes[1, 1].set_xlabel('Std Dev')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Dataset Image Statistics', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        # Print summary
        print("\n" + "=" * 60)
        print("IMAGE STATISTICS SUMMARY")
        print("=" * 60)
        print(f"Number of images:     {len(images):,}")
        print(f"\nHeight:")
        print(f"  Min:     {min(heights):,} px")
        print(f"  Max:     {max(heights):,} px")
        print(f"  Mean:    {np.mean(heights):.0f} px")
        print(f"  Median:  {np.median(heights):.0f} px")
        print(f"\nWidth:")
        print(f"  Min:     {min(widths):,} px")
        print(f"  Max:     {max(widths):,} px")
        print(f"  Mean:    {np.mean(widths):.0f} px")
        print(f"  Median:  {np.median(widths):.0f} px")
        print(f"\nMean Intensity:")
        print(f"  Min:     {min(means):.2f}")
        print(f"  Max:     {max(means):.2f}")
        print(f"  Average: {np.mean(means):.2f}")
        print("=" * 60 + "\n")


def create_data_exploration_report(data_dir: str, output_dir: str = 'reports'):
    """
    Create a comprehensive data exploration report
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory to save report visualizations
    """
    from utils.data_loader import DatasetLoader
    
    print("📊 Creating Data Exploration Report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    loader = DatasetLoader(data_dir)
    visualizer = XRayVisualizer()
    
    # Get statistics
    print("\n1. Dataset Statistics...")
    loader.print_dataset_info()
    
    # Load sample images
    print("\n2. Loading sample images...")
    train_paths, train_labels = loader.load_data_paths('train')
    
    if len(train_paths) > 0:
        # Sample images
        sample_indices = np.random.choice(len(train_paths), 
                                         min(100, len(train_paths)), 
                                         replace=False)
        sample_paths = [train_paths[i] for i in sample_indices]
        sample_labels = [train_labels[i] for i in sample_indices]
        
        # Load actual images
        sample_images = []
        for path in sample_paths[:20]:  # Limit to 20 for statistics
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sample_images.append(img)
        
        # Class distribution
        print("\n3. Class Distribution...")
        visualizer.show_class_distribution(
            train_labels,
            save_path=os.path.join(output_dir, 'class_distribution.png')
        )
        
        # Sample images
        print("\n4. Sample Images...")
        visualizer.show_sample_images(
            train_paths,
            train_labels,
            samples_per_class=5,
            save_path=os.path.join(output_dir, 'sample_images.png')
        )
        
        # Image statistics
        print("\n5. Image Statistics...")
        visualizer.show_image_statistics(
            sample_images,
            save_path=os.path.join(output_dir, 'image_statistics.png')
        )
        
        print(f"\n✅ Report saved to: {output_dir}/")
    else:
        print("⚠️  No images found in dataset!")


if __name__ == '__main__':
    print("X-ray Visualization Module")
    print("=" * 60)
    print("\nUsage examples:")
    print("1. Show image:")
    print("   viz = XRayVisualizer()")
    print("   viz.show_image(image, 'X-ray')")
    print("\n2. Create exploration report:")
    print("   create_data_exploration_report('data/', 'reports/')")
