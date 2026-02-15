"""
Data Loading Utilities for X-ray Bone Fracture Detection
Handles dataset organization, loading, and splitting
"""

import os
import glob
import numpy as np
from typing import Tuple, List, Optional
import random
from pathlib import Path


class DatasetLoader:
    """
    Handles loading and organizing X-ray datasets
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize dataset loader
        
        Args:
            data_dir: Root directory containing train/validation/test folders
        """
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'validation')
        self.test_dir = os.path.join(data_dir, 'test')
        
    def get_image_paths(self, directory: str, class_name: Optional[str] = None) -> List[str]:
        """
        Get all image paths from a directory
        
        Args:
            directory: Directory to search
            class_name: Optional class subdirectory ('fractured' or 'normal')
            
        Returns:
            List of image paths
        """
        if class_name:
            search_dir = os.path.join(directory, class_name)
        else:
            search_dir = directory
        
        # Support multiple image formats
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.PNG', '*.JPG']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(search_dir, '**', ext), recursive=True))
        
        return sorted(image_paths)
    
    def get_dataset_statistics(self) -> dict:
        """
        Get statistics about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'train': {
                'fractured': len(self.get_image_paths(self.train_dir, 'fractured')),
                'normal': len(self.get_image_paths(self.train_dir, 'normal'))
            },
            'validation': {
                'fractured': len(self.get_image_paths(self.val_dir, 'fractured')),
                'normal': len(self.get_image_paths(self.val_dir, 'normal'))
            },
            'test': {
                'fractured': len(self.get_image_paths(self.test_dir, 'fractured')),
                'normal': len(self.get_image_paths(self.test_dir, 'normal'))
            }
        }
        
        # Calculate totals
        for split in ['train', 'validation', 'test']:
            stats[split]['total'] = stats[split]['fractured'] + stats[split]['normal']
        
        stats['total'] = sum(s['total'] for s in stats.values())
        
        return stats
    
    def print_dataset_info(self):
        """
        Print detailed dataset information
        """
        stats = self.get_dataset_statistics()
        
        print("\n" + "=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)
        
        for split in ['train', 'validation', 'test']:
            print(f"\n{split.upper()}:")
            print(f"  Fractured: {stats[split]['fractured']:,}")
            print(f"  Normal:    {stats[split]['normal']:,}")
            print(f"  Total:     {stats[split]['total']:,}")
            
            if stats[split]['total'] > 0:
                frac_pct = 100 * stats[split]['fractured'] / stats[split]['total']
                norm_pct = 100 * stats[split]['normal'] / stats[split]['total']
                print(f"  Balance:   {frac_pct:.1f}% fractured, {norm_pct:.1f}% normal")
        
        print(f"\nTOTAL IMAGES: {stats['total']:,}")
        print("=" * 60 + "\n")
    
    def load_data_paths(self, split: str = 'train') -> Tuple[List[str], List[int]]:
        """
        Load image paths and labels for a dataset split
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            
        Returns:
            Tuple of (image_paths, labels)
        """
        if split == 'train':
            directory = self.train_dir
        elif split == 'validation':
            directory = self.val_dir
        elif split == 'test':
            directory = self.test_dir
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Get fractured images (label = 1)
        fractured_paths = self.get_image_paths(directory, 'fractured')
        fractured_labels = [1] * len(fractured_paths)
        
        # Get normal images (label = 0)
        normal_paths = self.get_image_paths(directory, 'normal')
        normal_labels = [0] * len(normal_paths)
        
        # Combine
        all_paths = fractured_paths + normal_paths
        all_labels = fractured_labels + normal_labels
        
        return all_paths, all_labels
    
    def create_balanced_subset(self, split: str = 'train', 
                              samples_per_class: int = 1000) -> Tuple[List[str], List[int]]:
        """
        Create a balanced subset of the dataset
        Useful for quick testing or handling class imbalance
        
        Args:
            split: Dataset split
            samples_per_class: Number of samples per class
            
        Returns:
            Tuple of (image_paths, labels)
        """
        if split == 'train':
            directory = self.train_dir
        elif split == 'validation':
            directory = self.val_dir
        elif split == 'test':
            directory = self.test_dir
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Get all paths for each class
        fractured_paths = self.get_image_paths(directory, 'fractured')
        normal_paths = self.get_image_paths(directory, 'normal')
        
        # Sample randomly
        random.seed(42)  # For reproducibility
        
        frac_sample = random.sample(fractured_paths, 
                                   min(samples_per_class, len(fractured_paths)))
        norm_sample = random.sample(normal_paths, 
                                   min(samples_per_class, len(normal_paths)))
        
        # Create labels
        frac_labels = [1] * len(frac_sample)
        norm_labels = [0] * len(norm_sample)
        
        # Combine and shuffle
        all_paths = frac_sample + norm_sample
        all_labels = frac_labels + norm_labels
        
        # Shuffle together
        combined = list(zip(all_paths, all_labels))
        random.shuffle(combined)
        all_paths, all_labels = zip(*combined)
        
        print(f"✅ Created balanced subset: {len(all_paths)} images")
        print(f"   Fractured: {len(frac_sample)}, Normal: {len(norm_sample)}")
        
        return list(all_paths), list(all_labels)
    
    def verify_data_integrity(self) -> dict:
        """
        Verify dataset integrity (check for corrupted images, etc.)
        
        Returns:
            Dictionary with verification results
        """
        import cv2
        
        print("\n🔍 Verifying dataset integrity...")
        
        results = {
            'total_checked': 0,
            'corrupted': [],
            'missing': [],
            'valid': 0
        }
        
        # Check all splits
        for split in ['train', 'validation', 'test']:
            paths, _ = self.load_data_paths(split)
            
            print(f"\nChecking {split}...")
            for i, path in enumerate(paths):
                results['total_checked'] += 1
                
                if i % 500 == 0 and i > 0:
                    print(f"  Checked {i}/{len(paths)}...")
                
                # Check if file exists
                if not os.path.exists(path):
                    results['missing'].append(path)
                    continue
                
                # Try to load image
                try:
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        results['corrupted'].append(path)
                    else:
                        results['valid'] += 1
                except Exception as e:
                    results['corrupted'].append(path)
        
        # Print results
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print(f"Total images checked: {results['total_checked']:,}")
        print(f"Valid images:         {results['valid']:,}")
        print(f"Corrupted images:     {len(results['corrupted']):,}")
        print(f"Missing images:       {len(results['missing']):,}")
        
        if results['corrupted']:
            print("\n❌ Corrupted images found:")
            for path in results['corrupted'][:5]:  # Show first 5
                print(f"   {path}")
            if len(results['corrupted']) > 5:
                print(f"   ... and {len(results['corrupted']) - 5} more")
        
        if results['missing']:
            print("\n❌ Missing images:")
            for path in results['missing'][:5]:
                print(f"   {path}")
            if len(results['missing']) > 5:
                print(f"   ... and {len(results['missing']) - 5} more")
        
        if not results['corrupted'] and not results['missing']:
            print("\n✅ All images are valid!")
        
        print("=" * 60 + "\n")
        
        return results


def organize_mura_dataset(mura_dir: str, output_dir: str, 
                         train_split: float = 0.8,
                         val_split: float = 0.1,
                         test_split: float = 0.1):
    """
    Organize MURA dataset into train/validation/test structure
    
    MURA comes with its own train/val split, but this function
    allows you to reorganize it or create a test set
    
    Args:
        mura_dir: Directory containing MURA dataset
        output_dir: Output directory for organized dataset
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
    """
    import shutil
    
    print("📦 Organizing MURA dataset...")
    print(f"Input:  {mura_dir}")
    print(f"Output: {output_dir}")
    
    # Create output structure
    for split in ['train', 'validation', 'test']:
        for class_name in ['fractured', 'normal']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # This is a template - actual MURA organization depends on its structure
    # Users should modify based on their specific MURA download
    
    print("\n⚠️  This is a template function.")
    print("Please modify according to your MURA dataset structure.")
    print("Typical MURA structure:")
    print("  MURA-v1.1/")
    print("    train/")
    print("      XR_WRIST/")
    print("        patient00001/")
    print("          study1_positive/")
    print("          study1_negative/")
    
    return output_dir


if __name__ == '__main__':
    # Example usage
    print("Dataset Loader Module")
    print("=" * 60)
    
    # Create a loader instance
    loader = DatasetLoader('data')
    
    # Print dataset info
    loader.print_dataset_info()
    
    print("\nUsage examples:")
    print("1. Get statistics:")
    print("   stats = loader.get_dataset_statistics()")
    print("\n2. Load training data:")
    print("   paths, labels = loader.load_data_paths('train')")
    print("\n3. Create balanced subset:")
    print("   paths, labels = loader.create_balanced_subset('train', 1000)")
    print("\n4. Verify integrity:")
    print("   results = loader.verify_data_integrity()")
