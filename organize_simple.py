"""
Simple Dataset Organizer - Manual Version
Just tell it the exact paths and it will work!
"""

import os
import shutil
import random

def organize_dataset():
    print("\n" + "="*60)
    print("SIMPLE DATASET ORGANIZER")
    print("="*60)
    
    # Get source path
    print("\n📁 Where is your Kaggle dataset?")
    print("Example: C:\\Users\\swapn\\Downloads\\bone-fracture-detection")
    source = input("\nSource path: ").strip().strip('"')
    
    # Check if path exists
    if not os.path.exists(source):
        print(f"\n❌ ERROR: Path doesn't exist: {source}")
        print("\nTry these:")
        print("1. Check spelling")
        print("2. Use full path (start with C:\\)")
        print("3. Copy-paste the path from File Explorer")
        return
    
    # Look for image folders
    print(f"\n🔍 Looking in: {source}")
    print("\nSearching for folders...")
    
    folders = os.listdir(source)
    print(f"Found folders: {folders}")
    
    # Try to find fractured/normal folders (case insensitive)
    fractured_folder = None
    normal_folder = None
    
    for folder in folders:
        folder_lower = folder.lower()
        folder_path = os.path.join(source, folder)
        
        if not os.path.isdir(folder_path):
            continue
            
        if 'fracture' in folder_lower and 'not' not in folder_lower:
            fractured_folder = folder
        elif 'not' in folder_lower or 'normal' in folder_lower:
            normal_folder = folder
    
    if not fractured_folder or not normal_folder:
        print(f"\n❌ ERROR: Can't find image folders!")
        print(f"\nFound folders: {folders}")
        print("\nI'm looking for:")
        print("  - A folder with 'Fractured' (without 'not')")
        print("  - A folder with 'Not Fractured' or 'Normal'")
        print("\nWhat are your folder names? Tell me and I'll fix it!")
        return
    
    print(f"\n✅ Found fractured folder: {fractured_folder}")
    print(f"✅ Found normal folder: {normal_folder}")
    
    # Get images
    fractured_path = os.path.join(source, fractured_folder)
    normal_path = os.path.join(source, normal_folder)
    
    print(f"\n🔍 Counting images...")
    
    # Find all images
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
    
    fractured_images = [
        os.path.join(fractured_path, f) 
        for f in os.listdir(fractured_path) 
        if f.endswith(extensions)
    ]
    
    normal_images = [
        os.path.join(normal_path, f) 
        for f in os.listdir(normal_path) 
        if f.endswith(extensions)
    ]
    
    print(f"\n✅ Fractured images: {len(fractured_images)}")
    print(f"✅ Normal images: {len(normal_images)}")
    print(f"✅ Total: {len(fractured_images) + len(normal_images)}")
    
    if len(fractured_images) == 0 or len(normal_images) == 0:
        print(f"\n❌ ERROR: No images found!")
        print(f"\nChecking folder contents:")
        print(f"Fractured folder: {os.listdir(fractured_path)[:5]}")
        print(f"Normal folder: {os.listdir(normal_path)[:5]}")
        return
    
    # Get output path
    print("\n📂 Where should I organize the data?")
    print("Press Enter for: data")
    output = input("\nOutput path [data]: ").strip().strip('"')
    if not output:
        output = 'data'
    
    print(f"\n✅ Will organize into: {output}/")
    print("\nReady to start? This will take 5-10 minutes.")
    input("Press Enter to continue...")
    
    # Create folders
    print("\n📁 Creating folders...")
    for split in ['train', 'validation', 'test']:
        for label in ['fractured', 'normal']:
            path = os.path.join(output, split, label)
            os.makedirs(path, exist_ok=True)
    print("✅ Folders created!")
    
    # Shuffle
    random.seed(42)
    random.shuffle(fractured_images)
    random.shuffle(normal_images)
    
    # Split function
    def split_list(lst):
        total = len(lst)
        train_end = int(0.70 * total)
        val_end = int(0.85 * total)
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]
    
    frac_train, frac_val, frac_test = split_list(fractured_images)
    norm_train, norm_val, norm_test = split_list(normal_images)
    
    print(f"\n📊 Split:")
    print(f"  Train:      {len(frac_train)} fractured + {len(norm_train)} normal")
    print(f"  Validation: {len(frac_val)} fractured + {len(norm_val)} normal")
    print(f"  Test:       {len(frac_test)} fractured + {len(norm_test)} normal")
    
    # Copy function with progress
    def copy_images(images, dest_folder, label):
        total = len(images)
        for i, src in enumerate(images, 1):
            ext = os.path.splitext(src)[1]
            dst = os.path.join(dest_folder, f'{label}_{i:04d}{ext}')
            shutil.copy2(src, dst)
            
            if i % 50 == 0 or i == total:
                percent = (i / total) * 100
                print(f"  {label}: {i}/{total} ({percent:.1f}%)")
    
    # Copy all files
    print("\n📦 Copying training set...")
    copy_images(frac_train, os.path.join(output, 'train', 'fractured'), 'fractured')
    copy_images(norm_train, os.path.join(output, 'train', 'normal'), 'normal')
    
    print("\n📦 Copying validation set...")
    copy_images(frac_val, os.path.join(output, 'validation', 'fractured'), 'fractured')
    copy_images(norm_val, os.path.join(output, 'validation', 'normal'), 'normal')
    
    print("\n📦 Copying test set...")
    copy_images(frac_test, os.path.join(output, 'test', 'fractured'), 'fractured')
    copy_images(norm_test, os.path.join(output, 'test', 'normal'), 'normal')
    
    print("\n" + "="*60)
    print("✅ SUCCESS! DATASET ORGANIZED!")
    print("="*60)
    print(f"\n📁 Location: {output}/")
    print("\n✅ Ready to train!")
    print("   Run: python train.py --batch_size 16")

if __name__ == '__main__':
    try:
        organize_dataset()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        print("\nPlease share this error message so I can help!")
