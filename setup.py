#!/usr/bin/env python3
"""
Setup script for Bone Fracture Detection System
This script helps set up the project environment
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher is required!")
        print("Please upgrade your Python installation.")
        return False
    
    print("✅ Python version is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print_header("Creating Virtual Environment")
    
    venv_name = "fracture_env"
    
    if os.path.exists(venv_name):
        print(f"ℹ️  Virtual environment '{venv_name}' already exists")
        response = input("Do you want to recreate it? (y/n): ")
        if response.lower() == 'y':
            print(f"Removing existing '{venv_name}'...")
            import shutil
            shutil.rmtree(venv_name)
        else:
            print("Skipping virtual environment creation")
            return True
    
    print(f"Creating virtual environment '{venv_name}'...")
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
        print(f"✅ Virtual environment '{venv_name}' created successfully")
        
        # Print activation instructions
        print("\n📋 To activate the virtual environment:")
        if platform.system() == "Windows":
            print(f"   {venv_name}\\Scripts\\activate")
        else:
            print(f"   source {venv_name}/bin/activate")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False

def install_dependencies():
    """Install required packages"""
    print_header("Installing Dependencies")
    
    if not os.path.exists("requirements.txt"):
        print("❌ ERROR: requirements.txt not found!")
        return False
    
    print("Installing packages from requirements.txt...")
    print("This may take several minutes...\n")
    
    try:
        # Use pip from the current Python interpreter
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("\n✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        print("You may need to install some packages manually")
        return False

def create_directories():
    """Create necessary project directories"""
    print_header("Creating Project Directories")
    
    directories = [
        'data/train/fractured',
        'data/train/normal',
        'data/validation/fractured',
        'data/validation/normal',
        'data/test/fractured',
        'data/test/normal',
        'models',
        'uploads',
        'logs',
        'logs/tensorboard',
        'static/css',
        'static/js',
        'templates',
        'utils',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("\n✅ All directories created successfully")
    return True

def create_gitkeep_files():
    """Create .gitkeep files in empty directories"""
    print_header("Creating .gitkeep Files")
    
    directories = [
        'uploads',
        'models',
        'data/train/fractured',
        'data/train/normal',
        'data/validation/fractured',
        'data/validation/normal',
        'data/test/fractured',
        'data/test/normal'
    ]
    
    for directory in directories:
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                f.write('')
            print(f"✅ Created: {gitkeep_path}")
    
    return True

def check_gpu():
    """Check GPU availability"""
    print_header("Checking GPU Availability")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) detected:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("ℹ️  No GPU detected - will use CPU for training")
            print("   (Training will be slower, but it will work)")
        
        return True
    except ImportError:
        print("⚠️  TensorFlow not installed yet - GPU check will be done later")
        return True

def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete! 🎉")
    
    print("Next steps:")
    print()
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   fracture_env\\Scripts\\activate")
    else:
        print("   source fracture_env/bin/activate")
    print()
    print("2. Download the MURA dataset:")
    print("   Visit: https://stanfordmlgroup.github.io/competitions/mura/")
    print("   Extract to the 'data/' directory")
    print()
    print("3. Verify your setup:")
    print("   python config.py")
    print()
    print("4. Start with data exploration:")
    print("   jupyter notebook notebooks/01_data_exploration.ipynb")
    print()
    print("5. Begin training:")
    print("   python train.py")
    print()
    print("For more information, see README.md")
    print()

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("  Bone Fracture Detection System - Setup Script")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Create .gitkeep files
    create_gitkeep_files()
    
    # Ask if user wants to create virtual environment
    print()
    response = input("Do you want to create a virtual environment? (recommended) (y/n): ")
    if response.lower() == 'y':
        if not create_virtual_environment():
            print("\n⚠️  Virtual environment setup failed")
            print("You can create it manually later with: python -m venv fracture_env")
    
    # Ask if user wants to install dependencies
    print()
    response = input("Do you want to install dependencies now? (y/n): ")
    if response.lower() == 'y':
        if not install_dependencies():
            print("\n⚠️  Dependency installation had issues")
            print("You can install manually with: pip install -r requirements.txt")
        else:
            # Check GPU if TensorFlow was installed
            check_gpu()
    
    # Print next steps
    print_next_steps()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
