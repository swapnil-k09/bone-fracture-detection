"""
Training Script for X-ray Bone Fracture Detection Model
Handles data loading, model training, and evaluation
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

# Import our utilities
from utils.model_builder import FractureDetectionModel, create_callbacks, print_model_summary
from utils.data_loader import DatasetLoader
from config import *


def create_data_generators(train_dir, val_dir, 
                          batch_size=32,
                          target_size=(224, 224),
                          augment=True):
    """
    Create data generators for training and validation
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        target_size: Target image size
        augment: Whether to apply augmentation
        
    Returns:
        train_generator, validation_generator
    """
    print("\n📊 Creating data generators...")
    
    if augment:
        # Training data with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='constant',
            cval=0
        )
    else:
        # Training data without augmentation
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Validation data (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=True,
        seed=42
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False,
        seed=42
    )
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {validation_generator.samples}")
    print(f"✅ Classes: {train_generator.class_indices}")
    
    return train_generator, validation_generator


def train_model(model_type='densenet121',
               epochs=50,
               batch_size=32,
               learning_rate=0.001,
               use_augmentation=True,
               model_dir='models',
               data_dir='data'):
    """
    Main training function
    
    Args:
        model_type: Type of model to train
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        use_augmentation: Whether to use data augmentation
        model_dir: Directory to save models
        data_dir: Root data directory
    """
    print("\n" + "="*70)
    print("FRACTURE DETECTION MODEL TRAINING")
    print("="*70)
    
    # Print configuration
    print(f"\n⚙️  Configuration:")
    print(f"   Model Type: {model_type}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Augmentation: {use_augmentation}")
    print(f"   Data Directory: {data_dir}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🎮 GPU Available: {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"   {gpu.name}")
    else:
        print("\n💻 No GPU detected - training will use CPU")
    
    # Create data generators
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    
    # Check if data exists
    if not os.path.exists(train_dir):
        print(f"\n❌ ERROR: Training directory not found: {train_dir}")
        print("\n📥 Please download the dataset first!")
        print("   Options:")
        print("   1. MURA dataset: https://stanfordmlgroup.github.io/competitions/mura/")
        print("   2. Kaggle dataset: https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays")
        print("\n   Extract to the 'data/' directory with this structure:")
        print("   data/")
        print("   ├── train/")
        print("   │   ├── fractured/")
        print("   │   └── normal/")
        print("   └── validation/")
        print("       ├── fractured/")
        print("       └── normal/")
        return
    
    train_gen, val_gen = create_data_generators(
        train_dir, val_dir,
        batch_size=batch_size,
        target_size=IMAGE_SIZE,
        augment=use_augmentation
    )
    
    # Build model
    print(f"\n🏗️  Building {model_type.upper()} model...")
    builder = FractureDetectionModel(
        input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS),
        num_classes=1
    )
    
    if model_type == 'custom':
        model = builder.build_custom_cnn(
            dropout_rate=DROPOUT_RATE,
            dense_units=DENSE_UNITS
        )
    else:
        model = builder.build_transfer_learning_model(
            base_model_name=model_type,
            dropout_rate=DROPOUT_RATE,
            dense_units=DENSE_UNITS
        )
    
    # Compile model
    model = builder.compile_model(model, learning_rate=learning_rate)
    
    # Print summary
    print_model_summary(model, save_path=os.path.join(model_dir, 'model_summary.txt'))
    
    # Create callbacks
    print("\n📋 Setting up callbacks...")
    callbacks = create_callbacks(
        model_dir=model_dir,
        patience=EARLY_STOPPING_PATIENCE,
        reduce_lr_patience=REDUCE_LR_PATIENCE
    )
    
    # Calculate steps
    steps_per_epoch = train_gen.samples // batch_size
    validation_steps = val_gen.samples // batch_size
    
    print(f"\n🎯 Training configuration:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    
    # Train model
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
    try:
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED!")
        print("="*70)
        print(f"Duration: {duration}")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        # Save final model
        final_model_path = os.path.join(model_dir, f'{model_type}_final.h5')
        model.save(final_model_path)
        print(f"\n💾 Final model saved: {final_model_path}")
        
        # Save training history
        import pickle
        history_path = os.path.join(model_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"📊 Training history saved: {history_path}")
        
        return model, history
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("Best model should be saved in:", os.path.join(model_dir, 'best_model.h5'))
        return None, None
    
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Train bone fracture detection model'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='densenet121',
        choices=['custom', 'vgg16', 'resnet50', 'densenet121', 'efficientnetb0'],
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=0.001,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--no_augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        default='models',
        help='Directory to save models'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Root data directory'
    )
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_augmentation=not args.no_augmentation,
        model_dir=args.model_dir,
        data_dir=args.data_dir
    )


if __name__ == '__main__':
    main()
