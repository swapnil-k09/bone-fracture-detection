"""
Model Architecture Builder for X-ray Bone Fracture Detection
Provides different CNN architectures including custom and transfer learning options
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import (
    VGG16, ResNet50, DenseNet121, EfficientNetB0, InceptionV3
)
from typing import Tuple, Optional
import os


class FractureDetectionModel:
    """
    Builder class for fracture detection models
    Supports both custom CNN and transfer learning approaches
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 1),
                 num_classes: int = 1):
        """
        Initialize model builder
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes (1 for binary classification)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build_custom_cnn(self, 
                        filters: list = [32, 64, 128, 256],
                        dropout_rate: float = 0.5,
                        dense_units: int = 512) -> Model:
        """
        Build a custom CNN architecture
        
        Args:
            filters: Number of filters for each conv block
            dropout_rate: Dropout rate for regularization
            dense_units: Number of units in dense layer
            
        Returns:
            Compiled Keras model
        """
        print("🏗️  Building Custom CNN...")
        
        model = models.Sequential(name='CustomCNN')
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # Convolutional blocks
        for i, num_filters in enumerate(filters):
            # Conv layer
            model.add(layers.Conv2D(
                num_filters, 
                (3, 3), 
                activation='relu',
                padding='same',
                name=f'conv_{i+1}_1'
            ))
            model.add(layers.Conv2D(
                num_filters, 
                (3, 3), 
                activation='relu',
                padding='same',
                name=f'conv_{i+1}_2'
            ))
            
            # Batch normalization
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            
            # Max pooling
            model.add(layers.MaxPooling2D((2, 2), name=f'pool_{i+1}'))
            
            # Dropout
            model.add(layers.Dropout(dropout_rate * 0.5, name=f'dropout_{i+1}'))
        
        # Flatten
        model.add(layers.Flatten(name='flatten'))
        
        # Dense layers
        model.add(layers.Dense(dense_units, activation='relu', name='dense_1'))
        model.add(layers.BatchNormalization(name='bn_dense'))
        model.add(layers.Dropout(dropout_rate, name='dropout_dense'))
        
        model.add(layers.Dense(dense_units // 2, activation='relu', name='dense_2'))
        model.add(layers.Dropout(dropout_rate * 0.5, name='dropout_dense_2'))
        
        # Output layer
        if self.num_classes == 1:
            # Binary classification
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
        else:
            # Multi-class classification
            model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
        
        print(f"✅ Custom CNN built with {model.count_params():,} parameters")
        
        return model
    
    def build_transfer_learning_model(self,
                                     base_model_name: str = 'densenet121',
                                     trainable_layers: int = 0,
                                     dropout_rate: float = 0.5,
                                     dense_units: int = 256) -> Model:
        """
        Build a transfer learning model using pre-trained base
        
        Args:
            base_model_name: Name of base model ('vgg16', 'resnet50', 'densenet121', etc.)
            trainable_layers: Number of base layers to make trainable (0 = freeze all)
            dropout_rate: Dropout rate
            dense_units: Number of units in dense layer
            
        Returns:
            Compiled Keras model
        """
        print(f"🏗️  Building Transfer Learning Model: {base_model_name.upper()}...")
        
        # Adjust input shape for pre-trained models (they expect 3 channels)
        if self.input_shape[-1] == 1:
            # Convert grayscale to RGB by repeating channel
            input_layer = layers.Input(shape=self.input_shape)
            x = layers.Concatenate()([input_layer, input_layer, input_layer])
            rgb_shape = (self.input_shape[0], self.input_shape[1], 3)
        else:
            input_layer = layers.Input(shape=self.input_shape)
            x = input_layer
            rgb_shape = self.input_shape
        
        # Load pre-trained base model
        base_models = {
            'vgg16': VGG16,
            'resnet50': ResNet50,
            'densenet121': DenseNet121,
            'efficientnetb0': EfficientNetB0,
            'inceptionv3': InceptionV3
        }
        
        if base_model_name.lower() not in base_models:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        base_model_class = base_models[base_model_name.lower()]
        
        # Load with ImageNet weights
        base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_shape=rgb_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Make some layers trainable if specified
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
        
        # Build model
        x = base_model(x, training=False)
        
        # Add custom top layers
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(dropout_rate * 0.5, name='dropout_1')(x)
        
        x = layers.Dense(dense_units, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(dropout_rate, name='dropout_2')(x)
        
        x = layers.Dense(dense_units // 2, activation='relu', name='dense_2')(x)
        x = layers.Dropout(dropout_rate * 0.5, name='dropout_3')(x)
        
        # Output layer
        if self.num_classes == 1:
            output = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            output = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output, name=f'{base_model_name}_transfer')
        
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        total_params = model.count_params()
        
        print(f"✅ Transfer learning model built:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters: {total_params - trainable_params:,}")
        
        return model
    
    def compile_model(self,
                     model: Model,
                     learning_rate: float = 0.001,
                     metrics: list = None) -> Model:
        """
        Compile model with optimizer and metrics
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
            metrics: List of metrics to track
            
        Returns:
            Compiled model
        """
        if metrics is None:
            metrics = [
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        
        # Binary classification
        if self.num_classes == 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print("✅ Model compiled successfully!")
        
        return model
    
    def get_model(self,
                 model_type: str = 'densenet121',
                 compile_model: bool = True,
                 **kwargs) -> Model:
        """
        Get a model by type (convenience function)
        
        Args:
            model_type: Type of model ('custom', 'vgg16', 'resnet50', 'densenet121', etc.)
            compile_model: Whether to compile the model
            **kwargs: Additional arguments for model building
            
        Returns:
            Keras model
        """
        if model_type == 'custom':
            model = self.build_custom_cnn(**kwargs)
        else:
            model = self.build_transfer_learning_model(
                base_model_name=model_type,
                **kwargs
            )
        
        if compile_model:
            model = self.compile_model(model)
        
        return model


def create_callbacks(model_dir: str = 'models',
                    patience: int = 10,
                    reduce_lr_patience: int = 5) -> list:
    """
    Create training callbacks
    
    Args:
        model_dir: Directory to save models
        patience: Early stopping patience
        reduce_lr_patience: Learning rate reduction patience
        
    Returns:
        List of callbacks
    """
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/tensorboard',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # CSV logger
        tf.keras.callbacks.CSVLogger(
            'logs/training_log.csv',
            append=True
        )
    ]
    
    return callbacks


def print_model_summary(model: Model, save_path: Optional[str] = None):
    """
    Print and optionally save model summary
    
    Args:
        model: Keras model
        save_path: Path to save summary (optional)
    """
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    model.summary()
    print("="*70 + "\n")
    
    if save_path:
        with open(save_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"✅ Model summary saved to: {save_path}")


if __name__ == '__main__':
    print("="*70)
    print("FRACTURE DETECTION MODEL BUILDER")
    print("="*70 + "\n")
    
    # Initialize builder
    builder = FractureDetectionModel(input_shape=(224, 224, 1))
    
    print("Available models:")
    print("1. Custom CNN")
    print("2. VGG16 (Transfer Learning)")
    print("3. ResNet50 (Transfer Learning)")
    print("4. DenseNet121 (Transfer Learning) ⭐ RECOMMENDED")
    print("5. EfficientNetB0 (Transfer Learning)")
    print("6. InceptionV3 (Transfer Learning)")
    
    print("\n" + "="*70)
    print("Example 1: Custom CNN")
    print("="*70)
    custom_model = builder.get_model('custom')
    print_model_summary(custom_model)
    
    print("\n" + "="*70)
    print("Example 2: DenseNet121 Transfer Learning")
    print("="*70)
    densenet_model = builder.get_model('densenet121')
    print_model_summary(densenet_model)
    
    print("\n✅ Model builder ready!")
    print("Usage:")
    print("  builder = FractureDetectionModel()")
    print("  model = builder.get_model('densenet121')")
