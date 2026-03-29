"""
Grad-CAM Implementation - Bulletproof Version
Works with any CNN model architecture, including nested sub-models
"""

import traceback
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2


class GradCAM:
    """Grad-CAM visualization that works with any model"""

    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        print(f"✅ Grad-CAM initialized with layer: {self.layer_name}")

    def _find_target_layer(self):
        """Find the last convolutional layer, including inside nested sub-models."""
        # Check direct layers first
        for layer in reversed(self.model.layers):
            try:
                if len(layer.output_shape) == 4:
                    return layer.name
            except AttributeError:
                pass
            # Also search inside nested sub-models
            if hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                    try:
                        if len(sub_layer.output_shape) == 4:
                            return sub_layer.name
                    except AttributeError:
                        pass
        raise ValueError("No convolutional layer found in model or sub-models")

    def _get_conv_layer(self):
        """Get the target layer, searching nested sub-models if needed."""
        try:
            return self.model.get_layer(self.layer_name)
        except ValueError:
            for layer in self.model.layers:
                if hasattr(layer, 'get_layer'):
                    try:
                        return layer.get_layer(self.layer_name)
                    except ValueError:
                        continue
        raise ValueError(f"Layer '{self.layer_name}' not found in model or sub-models")

    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap.
        Uses layer call patching to avoid keras.Model graph issues with nested models.
        """
        # Normalize input shape
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) == 3:
            # Only transpose if clearly channel-first format
            if image.shape[0] in (1, 3) and image.shape[1] > 4 and image.shape[2] > 4:
                image = np.transpose(image, (1, 2, 0))
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) == 4 and image.shape[0] != 1:
            image = image[:1]

        # Use tf.Variable so GradientTape tracks all ops automatically
        image_var = tf.Variable(tf.cast(image, tf.float32))

        # Patch the target layer's call() to capture its output tensor
        conv_layer = self._get_conv_layer()
        captured = []
        original_call = conv_layer.call

        def capturing_call(inputs, **kwargs):
            output = original_call(inputs, **kwargs)
            captured.append(output)
            return output

        conv_layer.call = capturing_call
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(image_var, training=False)

                if not captured:
                    raise ValueError(
                        f"Layer '{self.layer_name}' was never called during forward pass. "
                        "Try specifying a different layer name."
                    )

                conv_outputs = captured[0]

                # Compute loss
                if len(predictions.shape) == 2 and predictions.shape[-1] == 1:
                    loss = predictions[0][0]
                else:
                    idx = class_idx if class_idx is not None else tf.argmax(predictions[0])
                    loss = predictions[0][idx]
        finally:
            conv_layer.call = original_call  # Always restore

        # Gradients of loss w.r.t. conv layer outputs
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise ValueError(
                f"Gradients are None for layer '{self.layer_name}'. "
                "The layer may not contribute to the output. Try a different layer."
            )

        # Pool gradients over spatial dims, weight channels, average
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_out_np = conv_outputs[0].numpy()

        for i in range(pooled_grads.shape[0]):
            conv_out_np[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_out_np, axis=-1)

        # ReLU + normalize
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > eps:
            heatmap = heatmap / max_val

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Overlay heatmap on image"""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        # applyColorMap returns BGR; convert to RGB to match typical numpy pipelines
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Convert grayscale image to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)

        # Ensure uint8
        if image.dtype != np.uint8:
            image = np.uint8(255 * image) if image.max() <= 1.0 else np.uint8(image)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

    def generate_visualization(self, image, original_image=None, class_idx=None,
                               alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Generate complete Grad-CAM visualization"""
        try:
            heatmap = self.compute_heatmap(image, class_idx)

            if original_image is None:
                if len(image.shape) == 4:
                    original_image = image[0]
                    # Remove single channel/spatial dims carefully
                    if original_image.shape[-1] == 1:
                        original_image = original_image.squeeze(axis=-1)
                elif len(image.shape) == 3:
                    original_image = image
                else:
                    original_image = image

            overlay = self.overlay_heatmap(heatmap, original_image, alpha, colormap)
            return heatmap, overlay

        except Exception as e:
            print(f"❌ Grad-CAM generation failed: {e}")
            traceback.print_exc()
            raise
