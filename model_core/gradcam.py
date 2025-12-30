import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class GradCAMVisualizer:
    """
    Robust Grad-CAM with both gradient-based and activation-based methods.
    """
    
    def __init__(self, model, layer_name='conv5_block3_out'):
        """Initialize Grad-CAM visualizer."""
        self.original_model = model
        self.layer_name = layer_name
        self.use_gradcam = False  # Flag to track which method works
        
        print(f"{'='*70}\n🔥 GRAD-CAM VISUALIZER (v5.0 - Hybrid)\n{'='*70}")
        
        # Try to build gradient model
        try:
            self.grad_model = self._build_flat_grad_model()
            print(f"✓ Target layer: {layer_name}")
            print(f"✓ Gradient model built")
            
            # Test if gradients work
            test_img = np.random.random((1, 224, 224, 3)).astype(np.float32)
            if self._test_gradients(test_img):
                print(f"✓ Gradients working - using true Grad-CAM")
                self.use_gradcam = True
            else:
                print(f"⚠️  Gradients not working - using activation-based CAM")
                self.use_gradcam = False
        except Exception as e:
            print(f"⚠️  Gradient model failed: {e}")
            print(f"   Using activation-based CAM instead")
            self.grad_model = None
            self.use_gradcam = False
        
        print(f"{'='*70}")
    
    def _build_flat_grad_model(self):
        """
        Build a completely flat model for gradient computation.
        Uses a functional approach to extract intermediate layer outputs.
        """
        # Get the ResNet50 base model from nested structure
        try:
            base_model = self.original_model.get_layer('resnet50')
        except:
            raise ValueError("Could not find 'resnet50' layer in model")
        
        # Find the target layer
        try:
            target_layer = base_model.get_layer(self.layer_name)
        except:
            raise ValueError(f"Could not find layer '{self.layer_name}' in ResNet50")
        
        # Strategy: Create intermediate model from base_model input to target layer
        conv_model = Model(
            inputs=base_model.input,
            outputs=target_layer.output
        )
        
        # Now build the gradient model
        model_input = self.original_model.input
        conv_output = conv_model(model_input)
        final_output = self.original_model(model_input)
        
        grad_model = Model(
            inputs=model_input,
            outputs=[conv_output, final_output]
        )
        
        print(f"  ✓ Built flat model with {len(grad_model.layers)} layers")
        return grad_model
    
    def _test_gradients(self, test_img):
        """Test if gradient computation works."""
        try:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(test_img, training=False)
                tape.watch(conv_outputs)
                if predictions.shape[-1] == 1:
                    score = predictions[:, 0]
                else:
                    score = predictions[:, 0]
            
            grads = tape.gradient(score, conv_outputs)
            return grads is not None
        except:
            return False
    
    def make_gradcam_heatmap(self, img_array):
        """Generate heatmap using either Grad-CAM or activation-based CAM."""
        if self.use_gradcam and self.grad_model is not None:
            return self._make_gradcam_heatmap(img_array)
        else:
            return self._make_activation_heatmap(img_array)
    
    def _make_gradcam_heatmap(self, img_array):
        """True Grad-CAM with gradients."""
        img_tensor = tf.cast(img_array, tf.float32)
        
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = self.grad_model(img_tensor, training=False)
            tape.watch(conv_outputs)
            
            if predictions.shape[-1] == 1:
                class_score = predictions[:, 0]
            else:
                top_pred_index = tf.argmax(predictions[0])
                class_score = predictions[:, top_pred_index]
        
        grads = tape.gradient(class_score, conv_outputs)
        
        if grads is None:
            return self._make_activation_heatmap(img_array)
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def _make_activation_heatmap(self, img_array):
        """Activation-based Class Activation Map (CAM)."""
        base_model = self.original_model.get_layer('resnet50')
        target_layer = base_model.get_layer(self.layer_name)
        
        feature_model = Model(
            inputs=base_model.input,
            outputs=target_layer.output
        )
        
        activations = feature_model.predict(img_array, verbose=0)[0]
        heatmap = np.mean(activations, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def overlay_heatmap(self, img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Overlay heatmap on original image."""
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        superimposed = cv2.addWeighted(
            img, 1 - alpha,
            heatmap_colored, alpha,
            0
        )
        return superimposed
    
    def visualize_sample(self, img_path, save_path=None):
        """Visualize Grad-CAM for a single image."""
        print(f"\\n🖼️  Analyzing: {Path(img_path).name}")
        
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array_processed = np.expand_dims(img_array / 255.0, axis=0)
        
        pred = self.original_model.predict(img_array_processed, verbose=0)[0][0]
        pred_class = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        
        heatmap = self.make_gradcam_heatmap(img_array_processed)
        superimposed = self.overlay_heatmap(img_array, heatmap)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_array.astype(np.uint8))
        axes[0].set_title('Original X-Ray', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(superimposed)
        axes[2].set_title(f'Overlay - {pred_class} ({pred:.3f})',
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Grad-CAM Analysis: {Path(img_path).name}',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  ✓ Prediction: {pred_class} (confidence: {pred:.3f})")
        return heatmap, superimposed
    
    def visualize_batch(self, test_generator, num_samples=8, save_path=None):
        """Visualize heatmaps for multiple samples."""
        print(f"\\n{'─'*70}\\n🖼️  BATCH VISUALIZATION ({num_samples} samples)\\n{'─'*70}")
        
        method = "Grad-CAM" if self.use_gradcam else "Activation-based CAM"
        print(f"Using method: {method}")
        
        test_generator.reset()
        images, labels = next(test_generator)
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        class_names = ['NORMAL', 'PNEUMONIA']
        success_count = 0
        
        for idx in range(num_samples):
            img = images[idx]
            true_label = int(labels[idx])
            
            img_processed = np.expand_dims(img, axis=0)
            pred = self.original_model.predict(img_processed, verbose=0)[0][0]
            pred_label = int(pred > 0.5)
            
            try:
                heatmap = self.make_gradcam_heatmap(img_processed)
                success_count += 1
            except Exception as e:
                print(f"⚠️  Sample {idx} failed: {str(e)[:80]}")
                heatmap = np.ones((7, 7)) * 0.5
            
            img_uint8 = (img * 255).astype(np.uint8)
            superimposed = self.overlay_heatmap(img_uint8, heatmap)
            
            axes[idx].imshow(superimposed)
            color = 'green' if pred_label == true_label else 'red'
            title = f"True: {class_names[true_label]}\\n"
            title += f"Pred: {class_names[pred_label]} ({pred:.3f})"
            axes[idx].set_title(title, fontsize=10, fontweight='bold', color=color)
            axes[idx].axis('off')
        
        # Hide empty subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')
        
        title = f'{method} Batch Visualization'
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\\n✓ Successfully generated {success_count}/{num_samples} heatmaps using {method}")
