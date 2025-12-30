import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC

class ModelBuilder:
    """
    Builds ResNet-50 based architecture for pneumonia detection.
    """
    
    @staticmethod
    def build(img_size=(224, 224), trainable_backbone=False, 
              fine_tune_from=None, model_name="PneumoniaNet"):
        """
        Build the complete model.
        """
        print(f"\n{'─'*70}\n🏗️ BUILDING MODEL\n{'─'*70}")
        
        # Load pretrained ResNet-50
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(*img_size, 3),
            pooling=None
        )
        
        print(f"✓ ResNet-50 loaded: {len(base_model.layers)} layers")
        
        # Configure trainability
        if not trainable_backbone:
            base_model.trainable = False
            print("🔒 Backbone: FROZEN")
        elif fine_tune_from is not None:
            base_model.trainable = True
            for layer in base_model.layers[:fine_tune_from]:
                layer.trainable = False
            print(f"🔓 Backbone: Unfrozen from layer {fine_tune_from}")
        else:
            base_model.trainable = True
            print("🔓 Backbone: FULLY UNFROZEN")
        
        # Build architecture
        inputs = keras.Input(shape=(*img_size, 3), name='input')
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        x = layers.Dense(512, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        x = layers.Dense(256, activation='relu', name='fc2')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=model_name)
        
        trainable = sum(1 for layer in model.layers if layer.trainable)
        total = len(model.layers)
        
        print(f"✓ Architecture: GAP → FC(512) → Drop → FC(256) → Drop → FC(1)")
        print(f"✓ Total parameters: {model.count_params():,}")
        print(f"✓ Trainable layers: {trainable}/{total}")
        
        return model
    
    @staticmethod
    def compile(model, learning_rate=1e-4):
        """
        Compile model with optimizer, loss, and metrics.
        """
        print(f"\n{'─'*70}\n⚙️ COMPILING MODEL\n{'─'*70}")
        
        metrics = [
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            AUC(name='auc_pr', curve='PR')
        ]
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=BinaryCrossentropy(from_logits=False),
            metrics=metrics
        )
        
        print(f"✓ Optimizer: Adam (lr={learning_rate})")
        print(f"✓ Loss: Binary Crossentropy")
        print(f"✓ Metrics: {[m.name for m in metrics]}")
