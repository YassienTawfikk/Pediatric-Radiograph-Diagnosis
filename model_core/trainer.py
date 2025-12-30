import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    CSVLogger, TensorBoard
)

class Trainer:
    """
    Manages two-stage training process.
    """
    
    def __init__(self, model, output_dir="outputs"):
        self.model = model
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"run_{self.timestamp}"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Output directory: {self.output_dir}")
    
    def create_callbacks(self, stage='stage1', monitor='val_auc', patience=10):
        """Create training callbacks."""
        callbacks = [
            ModelCheckpoint(
                filepath=str(self.checkpoint_dir / f"{stage}_best.h5"),
                monitor=monitor,
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor=monitor,
                mode='max',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(str(self.logs_dir / f"{stage}_log.csv"), append=True),
            TensorBoard(log_dir=str(self.logs_dir / stage / 'tensorboard'))
        ]
        
        print(f"✓ Callbacks configured for {stage}")
        return callbacks
    
    def train(self, train_gen, val_gen, epochs, stage='stage1', 
              class_weight=None, monitor='val_auc'):
        """
        Execute training for one stage.
        """
        print(f"\n{'='*70}\n🚀 {stage.upper()}\n{'='*70}")
        
        callbacks = self.create_callbacks(stage, monitor, patience=10)
        
        print(f"⏱️ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Save final model
        final_path = self.checkpoint_dir / f"{stage}_final.h5"
        self.model.save(str(final_path))
        print(f"\n💾 Model saved: {final_path}")
        
        return history
    
    def plot_history(self, history, stage='stage1'):
        """Visualize training metrics."""
        metrics = ['loss', 'accuracy', 'auc', 'recall']
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        
        for idx, metric in enumerate(metrics):
            if metric in history.history:
                epochs = range(1, len(history.history[metric]) + 1)
                axes[idx].plot(epochs, history.history[metric], 
                             'b-o', label='Train')
                
                val_metric = f'val_{metric}'
                if val_metric in history.history:
                    axes[idx].plot(epochs, history.history[val_metric],
                                 'r-s', label='Val')
                
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel(metric.capitalize())
                axes[idx].set_title(f'{stage.upper()} - {metric.capitalize()}')
                axes[idx].legend()
                axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{stage}_history.png", dpi=200)
        plt.show()
