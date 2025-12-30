import sys
import os
# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_core.data_pipeline import DataPipeline
from model_core.model_builder import ModelBuilder
from model_core.trainer import Trainer
from model_core.utils import Utils

def run_pipeline(dataset_path, stage1_epochs=8, stage2_epochs=5, output_dir="outputs"):
    """
    Execute complete training pipeline.
    """
    print(f"\n{'='*70}\n🚀 PEDIATRIC PNEUMONIA DETECTION PIPELINE\n{'='*70}")
    
    # =========================================================================
    # STAGE 0: DATA PREPARATION
    # =========================================================================
    print("\n📊 PREPARING DATA...")
    
    pipeline = DataPipeline(dataset_path, img_size=(224, 224), batch_size=32)
    pipeline.explore_dataset()
    pipeline.create_validation_split(val_ratio=0.15)
    train_gen, val_gen, test_gen = pipeline.create_generators(use_augmentation=True)
    # pipeline.visualize_samples() # Skipping visualization in script mode
    
    # Calculate class weights
    class_weights = Utils.calculate_class_weights(train_gen)
    
    # =========================================================================
    # STAGE 1: FEATURE EXTRACTION
    # =========================================================================
    print("\n🔵 STAGE 1: FEATURE EXTRACTION")
    
    model = ModelBuilder.build(trainable_backbone=False)
    ModelBuilder.compile(model, learning_rate=1e-4)
    
    trainer = Trainer(model, output_dir=output_dir)
    history1 = trainer.train(
        train_gen, val_gen, 
        epochs=stage1_epochs,
        stage='stage1',
        class_weight=class_weights
    )
    
    trainer.plot_history(history1, 'stage1')
    Utils.print_best_metrics(history1, 'Stage 1')
    
    # =========================================================================
    # STAGE 2: FINE-TUNING
    # =========================================================================
    print("\n🔴 STAGE 2: FINE-TUNING")
    
    # Unfreeze layers
    base_model = model.layers[1]
    base_model.trainable = True
    for layer in base_model.layers[:140]:
        layer.trainable = False
    
    ModelBuilder.compile(model, learning_rate=1e-5)
    
    history2 = trainer.train(
        train_gen, val_gen,
        epochs=stage2_epochs,
        stage='stage2',
        class_weight=class_weights
    )
    
    trainer.plot_history(history2, 'stage2')
    Utils.print_best_metrics(history2, 'Stage 2')
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    print(f"\n{'='*70}\n🎉 PIPELINE COMPLETE!\n{'='*70}")
    print(f"✓ Total images: {train_gen.samples + val_gen.samples + test_gen.samples:,}")
    print(f"✓ Model parameters: {model.count_params():,}")
    print(f"✓ Output: {trainer.output_dir}")
    
    return trainer.output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Pneumonia Detection Model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to chest X-ray dataset')
    parser.add_argument('--stage1_epochs', type=int, default=8, help='Epochs for stage 1')
    parser.add_argument('--stage2_epochs', type=int, default=5, help='Epochs for stage 2')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    run_pipeline(
        dataset_path=args.dataset_path,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        output_dir=args.output_dir
    )
