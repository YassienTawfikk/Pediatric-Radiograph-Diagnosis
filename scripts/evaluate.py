import sys
import os
# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_core.evaluator import ModelEvaluator
from model_core.gradcam import GradCAMVisualizer

def run_evaluation(model_path, dataset_path, output_dir="evaluation_results"):
    """Execute complete evaluation pipeline."""
    print(f"\n{'='*70}\n🚀 EVALUATION & EXPLAINABILITY PIPELINE\n{'='*70}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n📥 Loading model...")
    model = load_model(model_path)
    print(f"✓ Model loaded: {model.name}")
    print(f"  • Total layers: {len(model.layers)}")
    print(f"  • Parameters: {model.count_params():,}")
    
    # Prepare test data
    print("\n📊 Preparing test data...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        str(Path(dataset_path) / 'test'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    print(f"✓ Test samples: {test_gen.samples}")
    
    # EVALUATION
    print(f"\n{'='*70}\n📈 EVALUATION PHASE\n{'='*70}")
    
    evaluator = ModelEvaluator(model, test_gen)
    evaluator.generate_predictions()
    metrics = evaluator.calculate_metrics()
    evaluator.plot_confusion_matrix(save_path=output_path / "confusion_matrix.png")
    evaluator.plot_roc_curve(save_path=output_path / "roc_curve.png")
    evaluator.plot_precision_recall_curve(save_path=output_path / "pr_curve.png")
    report = evaluator.generate_classification_report()
    
    # GRAD-CAM
    print(f"\n{'='*70}\n🔥 GRAD-CAM PHASE\n{'='*70}")
    
    try:
        gradcam = GradCAMVisualizer(model)
        gradcam.visualize_batch(
            test_gen,
            num_samples=8,
            save_path=output_path / "gradcam_batch.png"
        )
    except Exception as e:
        print(f"❌ Grad-CAM failed: {e}")
        print(f"   This might be due to model architecture incompatibility")
        gradcam = None
    
    # SAVE RESULTS
    print(f"\n{'='*70}\n💾 SAVING RESULTS\n{'='*70}")
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path / "metrics.csv", index=False)
    print(f"✓ Metrics: {output_path / 'metrics.csv'}")
    
    with open(output_path / "classification_report.txt", 'w') as f:
        f.write(report)
    print(f"✓ Report: {output_path / 'classification_report.txt'}")
    
    print(f"\n{'='*70}\n✅ EVALUATION COMPLETE!\n{'='*70}")
    print(f"📁 Results saved to: {output_path}")
    
    return evaluator, gradcam, metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Pneumonia Detection Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.h5)')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to chest X-ray dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir
    )
