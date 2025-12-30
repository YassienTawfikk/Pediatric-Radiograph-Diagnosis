from collections import Counter

class Utils:
    """Helper functions for the pipeline."""
    
    @staticmethod
    def calculate_class_weights(train_gen):
        """Calculate class weights for imbalanced datasets."""
        print(f"\n{'─'*70}\n⚖️ CALCULATING CLASS WEIGHTS\n{'─'*70}")
        
        class_counts = Counter(train_gen.classes)
        total = sum(class_counts.values())
        n_classes = len(class_counts)
        
        weights = {
            cls: total / (n_classes * count)
            for cls, count in class_counts.items()
        }
        
        for name, idx in train_gen.class_indices.items():
            print(f"  • {name}: {class_counts[idx]:,} samples (weight: {weights[idx]:.4f})")
        
        return weights
    
    @staticmethod
    def print_best_metrics(history, stage='Stage'):
        """Print best validation metrics."""
        print(f"\n📈 {stage} Best Metrics:")
        
        for key in ['val_loss', 'val_accuracy', 'val_auc', 'val_recall', 'val_precision']:
            if key in history.history:
                values = history.history[key]
                best = min(values) if 'loss' in key else max(values)
                epoch = values.index(best) + 1
                print(f"  • {key}: {best:.4f} (epoch {epoch})")
