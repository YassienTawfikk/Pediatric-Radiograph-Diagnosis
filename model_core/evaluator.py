import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)

class ModelEvaluator:
    """Comprehensive evaluation suite for pneumonia detection model."""
    
    def __init__(self, model, test_generator):
        self.model = model
        self.test_gen = test_generator
        self.predictions = None
        self.y_true = None
        self.y_pred_proba = None
        
        print(f"{'='*70}\n📊 MODEL EVALUATOR INITIALIZED\n{'='*70}")
        print(f"✓ Model: {model.name}")
        print(f"✓ Test samples: {test_generator.samples}")
        print(f"✓ Classes: {test_generator.class_indices}\n{'='*70}")
    
    def generate_predictions(self):
        """Generate predictions on test set."""
        print(f"\n{'─'*70}\n🔮 GENERATING PREDICTIONS\n{'─'*70}")
        
        self.test_gen.reset()
        
        self.y_pred_proba = self.model.predict(
            self.test_gen,
            verbose=1,
            steps=len(self.test_gen)
        )
        
        self.y_true = self.test_gen.classes
        self.predictions = (self.y_pred_proba > 0.5).astype(int).flatten()
        
        print(f"✓ Predictions shape: {self.y_pred_proba.shape}")
        print(f"✓ Positive predictions: {self.predictions.sum()}/{len(self.predictions)}")
        
        return self.predictions, self.y_true, self.y_pred_proba
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics."""
        print(f"\n{'─'*70}\n📈 CALCULATING METRICS\n{'─'*70}")
        
        if self.predictions is None:
            self.generate_predictions()
        
        accuracy = accuracy_score(self.y_true, self.predictions)
        precision = precision_score(self.y_true, self.predictions)
        recall = recall_score(self.y_true, self.predictions)
        f1 = f1_score(self.y_true, self.predictions)
        auc_roc = roc_auc_score(self.y_true, self.y_pred_proba)
        auc_pr = average_precision_score(self.y_true, self.y_pred_proba)
        
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.predictions).ravel()
        specificity = tn / (tn + fp)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall (Sensitivity)': recall,
            'Specificity': specificity,
            'F1-Score': f1,
            'AUC-ROC': auc_roc,
            'AUC-PR': auc_pr
        }
        
        print("\n🎯 Performance Metrics:")
        print("─" * 50)
        for metric, value in metrics.items():
            print(f"  {metric:.<30} {value:.4f} ({value*100:.2f}%)")
        print("─" * 50)
        
        self._interpret_metrics(metrics)
        return metrics
    
    def _interpret_metrics(self, metrics):
        """Provide medical context for metrics."""
        print("\n🏥 Medical Interpretation:")
        print("─" * 50)
        
        recall = metrics['Recall (Sensitivity)']
        specificity = metrics['Specificity']
        precision = metrics['Precision']
        
        if recall >= 0.95:
            print("  ✅ Excellent sensitivity - Few missed pneumonia cases")
        elif recall >= 0.85:
            print("  ⚠️  Good sensitivity - Some cases may be missed")
        else:
            print("  ❌ Low sensitivity - Risk of missing pneumonia")
        
        if specificity >= 0.90:
            print("  ✅ Excellent specificity - Few false alarms")
        elif specificity >= 0.80:
            print("  ⚠️  Good specificity - Some false positives")
        else:
            print("  ❌ Low specificity - Many false alarms")
        
        if precision >= 0.90:
            print("  ✅ High precision - Reliable positive predictions")
        else:
            print("  ⚠️  Moderate precision - Verify positives")
        print("─" * 50)
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix."""
        print(f"\n{'─'*70}\n🎨 PLOTTING CONFUSION MATRIX\n{'─'*70}")
        
        if self.predictions is None:
            self.generate_predictions()
        
        cm = confusion_matrix(self.y_true, self.predictions)
        class_names = ['NORMAL', 'PNEUMONIA']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, ax=ax,
            annot_kws={'size': 16, 'weight': 'bold'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / cm[i].sum() * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                       ha='center', va='center', fontsize=12, color='gray')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        print(f"\n📋 Confusion Matrix Breakdown:")
        print("─" * 50)
        print(f"  True Negatives:  {tn:4d} ({tn/total*100:5.1f}%)")
        print(f"  False Positives: {fp:4d} ({fp/total*100:5.1f}%)")
        print(f"  False Negatives: {fn:4d} ({fn/total*100:5.1f}%)")
        print(f"  True Positives:  {tp:4d} ({tp/total*100:5.1f}%)")
        print("─" * 50)
        
        return cm
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve."""
        print(f"\n{'─'*70}\n📉 PLOTTING ROC CURVE\n{'─'*70}")
        
        if self.y_pred_proba is None:
            self.generate_predictions()
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ AUC-ROC: {roc_auc:.4f}")
        return fpr, tpr, roc_auc
    
    def plot_precision_recall_curve(self, save_path=None):
        """Plot Precision-Recall curve."""
        print(f"\n{'─'*70}\n📉 PLOTTING PR CURVE\n{'─'*70}")
        
        if self.y_pred_proba is None:
            self.generate_predictions()
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        ap_score = average_precision_score(self.y_true, self.y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {ap_score:.4f})')
        ax.axhline(y=self.y_true.mean(), color='red', linestyle='--',
                   label=f'Baseline = {self.y_true.mean():.3f}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=12)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Average Precision: {ap_score:.4f}")
        return precision, recall, ap_score
    
    def generate_classification_report(self):
        """Generate sklearn classification report."""
        print(f"\n{'─'*70}\n📄 CLASSIFICATION REPORT\n{'─'*70}")
        
        if self.predictions is None:
            self.generate_predictions()
        
        report = classification_report(
            self.y_true, self.predictions,
            target_names=['NORMAL', 'PNEUMONIA'],
            digits=4
        )
        print(report)
        return report
