"""
Model Evaluation Script for Bone Fracture Detection
Evaluates trained model performance and generates metrics
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve
)
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def evaluate_model(model_path, test_dir, batch_size=32, target_size=(224, 224)):
    """
    Evaluate model on test set
    
    Args:
        model_path: Path to saved model
        test_dir: Test data directory
        batch_size: Batch size for evaluation
        target_size: Target image size
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Load model
    print(f"📦 Loading model from: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # Create test generator
    print(f"\n📊 Loading test data from: {test_dir}")
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False
    )
    
    print(f"✅ Test samples: {test_generator.samples}")
    print(f"✅ Classes: {test_generator.class_indices}")
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Evaluate
    print("\n🔍 Evaluating model...")
    results = model.evaluate(test_generator, verbose=1)
    
    metrics_names = model.metrics_names
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    for name, value in zip(metrics_names, results):
        print(f"{name:15s}: {value:.4f}")
    print("="*70 + "\n")
    
    # Get predictions
    print("🎯 Generating predictions...")
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    
    # Get true labels
    y_true = test_generator.classes
    y_pred_proba = predictions.flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate additional metrics
    print("\n📈 Calculating detailed metrics...")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification Report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Compile results
    evaluation_results = {
        'model_metrics': dict(zip(metrics_names, results)),
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
        'pr_curve': {'precision': precision, 'recall': recall, 'auc': pr_auc},
        'predictions': {'y_true': y_true, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba},
        'class_names': class_names
    }
    
    return evaluation_results


def print_classification_report(results):
    """Print detailed classification report"""
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70 + "\n")
    
    report = results['classification_report']
    class_names = results['class_names']
    
    # Print per-class metrics
    for class_name in class_names:
        metrics = report[class_name]
        print(f"{class_name.upper()}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1-score']:.4f}")
        print(f"  Support:   {metrics['support']}")
        print()
    
    # Print overall metrics
    print("OVERALL:")
    print(f"  Accuracy:  {report['accuracy']:.4f}")
    print(f"  Macro Avg: {report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted Avg: {report['weighted avg']['f1-score']:.4f}")
    print("="*70 + "\n")


def plot_confusion_matrix(results, save_path=None):
    """Plot confusion matrix"""
    cm = results['confusion_matrix']
    class_names = results['class_names']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")
    
    plt.show()


def plot_roc_curve(results, save_path=None):
    """Plot ROC curve"""
    roc_data = results['roc_curve']
    
    plt.figure(figsize=(8, 6))
    plt.plot(
        roc_data['fpr'], 
        roc_data['tpr'], 
        color='darkorange', 
        lw=2,
        label=f'ROC curve (AUC = {roc_data["auc"]:.4f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ ROC curve saved: {save_path}")
    
    plt.show()


def plot_precision_recall_curve(results, save_path=None):
    """Plot Precision-Recall curve"""
    pr_data = results['pr_curve']
    
    plt.figure(figsize=(8, 6))
    plt.plot(
        pr_data['recall'], 
        pr_data['precision'], 
        color='blue', 
        lw=2,
        label=f'PR curve (AUC = {pr_data["auc"]:.4f})'
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ PR curve saved: {save_path}")
    
    plt.show()


def plot_prediction_distribution(results, save_path=None):
    """Plot distribution of prediction probabilities"""
    y_true = results['predictions']['y_true']
    y_pred_proba = results['predictions']['y_pred_proba']
    class_names = results['class_names']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot for class 0 (normal)
    normal_probs = y_pred_proba[y_true == 0]
    ax1.hist(normal_probs, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'{class_names[0].upper()} Cases', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot for class 1 (fractured)
    fractured_probs = y_pred_proba[y_true == 1]
    ax2.hist(fractured_probs, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'{class_names[1].upper()} Cases', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Distribution of Prediction Probabilities', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Prediction distribution saved: {save_path}")
    
    plt.show()


def create_evaluation_report(results, output_dir='reports'):
    """Create comprehensive evaluation report with visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n📊 Creating evaluation report...")
    
    # Print classification report
    print_classification_report(results)
    
    # Generate plots
    print("\n📈 Generating visualizations...")
    plot_confusion_matrix(results, save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    plot_roc_curve(results, save_path=os.path.join(output_dir, 'roc_curve.png'))
    plot_precision_recall_curve(results, save_path=os.path.join(output_dir, 'pr_curve.png'))
    plot_prediction_distribution(results, save_path=os.path.join(output_dir, 'prediction_distribution.png'))
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("BONE FRACTURE DETECTION - EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL METRICS:\n")
        for name, value in results['model_metrics'].items():
            f.write(f"  {name:15s}: {value:.4f}\n")
        f.write("\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        report = results['classification_report']
        for class_name in results['class_names']:
            metrics = report[class_name]
            f.write(f"\n{class_name.upper()}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1-score']:.4f}\n")
            f.write(f"  Support:   {metrics['support']}\n")
        
        f.write(f"\nOVERALL:\n")
        f.write(f"  Accuracy:  {report['accuracy']:.4f}\n")
        f.write(f"  ROC AUC:   {results['roc_curve']['auc']:.4f}\n")
        f.write(f"  PR AUC:    {results['pr_curve']['auc']:.4f}\n")
    
    print(f"✅ Evaluation metrics saved: {metrics_path}")
    print(f"\n✅ Full report saved to: {output_dir}/")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Evaluate bone fracture detection model'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model.h5',
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--test_dir',
        type=str,
        default='data/test',
        help='Test data directory'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='reports',
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"\n❌ ERROR: Model not found: {args.model}")
        print("\n💡 Please train a model first:")
        print("   python train.py")
        return
    
    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"\n❌ ERROR: Test directory not found: {args.test_dir}")
        print("\n📥 Please ensure test data is available in the correct location")
        return
    
    # Evaluate model
    results = evaluate_model(
        model_path=args.model,
        test_dir=args.test_dir,
        batch_size=args.batch_size
    )
    
    # Create report
    create_evaluation_report(results, output_dir=args.output_dir)
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
