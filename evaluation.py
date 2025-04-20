from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import numpy as np

def evaluate_model(y_true, y_pred, y_prob, target_names):
    """
    Evaluate the model using various metrics.
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        target_names: Names of the target classes
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Calculate ROC AUC (for binary classification)
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=target_names)
    
    # Print results
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(report)
    
    # Return metrics as a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics 