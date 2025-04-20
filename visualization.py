import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, target_names):
    """
    Plot the confusion matrix.
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of the target classes
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for the Naive Bayes model.
    Args:
        model: Trained Naive Bayes model
        feature_names: Names of the features
    """
    # Get the standard deviations of the features for each class
    stds = np.array([np.sqrt(model.var_[i]) for i in range(len(model.classes_))])
    
    plt.figure(figsize=(10, 6))
    for i in range(len(model.classes_)):
        plt.bar(feature_names, stds[i], alpha=0.5, label=f'Class {i}')
    
    plt.title('Feature Standard Deviations by Class')
    plt.xlabel('Features')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_class_distribution(y_train, y_test, target_names):
    """
    Plot the distribution of classes in training and test sets.
    Args:
        y_train: Training labels
        y_test: Test labels
        target_names: Names of the target classes
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_train)
    plt.title('Training Set Class Distribution')
    plt.xticks(range(len(target_names)), target_names)
    
    plt.subplot(1, 2, 2)
    sns.countplot(x=y_test)
    plt.title('Test Set Class Distribution')
    plt.xticks(range(len(target_names)), target_names)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close() 