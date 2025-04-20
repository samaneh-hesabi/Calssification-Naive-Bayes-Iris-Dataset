from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import numpy as np

def train_naive_bayes(X_train, y_train):
    """
    Train a Naive Bayes classifier and perform cross-validation.
    Args:
        X_train: Training features
        y_train: Training labels
    Returns:
        model: Trained Naive Bayes classifier
        cv_scores: Cross-validation scores
    """
    # Initialize the Naive Bayes classifier
    model = GaussianNB()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    print("\nCross-validation scores:")
    print(f"Mean CV score: {np.mean(cv_scores):.4f}")
    print(f"Standard deviation: {np.std(cv_scores):.4f}")
    
    return model, cv_scores

def predict(model, X_test):
    """
    Make predictions using the trained model.
    Args:
        model: Trained Naive Bayes classifier
        X_test: Test features
    Returns:
        y_pred: Predicted labels
        y_prob: Predicted probabilities
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    return y_pred, y_prob 