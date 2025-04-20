from data_preprocessing import load_and_preprocess_data
from model_training import train_naive_bayes, predict
from evaluation import evaluate_model
from visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_class_distribution
)

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_preprocess_data()
    
    # Train the model
    print("\nTraining Naive Bayes model...")
    model, cv_scores = train_naive_bayes(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred, y_prob = predict(model, X_test)
    
    # Evaluate the model
    print("\nEvaluating model...")
    metrics = evaluate_model(y_test, y_pred, y_prob, target_names)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_confusion_matrix(y_test, y_pred, target_names)
    plot_feature_importance(model, feature_names)
    plot_class_distribution(y_train, y_test, target_names)
    
    print("\nAll visualizations have been saved as PNG files.")
    print("Process completed successfully!")

if __name__ == "__main__":
    main() 