"""
Simple Iris Flower Classification with Naive Bayes
-------------------------------------------------
This program:
1. Loads the famous Iris flower dataset
2. Trains a simple Naive Bayes model to classify flowers
3. Shows how well the model performs
4. Visualizes the results
"""

# Import the tools we need
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Load the Iris dataset (a famous dataset of flower measurements)
print("Loading the Iris flower dataset...")
iris = datasets.load_iris()
X = iris.data        # X = flower measurements (features)
y = iris.target      # y = flower species (what we want to predict)
flower_names = iris.target_names  # The three types of flowers

# Print information about our dataset
print(f"Dataset has {len(X)} flowers with {X.shape[1]} measurements each")
print(f"The three flower species are: {', '.join(flower_names)}")

# Step 2: Split data into training set (to learn from) and testing set (to evaluate)
print("\nSplitting data into training (80%) and testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,     # Use 20% for testing
    random_state=42    # For reproducible results
)
print(f"Training set: {len(X_train)} flowers")
print(f"Testing set: {len(X_test)} flowers")

# Step 3: Create and train our model
print("\nTraining the Naive Bayes model...")
model = GaussianNB()  # Create a Naive Bayes model
model.fit(X_train, y_train)       # Train the model with our data
print("Model training complete!")

# Step 4: Make predictions on the test data
print("\nMaking predictions on test data...")
predictions = model.predict(X_test)

# Step 5: Check how well our model performed
print("\nEvaluating model performance...")
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print("This means the model correctly identified " + 
      f"{accuracy*100:.1f}% of the flowers in the test set.")

# Step 6: Show a detailed report
print("\nDetailed Classification Report:")
report = classification_report(y_test, predictions, 
                              target_names=flower_names)
print(report)

# Step 7: Visualize the results with a confusion matrix
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=flower_names, 
            yticklabels=flower_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Step 8: Visualize the feature importance
feature_names = iris.feature_names
feature_importance = np.abs(model.theta_)

plt.figure(figsize=(12, 6))
for i, class_name in enumerate(flower_names):
    plt.subplot(1, 3, i+1)
    sns.barplot(x=feature_names, y=feature_importance[i])
    plt.title(f'Feature Importance: {class_name}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nDone! You've successfully trained, evaluated, and visualized a Naive Bayes model!")

