<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Simple Iris Classification with Naive Bayes</div>

# 1. Project Overview
This is a simple implementation of a Naive Bayes classifier for the famous Iris flower dataset. The code demonstrates how to:

1. Load the Iris dataset
2. Split data into training and testing sets
3. Train a Gaussian Naive Bayes model
4. Make predictions
5. Evaluate model performance
6. Visualize results with informative plots

# 1.1 Dataset Description
The Iris dataset contains measurements of 150 iris flowers from three different species:
- Iris setosa
- Iris versicolor
- Iris virginica

Each sample has four features:
1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

# 1.2 How to Run

1. Make sure you have Python installed
2. Install the required packages:
```bash
pip install scikit-learn matplotlib seaborn numpy
```
3. Run the classifier:
```bash
python simple_iris_classifier.py
```

# 1.3 Expected Output
The program will:
1. Train the model on the Iris dataset
2. Display the model's accuracy and a detailed classification report
3. Generate two visualizations:
   - Confusion matrix showing prediction results
   - Feature importance plots for each flower species

The visualizations will be both displayed and saved as:
- `confusion_matrix.png`
- `feature_importance.png`

# 1.4 Dependencies
- scikit-learn
- matplotlib
- seaborn
- numpy

# 2. Understanding the Code

The `simple_iris_classifier.py` file follows these steps:

1. **Data Loading**: Loads the Iris dataset from scikit-learn
2. **Data Splitting**: Divides the data into training (80%) and testing (20%) sets
3. **Model Training**: Creates and trains a Gaussian Naive Bayes classifier
4. **Prediction**: Uses the trained model to predict flower species
5. **Evaluation**: Calculates accuracy and generates a classification report
6. **Visualization**:
   - Creates a confusion matrix to visualize classification results
   - Generates feature importance plots to show which measurements are most significant for identifying each species

# 3. Visualizations

## 3.1 Confusion Matrix
The confusion matrix shows how well the model classified each species. The rows represent the true species, and the columns represent the predicted species. The diagonal elements show correct classifications, while off-diagonal elements show misclassifications.

## 3.2 Feature Importance
The feature importance plots show which of the four measurements (sepal length, sepal width, petal length, petal width) are most important for identifying each flower species. Higher values indicate greater importance of that feature for classifying the specific species.

# 4. Next Steps

Potential improvements and extensions:
- Try different classification algorithms (Random Forest, SVM, etc.)
- Implement cross-validation for more robust evaluation
- Add more advanced visualizations like decision boundaries
- Create an interactive version that can classify new flower measurements 