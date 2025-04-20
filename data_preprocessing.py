import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    """
    Load the Iris dataset and perform preprocessing steps.
    Returns:
        X_train, X_test, y_train, y_test: Split and preprocessed data
    """
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Convert to DataFrame for better visualization
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y
    
    # Check for missing values
    print("Missing values in dataset:")
    print(df.isnull().sum())
    
    # Check for outliers using IQR method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    print("\nNumber of outliers per feature:")
    print(outliers)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, iris.feature_names, iris.target_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_preprocess_data()
    print("\nData shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}") 