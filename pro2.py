import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras


diabetes_data = load_diabetes()
X_diabetes = diabetes_data.data
y_diabetes = diabetes_data.target
y_diabetes = (y_diabetes > y_diabetes.mean()).astype(int)  # Convert to binary classification


cancer_data = load_breast_cancer()
X_cancer = cancer_data.data
y_cancer = cancer_data.target


sonar_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data", header=None)
X_sonar = sonar_data.iloc[:, :-1].values
y_sonar = sonar_data.iloc[:, -1].map({'R': 0, 'M': 1}).values  # Convert to binary

# Function to create and train the model
def create_and_train_model(X, y, activation_function='relu'):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(64, activation=activation_function, input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation=activation_function),
        keras.layers.Dense(1, activation='sigmoid')  # Binary classification output
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Evaluate on Diabetes Dataset
diabetes_accuracy = create_and_train_model(X_diabetes, y_diabetes)
print(f"Diabetes Dataset Accuracy: {diabetes_accuracy:.2f}")

# Evaluate on Cancer Dataset
cancer_accuracy = create_and_train_model(X_cancer, y_cancer)
print(f"Cancer Dataset Accuracy: {cancer_accuracy:.2f}")

# Evaluate on Sonar Dataset
sonar_accuracy = create_and_train_model(X_sonar, y_sonar)
print(f"Sonar Dataset Accuracy: {sonar_accuracy:.2f}")
  