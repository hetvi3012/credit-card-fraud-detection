# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load the dataset
# Ensure the dataset 'creditcard.csv' is in your working directory
data = pd.read_csv('creditcard.csv')

# Check for missing values
print("Checking for missing values...")
print(data.isnull().sum())

# Display basic information about the dataset
print("\nDataset Info:")
print(data.info())

# Check the distribution of classes (fraud vs. non-fraud)
print("\nClass distribution:")
print(data['Class'].value_counts())

# Split the data into features (X) and target (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the Random Forest model
print("\nTraining the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Save the model for future use
joblib.dump(model, 'credit_card_fraud_model.pkl')
print("\nModel saved as 'credit_card_fraud_model.pkl'.")
