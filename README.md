# Credit Card Fraud Detection using Random Forest

## Tech Stack
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
- **Model:** Random Forest Classifier
- **Tools:** Jupyter Notebook, IDE/Code Editor

## Project Overview
This project is focused on developing a machine learning model to detect fraudulent credit card transactions. The model is trained using a dataset that contains a large number of transactions, both legitimate and fraudulent. The Random Forest algorithm was used for classification due to its robustness and effectiveness in handling imbalanced datasets.

## Dataset
- **Source:** The dataset used is `creditcard.csv`.
- **Size:** 284,807 transactions.
- **Features:** 30 features including time and amount.
- **Target:** The target variable `Class` represents whether a transaction is fraudulent (1) or not (0).

## Key Steps in the Project
1. **Data Loading and Preprocessing:**
   - Loaded the dataset using Pandas.
   - Checked for missing values and verified data integrity.
   - Split the dataset into features (`X`) and target (`y`).

2. **Data Splitting:**
   - The dataset was split into training (80%) and testing (20%) sets using `train_test_split` with stratification on the `Class` label.

3. **Model Training:**
   - Trained a Random Forest Classifier with 100 estimators.
   - The model was trained on the training dataset.

4. **Model Evaluation:**
   - Made predictions on the test dataset.
   - Evaluated the model's performance using accuracy, confusion matrix, and classification report.
   - Achieved an accuracy of 99.96% on the test set.

5. **Saving the Model:**
   - The trained model was saved using Joblib for future use or deployment.

## Results
- **Accuracy:** 99.96%
- **Confusion Matrix:**

- **Classification Report:**
-          precision    recall  f1-score   support

         0       1.00      1.00      1.00     56864
         1       0.94      0.82      0.87        98

  accuracy                           1.00     56962
 macro avg       0.97      0.91      0.94     56962

weighted avg 1.00 1.00 1.00 56962

## How to Run the Project

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/credit-card-fraud-detection.git

2. Navigate to the project directory:
   ```sh
   cd credit-card-fraud-detection
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
4. Run the Jupyter Notebook or Python script:
  a. If using Jupyter Notebook:
      ```sh
      jupyter notebook CreditCardFraudDetection.ipynb
  a. If using Jupyter Notebook:
      
      ```sh
     python credit_card_fraud_detection.py
## Saved Model

The trained Random Forest model is saved as `credit_card_fraud_model.pkl`. You can load and use this model for predictions on new data.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.


  


