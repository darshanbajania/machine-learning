import pandas as pd
import joblib

def predict_fraud(transaction_data):
    """
    Loads a pre-trained machine learning model and uses it to predict
    if a new transaction is fraudulent.

    Args:
        transaction_data (dict): A dictionary representing a single transaction
                                 with the same features as the training data.

    Returns:
        int: The predicted class (0 for non-fraudulent, 1 for fraudulent).
    """
    # Load the pre-trained model from the joblib file
    try:
        loaded_model = joblib.load('gradient_boosting_model.joblib')
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: 'gradient_boosting_model.joblib' not found. Please run load_data.py first to train and save the model.")
        return None

    # Create a pandas DataFrame from the transaction data
    # Ensure the columns are in the same order as the training data
    features = [
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'
    ]
    transaction_df = pd.DataFrame([transaction_data], columns=features)

    # Use the loaded model to make a prediction
    prediction = loaded_model.predict(transaction_df)

    return prediction[0]

# --- Example Usage ---
# This is an example of a new transaction. Note that we are using
# dummy values here, but in a real-world scenario, this would be
# a new transaction from your system.
new_transaction = {
    'V1': -0.42, 'V2': 1.0, 'V3': -1.2, 'V4': 0.8, 'V5': -0.7,
    'V6': -0.2, 'V7': -0.3, 'V8': 0.1, 'V9': -0.5, 'V10': -0.4,
    'V11': 0.6, 'V12': -0.6, 'V13': 0.2, 'V14': -0.7, 'V15': 0.5,
    'V16': -0.8, 'V17': -0.5, 'V18': -0.4, 'V19': 0.3, 'V20': 0.1,
    'V21': -0.2, 'V22': 0.1, 'V23': 0.3, 'V24': -0.2, 'V25': 0.2,
    'V26': 0.1, 'V27': -0.1, 'V28': 0.0
}

# Make a prediction on the new transaction
prediction = predict_fraud(new_transaction)

if prediction is not None:
    if prediction == 1:
        print("\nPrediction: This transaction is likely FRAUDULENT.")
    else:
        print("\nPrediction: This transaction is likely NON-FRAUDULENT.")
