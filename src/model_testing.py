import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import warnings
from tabulate import tabulate

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def load_test_data(filepath):
    """
    Loads the test data from a .npz file.

    Args:
        filepath (str): Path to the .npz file.

    Returns:
        np.array, np.array: Test features and targets.
    """
    data = np.load(filepath)
    return data['X_test'], data['y_test']

def load_model(model_filepath):
    """
    Loads a trained LSTM model and the corresponding scaler.

    Args:
        model_filepath (str): Path to the model file.

    Returns:
        model: Loaded LSTM model.
        scaler: Loaded MinMaxScaler.
    """
    model = tf.keras.models.load_model(model_filepath)
    scaler = joblib.load(f'{model_filepath}_scaler.joblib')
    return model, scaler

def test_model(model, scaler, X_test, y_test):
    """
    Tests the LSTM model on the test data.

    Args:
        model: Trained LSTM model.
        scaler: Fitted MinMaxScaler.
        X_test (np.array): Test features.
        y_test (np.array): Test targets.

    Returns:
        float: RMSE on the test data.
    """
    y_pred = model.predict(X_test)
    
    # Inverse transform the predictions and test targets
    y_pred_rescaled = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((y_pred.shape[0], 2))), axis=1))[:, 0]
    y_test_rescaled = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 2))), axis=1))[:, 0]
    
    rmse = mean_squared_error(y_test_rescaled, y_pred_rescaled, squared=False)
    return rmse

if __name__ == "__main__":
    model_dir = 'models'  # Directory containing trained models
    test_data_dir = 'data/test'  # Directory containing test data
    
    results = []
    
    for filename in os.listdir(test_data_dir):
        if filename.endswith('.npz'):
            test_data_filepath = os.path.join(test_data_dir, filename)
            model_filename = f"{os.path.splitext(filename)[0].replace('_test_data', '')}_model.h5"
            model_filepath = os.path.join(model_dir, model_filename)
            
            if not os.path.exists(model_filepath):
                print(f"Model file {model_filepath} does not exist.")
                continue
            
            X_test, y_test = load_test_data(test_data_filepath)
            model, scaler = load_model(model_filepath)
            
            rmse = test_model(model, scaler, X_test, y_test)
            results.append((filename, rmse))
    
    # Print results in a table
    print(tabulate(results, headers=["Test Data File", "RMSE"], tablefmt="grid"))
