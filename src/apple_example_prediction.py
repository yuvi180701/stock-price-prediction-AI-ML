import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

def load_model_and_scaler(symbol):
    """
    Loads a trained LSTM model and the corresponding scaler.

    Args:
        symbol (str): Stock symbol to load model and scaler for.

    Returns:
        model: Loaded LSTM model.
        scaler: Loaded MinMaxScaler.
    """
    model_filepath = f'models/{symbol}_daily_model.h5'
    scaler_filepath = f'models/{symbol}_daily_model.h5_scaler.joblib'
    
    model = tf.keras.models.load_model(model_filepath)
    scaler = joblib.load(scaler_filepath)
    
    return model, scaler

def preprocess_features(features, scaler, n_steps=50):
    """
    Preprocesses the input features for LSTM model.

    Args:
        features (list): List of feature dictionaries.
        scaler (MinMaxScaler): Fitted MinMaxScaler.
        n_steps (int): Number of time steps for LSTM.

    Returns:
        np.array: Preprocessed features ready for LSTM model.
    """
    features_df = pd.DataFrame(features)
    scaled_features = scaler.transform(features_df)
    
    X_input = []
    for i in range(n_steps, len(scaled_features) + 1):
        X_input.append(scaled_features[i-n_steps:i])
    
    X_input = np.array(X_input)
    
    return X_input

def make_predictions(symbol, features):
    """
    Makes predictions using the trained LSTM model for a given symbol.

    Args:
        symbol (str): Stock symbol.
        features (list): List of feature dictionaries.

    Returns:
        list: Predicted stock prices.
    """
    model, scaler = load_model_and_scaler(symbol)
    X_input = preprocess_features(features, scaler)
    
    predictions = model.predict(X_input)
    
    predictions_rescaled = scaler.inverse_transform(
        np.concatenate((predictions, np.zeros((predictions.shape[0], 2))), axis=1)
    )[:, 0]
    
    return predictions_rescaled.tolist()

if __name__ == "__main__":
    symbol = "AAPL"
    features = [
        {"close": 150.0, "MA_5": 148.0, "MA_20": 145.0},
        {"close": 151.0, "MA_5": 149.0, "MA_20": 146.0},
        {"close": 152.0, "MA_5": 150.0, "MA_20": 147.0},
        {"close": 153.0, "MA_5": 151.0, "MA_20": 148.0},
        {"close": 154.0, "MA_5": 152.0, "MA_20": 149.0},
        {"close": 155.0, "MA_5": 153.0, "MA_20": 150.0},
        {"close": 156.0, "MA_5": 154.0, "MA_20": 151.0},
        {"close": 157.0, "MA_5": 155.0, "MA_20": 152.0},
        {"close": 158.0, "MA_5": 156.0, "MA_20": 153.0},
        {"close": 159.0, "MA_5": 157.0, "MA_20": 154.0},
        {"close": 160.0, "MA_5": 158.0, "MA_20": 155.0},
        {"close": 161.0, "MA_5": 159.0, "MA_20": 156.0},
        {"close": 162.0, "MA_5": 160.0, "MA_20": 157.0},
        {"close": 163.0, "MA_5": 161.0, "MA_20": 158.0},
        {"close": 164.0, "MA_5": 162.0, "MA_20": 159.0},
        {"close": 165.0, "MA_5": 163.0, "MA_20": 160.0},
        {"close": 166.0, "MA_5": 164.0, "MA_20": 161.0},
        {"close": 167.0, "MA_5": 165.0, "MA_20": 162.0},
        {"close": 168.0, "MA_5": 166.0, "MA_20": 163.0},
        {"close": 169.0, "MA_5": 167.0, "MA_20": 164.0},
        {"close": 170.0, "MA_5": 168.0, "MA_20": 165.0},
        {"close": 171.0, "MA_5": 169.0, "MA_20": 166.0},
        {"close": 172.0, "MA_5": 170.0, "MA_20": 167.0},
        {"close": 173.0, "MA_5": 171.0, "MA_20": 168.0},
        {"close": 174.0, "MA_5": 172.0, "MA_20": 169.0},
        {"close": 175.0, "MA_5": 173.0, "MA_20": 170.0},
        {"close": 176.0, "MA_5": 174.0, "MA_20": 171.0},
        {"close": 177.0, "MA_5": 175.0, "MA_20": 172.0},
        {"close": 178.0, "MA_5": 176.0, "MA_20": 173.0},
        {"close": 179.0, "MA_5": 177.0, "MA_20": 174.0},
        {"close": 180.0, "MA_5": 178.0, "MA_20": 175.0},
        {"close": 181.0, "MA_5": 179.0, "MA_20": 176.0},
        {"close": 182.0, "MA_5": 180.0, "MA_20": 177.0},
        {"close": 183.0, "MA_5": 181.0, "MA_20": 178.0},
        {"close": 184.0, "MA_5": 182.0, "MA_20": 179.0},
        {"close": 185.0, "MA_5": 183.0, "MA_20": 180.0},
        {"close": 186.0, "MA_5": 184.0, "MA_20": 181.0},
        {"close": 187.0, "MA_5": 185.0, "MA_20": 182.0},
        {"close": 188.0, "MA_5": 186.0, "MA_20": 183.0},
        {"close": 189.0, "MA_5": 187.0, "MA_20": 184.0},
        {"close": 190.0, "MA_5": 188.0, "MA_20": 185.0},
        {"close": 191.0, "MA_5": 189.0, "MA_20": 186.0},
        {"close": 192.0, "MA_5": 190.0, "MA_20": 187.0},
        {"close": 193.0, "MA_5": 191.0, "MA_20": 188.0},
        {"close": 194.0, "MA_5": 192.0, "MA_20": 189.0},
        {"close": 195.0, "MA_5": 193.0, "MA_20": 190.0},
        {"close": 196.0, "MA_5": 194.0, "MA_20": 191.0},
        {"close": 197.0, "MA_5": 195.0, "MA_20": 192.0},
        {"close": 198.0, "MA_5": 196.0, "MA_20": 193.0},
        {"close": 199.0, "MA_5": 197.0, "MA_20": 194.0},
        {"close": 200.0, "MA_5": 198.0, "MA_20": 195.0}
    ]
    
    predictions = make_predictions(symbol, features)
    print(predictions)
