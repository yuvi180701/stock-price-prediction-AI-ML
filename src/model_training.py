import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_data(filepath):
    """
    Loads the preprocessed stock data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def preprocess_data(df, n_steps=50):
    """
    Preprocesses the stock data for LSTM model.

    Args:
        df (pd.DataFrame): DataFrame containing the preprocessed stock data.
        n_steps (int): Number of time steps for LSTM.

    Returns:
        np.array, np.array, MinMaxScaler: Scaled features and target arrays, and the scaler.
    """
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['close', 'MA_5', 'MA_20']])
    
    X = []
    y = []

    for i in range(n_steps, len(df_scaled)):
        X.append(df_scaled[i-n_steps:i])
        y.append(df_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def build_lstm_model(input_shape):
    """
    Builds and compiles the LSTM model.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train, X_val, y_val):
    """
    Trains the LSTM model on the stock data.

    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training targets.
        X_val (np.array): Validation features.
        y_val (np.array): Validation targets.

    Returns:
        model: Trained LSTM model.
    """
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])
    
    return model

def save_model(model, scaler, model_filepath):
    """
    Saves the trained model and the scaler to a file.

    Args:
        model: Trained model.
        scaler: Fitted MinMaxScaler.
        model_filepath (str): Path to save the model file.
    """
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    model.save(model_filepath)
    joblib.dump(scaler, f'{model_filepath}_scaler.joblib')
    print(f'Model and scaler saved to {model_filepath} and {model_filepath}_scaler.joblib')

def save_test_data(X_test, y_test, test_data_filepath):
    """
    Saves the test data to a file.

    Args:
        X_test (np.array): Test features.
        y_test (np.array): Test targets.
        test_data_filepath (str): Path to save the test data file.
    """
    os.makedirs(os.path.dirname(test_data_filepath), exist_ok=True)
    np.savez(test_data_filepath, X_test=X_test, y_test=y_test)
    print(f'Test data saved to {test_data_filepath}')

def main(processed_data_dir, model_dir, test_data_dir):
    for filename in os.listdir(processed_data_dir):
        if filename.endswith('.csv'):
            processed_filepath = os.path.join(processed_data_dir, filename)
            model_filepath = os.path.join(model_dir, f'{os.path.splitext(filename)[0]}_model.h5')
            test_data_filepath = os.path.join(test_data_dir, f'{os.path.splitext(filename)[0]}_test_data.npz')
            
            df = load_data(processed_filepath)
            X, y, scaler = preprocess_data(df)
            
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Further split the training set into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            model = train_model(X_train, y_train, X_val, y_val)
            save_model(model, scaler, model_filepath)
            save_test_data(X_test, y_test, test_data_filepath)

if __name__ == "__main__":
    processed_data_dir = 'data/processed'  # Directory containing processed data CSV files
    model_dir = 'models'  # Directory to save trained models
    test_data_dir = 'data/test'  # Directory to save test data
    
    main(processed_data_dir, model_dir, test_data_dir)
