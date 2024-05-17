import pandas as pd
import os

def load_data(filepath):
    """
    Loads the stock data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def preprocess_data(df):
    """
    Preprocesses the stock data by adding moving averages and handling missing values.

    Args:
        df (pd.DataFrame): DataFrame containing the stock data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    # Adding moving averages as new features
    df['MA_5'] = df['close'].rolling(window=5).mean()
    df['MA_20'] = df['close'].rolling(window=20).mean()
    
    # Dropping rows with NaN values (resulting from rolling window operations)
    df.dropna(inplace=True)
    
    return df

def save_data(df, filepath):
    """
    Saves the preprocessed data to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the preprocessed data.
        filepath (str): Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath)
    print(f'Processed data saved to {filepath}')

def process_all_files(raw_data_dir, processed_data_dir):
    """
    Processes all CSV files in the raw data directory and saves the processed data to the processed data directory.

    Args:
        raw_data_dir (str): Directory containing raw data CSV files.
        processed_data_dir (str): Directory to save the processed data CSV files.
    """
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.csv'):
            raw_filepath = os.path.join(raw_data_dir, filename)
            processed_filepath = os.path.join(processed_data_dir, filename)
            
            df = load_data(raw_filepath)
            df = preprocess_data(df)
            save_data(df, processed_filepath)

if __name__ == "__main__":
    raw_data_dir = 'data/raw'  # Directory containing raw data CSV files
    processed_data_dir = 'data/processed'  # Directory to save processed data CSV files
    
    process_all_files(raw_data_dir, processed_data_dir)
