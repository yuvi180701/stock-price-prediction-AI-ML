import requests
import pandas as pd
import os
import argparse

def fetch_stock_data(symbol, api_key, output_size='full'):
    """
    Fetches historical stock data from Alpha Vantage API.

    Args:
        symbol (str): Stock symbol to fetch data for.
        api_key (str): API key for Alpha Vantage.
        output_size (str): Output size of data. 'full' for full-length data, 'compact' for last 100 data points.

    Returns:
        pd.DataFrame: DataFrame containing the stock data.
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={output_size}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    if 'Time Series (Daily)' not in data:
        if 'Note' in data:
            raise ValueError(f"API call limit reached: {data['Note']}")
        elif 'Error Message' in data:
            raise ValueError(f"Error message from API: {data['Error Message']}")
        else:
            raise ValueError(f"Unknown error: {data}")
    
    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index', dtype=float)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    return df

def save_data(symbol, df, raw_data_path):
    """
    Saves the DataFrame to a CSV file.

    Args:
        symbol (str): Stock symbol to be used in the filename.
        df (pd.DataFrame): DataFrame containing the stock data.
        raw_data_path (str): Path to save the raw data CSV file.
    """
    file_path = os.path.join(raw_data_path, f'{symbol}_daily.csv')
    os.makedirs(raw_data_path, exist_ok=True)
    df.to_csv(file_path)
    print(f'Data for {symbol} saved to {file_path}')

def main(symbols, api_key, output_size):
    raw_data_path = 'data/raw'
    for symbol in symbols:
        try:
            data = fetch_stock_data(symbol, api_key, output_size)
            save_data(symbol, data, raw_data_path)
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch historical stock data from Alpha Vantage.')
    parser.add_argument('--api_key', required=True, help='Alpha Vantage API key.')
    parser.add_argument('--output_size', default='full', choices=['full', 'compact'], help='Output size of data.')
    parser.add_argument('symbols', nargs='+', help='List of stock symbols to fetch data for.')
    
    args = parser.parse_args()
    
    main(args.symbols, args.api_key, args.output_size)
