# Stock Price Prediction Using LSTM

This project uses LSTM (Long Short-Term Memory) networks to predict stock prices. It includes scripts for data collection, preprocessing, model training, testing, and making predictions. The project is designed to work with multiple stock symbols and can be easily extended to include more stocks.

## Project Structure

├── data/
│ ├── raw/ # Raw data files
│ ├── processed/ # Processed data files
│ └── test/ # Test data files
├── models/ # Trained model files
├── src/
│ ├── data_collection.py # Script for data collection
│ ├── data_preprocessing.py # Script for data preprocessing
│ ├── model_training.py # Script for model training
│ ├── model_testing.py # Script for model testing
│ └── apple_example_prediction.py # Script to make predictions for Apple stock as an example
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## Project Details

### data_collection.py

Collects historical stock price data from the Alpha Vantage API.

### data_preprocessing.py

Preprocesses the collected data by adding features and handling missing values.

### model_training.py

Trains LSTM models on the preprocessed data and saves the models and scalers.

### model_testing.py

Tests the trained models on the test data and prints the RMSE for each model.

### apple_example_prediction.py

Provides an example of making predictions for Apple stock using a trained LSTM model.

## Data Collection:

Run the script:

```bash
python src/data_collection.py --api_key YOUR_ALPHA_VANTAGE_API_KEY --output_size full STOCK_LIST
```

and replace YOUR_ALPHA_VANTAGE_API_KEY and STOCK_LIST with the stocks you want in the following format - AAPL GOOGL MSFT

No additional command line arguments required for the remaining files. 
