import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def fetch_sp500_tickers():
    """Fetch an up-to-date list of S&P 500 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df = html[0]
    tickers = df.Symbol.to_list()
    # Some tickers might not be directly usable (e.g., BRK.B needs to be BRK-B for yfinance)
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers


def download_stock_data(tickers, start_date, end_date, filename):
    """Download stock data using Yahoo Finance and save to a CSV file to avoid repeated downloads."""
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    data.to_csv(filename)


def load_and_preprocess_data(filename, start_date=None, end_date=None):
    """
    Load and preprocess the stock data for training the model.
    Preprocessing steps are: filling missing values, normalizing the data, and creating a scaler.
    """
    if not os.path.isfile(filename):
        print(f"{filename} not found, fetching S&P 500 tickers and downloading stock data...")
        tickers = fetch_sp500_tickers()
        download_stock_data(tickers, start_date, end_date, filename)

    df = pd.read_csv(filename, header=[0, 1], index_col=[0], parse_dates=[0])
    # Normalize features (example for close prices, adjust as needed)
    close_prices = df.xs('Close', axis=1, level=1, drop_level=False)
    close_prices = close_prices.dropna(axis=1, how='all')

    print("Tickers after preprocessing:", close_prices.columns.tolist())

    # Apply forward and backward fill directly to close_prices
    close_prices.ffill(inplace=True)  # Forward fill
    close_prices.bfill(inplace=True)  # Backward fill

    # Create a scaler and fit it to the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices.values)  # Now scaling the filled data

    return scaled_data, scaler, close_prices.columns  # Return columns for stock mapping


def create_sequences(data, seq_length):
    """Create sequences for training the model."""

    # Initialize empty lists to store the sequences
    xs, ys = [], []
    # Create the sequences
    for i in range(len(data) - seq_length - 1):
        # Instantiate a sequence of length seq_length
        x = data[i:(i + seq_length)]
        # Instantiate the target value
        y = data[i + seq_length]
        # Append the sequence and target to the respective lists
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
