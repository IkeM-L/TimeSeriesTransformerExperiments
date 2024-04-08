import os
import pandas as pd
import torch
from data_preprocessing import download_stock_data
from test_model import load_model, load_scaler, predict_price, prepare_data_for_prediction


# Data loading and preparation functions

def load_stock_data(tickers, start_date, end_date, filename):
    """
    Ensure stock data is downloaded and loaded from CSV.
    :param tickers: A list of stock tickers to download data for
    :param start_date: The start date for the data
    :param end_date: The end date for the data
    :param filename: The filename to save the data to
    :return: A DataFrame containing the stock data
    """
    if not os.path.isfile(filename):
        download_stock_data(tickers, start_date, end_date, filename)
    return pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)


def get_tickers_from_last_day(filename):
    """
    Extract tickers from the last day of available training data.
    :param filename: The filename containing the last day of training data
    :return: A list of tickers from the last day of training data
    """
    df_last_day = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)
    return df_last_day.xs('Close', level=1, axis=1).dropna(axis=1).columns.get_level_values(0).tolist()


def get_actual_prices(stock_data, index):
    """
    Get actual open and close prices for trading day.
    :param stock_data: The stock data DataFrame
    :param index: The index of the trading day
    :return: The actual open and close prices for the trading day
    """
    open_prices = stock_data.xs('Open', level=1, axis=1).iloc[index].values
    close_prices = stock_data.xs('Close', level=1, axis=1).iloc[index].values
    return zip(open_prices, close_prices)

# Prediction functions


def prepare_and_predict(stock_data, i, seq_length, model, scaler, device, tickers):
    """
    Prepare data for prediction and predict next day's prices.
    :param stock_data: The stock data DataFrame
    :param i: The index of the trading day
    :param seq_length: The sequence length used for prediction
    :param model: The PyTorch model for prediction
    :param scaler: The MinMaxScaler object used for scaling data
    :param device: The device to run the model on
    :param tickers: The list of stock tickers
    :return: The recent data, actual prices, and predicted prices
    """

    # Extract the Close prices of recent data for prediction, drop any columns with NaN values
    recent_data = stock_data.iloc[i:i + seq_length].xs('Close', level=1, axis=1).dropna(axis=1).values
    # Convert the recent data to a PyTorch tensor for prediction
    recent_sequence_tensor = prepare_data_for_prediction(recent_data, scaler, device)
    # Predict the next day's prices using the model
    predicted_prices = predict_price(model, recent_sequence_tensor, scaler, device)
    # Get the actual open and close prices for the trading day to compare with predictions
    actual_prices = get_actual_prices(stock_data, i + seq_length)
    return recent_data, actual_prices, predicted_prices

# Trading functions


def trade(predicted_prices, actual_prices, capital):
    """
    Calculate profit/loss from trading based on predictions.

    :param predicted_prices: The predicted prices for the trading day
    :param actual_prices: The actual open and close prices for the trading day
    :param capital: The current capital available for trading
    :return: The updated capital, number of profitable trades, and total trades made
    """
    profitable_trades, total_trades = 0, 0
    for predicted, (actual_open, actual_close) in zip(predicted_prices, actual_prices):
        if predicted > actual_open:
            profit = actual_close - actual_open
            capital += profit
            total_trades += 1
            if profit > 0:
                profitable_trades += 1
    return capital, profitable_trades, total_trades


def execute_trades(stock_data, model, scaler, device, seq_length, tickers, initial_capital, last_date_in_training_data):
    """
    Execute trades based on model predictions and update capital.

    We use a very simple strategy here: buy if the predicted price is higher than the actual open price, and sell at the
    actual close price. We assume that we can buy/sell at the open/close prices and ignore other factors like slippage,
    transaction costs, etc.
    We also do not try to short stocks
    We spend an equal amount on each stock, so the capital is divided equally among the stocks that are bought.
    :param stock_data: The stock data DataFrame
    :param model: The PyTorch model for prediction
    :param scaler: The MinMaxScaler object used for scaling data
    :param device: The device to run the model on
    :param seq_length: The sequence length used for prediction
    :param tickers: The list of stock tickers we are trading
    :param initial_capital: The initial capital available for trading
    :param last_date_in_training_data: The last date in the training data
    :return: The final capital after trading
    """

    capital = initial_capital
    profitable_trades, total_trades = 0, 0

    # Find the index for the day immediately after the last date in training data
    start_index = stock_data.index.get_loc(last_date_in_training_data) + 1 - seq_length

    for i in range(start_index, len(stock_data) - seq_length):
        # Prepare and predict the next day's prices
        recent_data, actual_prices, predicted_prices = prepare_and_predict(stock_data, i, seq_length, model, scaler,
                                                                           device, tickers)
        # Execute trades based on predictions and update capital
        capital, profitable, total = trade(predicted_prices, actual_prices, capital)
        # Update total profitable trades and total trades made
        profitable_trades += profitable
        total_trades += total
        # Print results for the day
        print_day_results(stock_data.index[i + seq_length], capital, profitable, total)
    return capital


def print_day_results(date, capital, profitable_trades, total_trades):
    """Print results of trading for a single day."""
    print(f"Day {date.strftime('%Y-%m-%d')}: ${capital:.2f} | "
          f"{profitable_trades} profitable trades of {total_trades} total trades "
          f"or {profitable_trades / total_trades * 100:.2f}% profitable if any trades made.")


# Statistics functions


def print_final_statistics(initial_capital, capital, stock_data):
    """Print final statistics after trading."""
    print(f"Starting capital: ${initial_capital}")
    print(f"Ending capital: ${capital}")
    roi = (capital - initial_capital) / initial_capital
    print(f"Return on Investment: {roi * 100:.2f}%")
    # Additional statistics like annualized return can be calculated here.


def print_buy_and_hold_benchmark(stock_data, initial_capital, last_date_in_training_data):
    """Prints the buy and hold benchmark statistics."""
    # Determine the start index for trading, which is the day after last_date_in_training_data
    start_index = stock_data.index.get_loc(last_date_in_training_data) + 1
    start_date = stock_data.index[start_index]
    end_date = stock_data.index[-1]

    # Calculate initial and final portfolio values
    open_prices_start = stock_data.xs('Open', level=1, axis=1).iloc[start_index]
    close_prices_end = stock_data.xs('Close', level=1, axis=1).iloc[-1]

    shares = initial_capital / len(open_prices_start) / open_prices_start
    final_value = (shares * close_prices_end).sum()

    # Print the results
    print(f"Buy and Hold Benchmark from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}:")
    print(f"Initial investment: ${initial_capital:.2f}")
    print(f"Final portfolio value: ${final_value:.2f}")
    buy_and_hold_return = (final_value - initial_capital) / initial_capital
    print(f"Buy and hold return: {buy_and_hold_return * 100:.2f}%")


# Main function

def main():
    """
    The main function to execute the virtual trading simulation.
    :return:
    """
    training_data_filename = "sp500_stock_data_last_day.csv"
    testing_filename = "sp500_stock_data_jan_and_feb.csv"
    start_date, end_date, last_date_in_training_data = "2023-01-01", "2023-03-01", "2023-01-30"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tickers = get_tickers_from_last_day(training_data_filename)
    stock_data = load_stock_data(tickers, start_date, end_date, testing_filename)
    model = load_model("stock_prediction_model.pth", device)
    scaler = load_scaler("scaler.pkl")
    seq_length = 10  # Ensure this is the correct sequence length as per your model's training
    initial_capital = 10000  # Define your initial capital here if it's not done elsewhere

    # Execute trades and print final statistics
    capital = execute_trades(stock_data, model, scaler, device, seq_length, tickers, initial_capital, last_date_in_training_data)
    print_final_statistics(initial_capital, capital, stock_data)
    print_buy_and_hold_benchmark(stock_data, initial_capital, last_date_in_training_data)


if __name__ == "__main__":
    main()
