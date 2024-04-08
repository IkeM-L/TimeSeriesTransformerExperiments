import math
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import mean_squared_error
from transformers import BertConfig

from model import StockPredictionModel  # Adjust the path as necessary


def load_model(model_path, device):
    """Load the model from the given path and move it to the given device."""
    config = BertConfig()
    model = StockPredictionModel(config, num_stocks=499, num_features=499).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_scaler(scaler_path):
    """Load the MinMaxScaler from the given path"""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def read_data(filename, seq_length):
    """Read the most recent data and prepare for prediction"""
    df = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)
    # Select only 'Close' prices and drop columns with any NaN values
    close_prices = df.xs('Close', level='Price', axis=1).fillna(method='ffill').fillna(method='bfill').dropna(axis=1)
    recent_data = close_prices.iloc[-seq_length:].values
    return recent_data, close_prices


def prepare_data_for_prediction(recent_data, scaler, device):
    """Prepare the recent data for prediction by scaling and converting to a PyTorch tensor."""
    data_scaled = scaler.transform(recent_data)
    sequence = np.expand_dims(data_scaled, axis=0)
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).to(device)
    return sequence_tensor


def predict_price(model, data_tensor, scaler, device):
    """Predict the next day's prices using the model and return the predicted prices."""
    with torch.no_grad():
        attention_mask = torch.ones(data_tensor.shape[:2], dtype=torch.long).to(device)
        prediction = model(data_tensor, attention_mask=attention_mask)
    predicted_prices = scaler.inverse_transform(prediction.cpu().numpy()).flatten()
    return predicted_prices


def prepare_comparison_data(filename, predicted_prices):
    """Prepare the actual prices for comparison with the predicted prices and return the tickers in the same order as
    the prices"""
    df_last_day = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)
    # Adjust to only include 'Close' prices and drop NaN columns
    actual_prices = df_last_day.xs('Close', level='Price', axis=1).fillna(method='ffill').fillna(method='bfill').dropna(axis=1).iloc[-1].values
    tickers = df_last_day.xs('Close', level='Price', axis=1).fillna(method='ffill').fillna(method='bfill').dropna(axis=1).columns.get_level_values(0).tolist()
    return tickers, actual_prices, predicted_prices


def calculate_rmse_and_print_table(tickers, actual_prices, predicted_prices, previous_day_prices):
    """Calculate RMSE and print a comparison table for each stock"""
    min_length = min(len(actual_prices), len(predicted_prices), len(previous_day_prices))
    actual_prices, predicted_prices, previous_day_prices = actual_prices[:min_length], predicted_prices[:min_length], previous_day_prices[:min_length]
    valid_indices = ~np.isnan(actual_prices) & ~np.isnan(predicted_prices) & ~np.isnan(previous_day_prices)
    rmse = math.sqrt(mean_squared_error(actual_prices[valid_indices], predicted_prices[valid_indices]))

    print("Ticker | Previous Price | Predicted Price | Actual Price | Difference")
    for i in np.where(valid_indices)[0]:
        print(f"{tickers[i]} | {previous_day_prices[i]:.2f} | {predicted_prices[i]:.2f} | {actual_prices[i]:.2f} | {predicted_prices[i] - actual_prices[i]:.2f}")
    print(f"RMSE: {rmse:.4f}")


def plot_differences(tickers, actual_prices, predicted_prices, previous_day_prices):
    """Plot the differences between predicted and actual prices for each stock"""
    # Ensure the same length
    min_length = min(len(actual_prices), len(predicted_prices))
    actual_prices = actual_prices[:min_length]
    predicted_prices = predicted_prices[:min_length]
    tickers = tickers[:min_length]

    differences = predicted_prices - actual_prices

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, differences, color='skyblue')
    plt.xlabel('Tickers')
    plt.ylabel('Difference (Predicted - Actual)')
    plt.title('Differences between Predicted and Actual Prices')
    plt.xticks(rotation=90)  # Rotate tick labels for better visibility
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()


def calculate_statistics(actual_prices, predicted_prices, random_walks_no_drift, random_walks_with_drift):
    """Calculate and print statistics for comparison with random walks and benchmarks"""
    # Basic Statistics
    mae_model = np.mean(np.abs(predicted_prices - actual_prices))
    mape_model = mape(actual_prices, predicted_prices)
    mse_model = np.mean((predicted_prices - actual_prices) ** 2)

    print(f"MAE - Model: {mae_model}")
    print(f"MAPE - Model: {mape_model}")
    print(f"MSE - Model: {mse_model}")

    # Calculate statistics for benchmarks
    mae_random_no_drift = np.mean(np.abs(random_walks_no_drift - actual_prices))
    mape_random_no_drift = mape(actual_prices, random_walks_no_drift)
    mse_random_no_drift = np.mean((random_walks_no_drift - actual_prices) ** 2)

    mae_random_with_drift = np.mean(np.abs(random_walks_with_drift - actual_prices))
    mape_random_with_drift = mape(actual_prices, random_walks_with_drift)
    mse_random_with_drift = np.mean((random_walks_with_drift - actual_prices) ** 2)

    # Print benchmark statistics
    print(f"MAE - Random Walk No Drift: {mae_random_no_drift}")
    print(f"MAPE - Random Walk No Drift: {mape_random_no_drift}")
    print(f"MSE - Random Walk No Drift: {mse_random_no_drift}")

    print(f"MAE - Random Walk With Drift: {mae_random_with_drift}")
    print(f"MAPE - Random Walk With Drift: {mape_random_with_drift}")
    print(f"MSE - Random Walk With Drift: {mse_random_with_drift}")

    # T-Tests
    t_stat_no_drift, p_value_no_drift = stats.ttest_rel(predicted_prices - actual_prices,
                                                        random_walks_no_drift - actual_prices)
    t_stat_with_drift, p_value_with_drift = stats.ttest_rel(predicted_prices - actual_prices,
                                                            random_walks_with_drift - actual_prices)

    print(f"T-Stat - Comparison with No Drift: {t_stat_no_drift}, P-Value: {p_value_no_drift}")
    print(f"T-Stat - Comparison with Drift: {t_stat_with_drift}, P-Value: {p_value_with_drift}")


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (MAPE)"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Main function to orchestrate the workflow
def main():
    # Set up parameters, load model, data, and make predictions
    # Parameters
    filename = "sp500_stock_data.csv"
    last_day_filename = "sp500_stock_data_last_day.csv"
    seq_length = 10
    model_save_path = "stock_prediction_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and scaler
    model = load_model(model_save_path, device)
    scaler = load_scaler("scaler.pkl")

    # Load the most recent data and prepare for prediction
    recent_data, close_prices = read_data(filename, seq_length)
    recent_sequence_tensor = prepare_data_for_prediction(recent_data, scaler, device)

    # Predict the next day's prices
    predicted_prices = predict_price(model, recent_sequence_tensor, scaler, device)

    # Prepare for comparison
    tickers, actual_prices, _ = prepare_comparison_data(last_day_filename, predicted_prices)
    previous_day_prices = close_prices.iloc[-2].values  # For the previous day's comparison

    # Assuming 'tickers' is your list of ticker symbols
    print("Tickers from test data:", tickers)

    # Calculate RMSE and print the comparison table
    calculate_rmse_and_print_table(tickers, actual_prices, predicted_prices, previous_day_prices)
    plot_differences(tickers, actual_prices, predicted_prices, previous_day_prices)

    # Generate random walks for comparison
    start_prices = previous_day_prices  # Assuming this is correct for your case
    # Generate a single-step forecast for each stock without drift
    random_walks_no_drift = start_prices + np.random.normal(0, 1, len(start_prices))

    # Generate a single-step forecast for each stock with drift
    random_walks_with_drift = start_prices + np.random.normal(0.1, 1, len(start_prices))

    # Calculate and print statistics
    calculate_statistics(actual_prices, predicted_prices, random_walks_no_drift, random_walks_with_drift)


if __name__ == "__main__":
    main()
