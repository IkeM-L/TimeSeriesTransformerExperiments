import pandas as pd

# Load the original dataset
filename = "sp500_stock_data.csv"
df = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)

# Remove the last day of data and save the remaining dataset
df_adjusted = df.iloc[:-1, :]
df_adjusted.to_csv("sp500_stock_data_adjusted.csv")

# Save the last day's data to a separate file
last_day_data = df.iloc[-1:, :]
last_day_data.to_csv("sp500_stock_data_last_day.csv")
