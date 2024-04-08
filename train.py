import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertConfig

from data_preprocessing import load_and_preprocess_data, create_sequences
from model import StockPredictionModel

# Define training parameters
config = {
    "start_date": "2020-01-01",
    "end_date": "2021-01-01",
    "data_filename": "sp500_stock_data.csv",
    "scaler_filename": "scaler.pkl",
    "model_filename": "stock_prediction_model.pth",
    "seq_length": 10,  # Sequence length for the model
    "batch_size": 13,  # Batch size for training
    "num_epochs": 30000,  # Number of training epochs
    "learning_rate": 1e-6,  # Learning rate for the optimizer
}


bert_config = BertConfig()

# Define the BERT configuration
# bert_config.hidden_size = 64
# Values I found to work well for this model are redacted for obvious reasons
# Above to demonstrate how to change the configuration


# Load and preprocess data
data, scaler, tickers = load_and_preprocess_data(config["data_filename"], start_date=config["start_date"], end_date=config["end_date"])
print("Tickers used for training:", tickers)
X, y = create_sequences(data, config["seq_length"])


# Define model parameters that are dependent on the data
num_stocks = len(tickers)
num_features = len(tickers)

# Check for NaN in data
if np.isnan(X).any() or np.isnan(y).any():
    raise ValueError("NaN values detected in X or y. Please check your data preprocessing.")

# Convert the sequences into PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create a DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StockPredictionModel(bert_config, num_stocks=num_stocks, num_features=num_features).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
criterion = torch.nn.MSELoss()

# Training loop
model.train()
for epoch in range(config["num_epochs"]):
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        # Move data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        # Create attention mask
        attention_mask = torch.ones(x_batch.shape[:2], dtype=torch.long).to(device)

        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        predictions = model(x_batch, attention_mask=attention_mask)
        # Calculate loss
        loss = criterion(predictions, y_batch)

        # Check for NaN in loss and exit if found
        if torch.isnan(loss):
            print(f"NaN detected in loss for epoch {epoch + 1}, batch {batch_idx + 1}")
            break

        # Backward pass
        loss.backward()
        # Zero the gradients if NaN detected in gradients
        if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
            print(f"NaN gradient detected in epoch {epoch + 1}, batch {batch_idx + 1}")
            optimizer.zero_grad()
            continue

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item()}")

print("Training complete.")

# Save the model and scaler
torch.save(model.state_dict(), config["model_filename"])
print("Model saved to stock_prediction_model.pth")

with open(config["scaler_filename"], 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {config['scaler_filename']}")
