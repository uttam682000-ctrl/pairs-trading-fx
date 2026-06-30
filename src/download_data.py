import yfinance as yf
import os

# Create the data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download EUR/USD and GBP/USD data
eurusd = yf.download("EURUSD=X", start="2020-01-01", end="2025-01-01")
gbpusd = yf.download("GBPUSD=X", start="2020-01-01", end="2025-01-01")

# Save to CSV
eurusd.to_csv("data/EURUSD.csv")
gbpusd.to_csv("data/GBPUSD.csv")

print("Data downloaded successfully!")
