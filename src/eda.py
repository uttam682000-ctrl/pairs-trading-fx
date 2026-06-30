import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Load the merged FX dataset
path = Path("data/merged_data.csv")
df = pd.read_csv(path, parse_dates=["Date"])
df = df.sort_values("Date").dropna()
df = df.set_index("Date")

# Create the figures directory if it does not already exist
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# 1. EUR/USD closing price over time
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["EURUSD_Close"], color="tab:blue", label="EUR/USD")
plt.title("EUR/USD Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "eurusd_close_over_time.png", dpi=300)
plt.close()

# 2. GBP/USD closing price over time
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["GBPUSD_Close"], color="tab:orange", label="GBP/USD")
plt.title("GBP/USD Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "gbpusd_close_over_time.png", dpi=300)
plt.close()

# 3. Both exchange rates on the same graph
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["EURUSD_Close"], color="tab:blue", label="EUR/USD")
plt.plot(df.index, df["GBPUSD_Close"], color="tab:orange", label="GBP/USD")
plt.title("EUR/USD and GBP/USD Closing Prices")
plt.xlabel("Date")
plt.ylabel("Close")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "both_exchange_rates.png", dpi=300)
plt.close()

# 4. Normalized prices where both series start at 100
normalized = df / df.iloc[0] * 100

plt.figure(figsize=(10, 4))
plt.plot(normalized.index, normalized["EURUSD_Close"], color="tab:blue", label="EUR/USD")
plt.plot(normalized.index, normalized["GBPUSD_Close"], color="tab:orange", label="GBP/USD")
plt.title("Normalized Exchange Rates (Base Value = 100)")
plt.xlabel("Date")
plt.ylabel("Normalized Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "normalized_exchange_rates.png", dpi=300)
plt.close()

# 5. Print summary statistics and correlation
print("Summary statistics:")
print(df.describe())

print("\nCorrelation between EUR/USD and GBP/USD:")
print(df["EURUSD_Close"].corr(df["GBPUSD_Close"]))

print(f"\nSaved plots to {figures_dir}")
