from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "hedge_ratio_data.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "spread_data.csv"
FIGURE_PATH = Path(__file__).resolve().parents[1] / "figures" / "spread_plot.png"
BETA = 1.1246380639323166


def load_merged_data(input_path):
    """Load the merged EUR/USD and NZD/USD dataset from disk."""
    return pd.read_csv(input_path, parse_dates=["Date"])


def calculate_spread(merged_data, beta):
    """Create a spread column using the hedge ratio beta."""
    merged_data = merged_data.copy()
    merged_data["Spread"] = merged_data["EURUSD_Close"] - beta * merged_data["NZDUSD_Close"]
    return merged_data


def save_spread_data(spread_data, output_path):
    """Save the dataset with the spread column to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spread_data.to_csv(output_path, index=False)
    print(f"Saved spread data to {output_path}")


def plot_spread(spread_data, figure_path):
    """Plot the spread series and its mean/standard deviation bands."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    mean_spread = spread_data["Spread"].mean()
    std_spread = spread_data["Spread"].std()

    plt.figure(figsize=(12, 6))
    plt.plot(spread_data["Date"], spread_data["Spread"], label="Spread", color="tab:blue")
    plt.axhline(mean_spread, color="tab:red", linestyle="--", label="Mean")
    plt.axhline(mean_spread + 2 * std_spread, color="tab:green", linestyle=":", label="Mean + 2 SD")
    plt.axhline(mean_spread - 2 * std_spread, color="tab:orange", linestyle=":", label="Mean - 2 SD")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.title("EUR/USD vs NZD/USD Spread")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Saved plot to {figure_path}")


def print_summary_stats(spread_data):
    """Print descriptive statistics for the spread."""
    spread = spread_data["Spread"]
    print("Spread summary")
    print("=" * 30)
    print("Mean:", spread.mean())
    print("Standard deviation:", spread.std())
    print("Minimum:", spread.min())
    print("Maximum:", spread.max())


def main():
    """Load the merged dataset, calculate the spread, save it, plot it, and report stats."""
    merged_data = load_merged_data(INPUT_PATH)
    spread_data = calculate_spread(merged_data, BETA)
    save_spread_data(spread_data, OUTPUT_PATH)
    plot_spread(spread_data, FIGURE_PATH)
    print_summary_stats(spread_data)


if __name__ == "__main__":
    main()
