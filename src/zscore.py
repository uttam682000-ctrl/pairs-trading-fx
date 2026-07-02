from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "spread_data.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "zscore_data.csv"
FIGURE_DIR = ROOT_DIR / "figures"
WINDOW = 20


def load_spread_data(input_path: Path) -> pd.DataFrame:
    """Load the spread dataset from disk."""
    return pd.read_csv(input_path, parse_dates=["Date"])


def calculate_zscore(spread_data: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Add rolling mean, rolling standard deviation, and z-score columns."""
    zscore_data = spread_data.copy()
    zscore_data["RollingMean"] = (
        zscore_data["Spread"].rolling(window=window, min_periods=window).mean()
    )
    zscore_data["RollingStd"] = (
        zscore_data["Spread"].rolling(window=window, min_periods=window).std()
    )
    zscore_data["ZScore"] = (
        (zscore_data["Spread"] - zscore_data["RollingMean"])
        / zscore_data["RollingStd"].where(zscore_data["RollingStd"] != 0, pd.NA)
    )
    return zscore_data


def save_zscore_data(zscore_data: pd.DataFrame, output_path: Path) -> None:
    """Save the updated dataset with z-score columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zscore_data.to_csv(output_path, index=False)
    print(f"Saved z-score data to {output_path}")


def plot_spread_with_mean(zscore_data: pd.DataFrame, figure_path: Path) -> None:
    """Plot the spread series along with its rolling mean."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(zscore_data["Date"], zscore_data["Spread"], label="Spread", color="tab:blue")
    plt.plot(
        zscore_data["Date"],
        zscore_data["RollingMean"],
        label="Rolling Mean",
        color="tab:red",
        linestyle="--",
    )
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.title("Spread and 20-Day Rolling Mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Saved spread plot to {figure_path}")


def plot_zscore(zscore_data: pd.DataFrame, figure_path: Path) -> None:
    """Plot the z-score with reference lines at -2, 0, and +2."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(zscore_data["Date"], zscore_data["ZScore"], label="Z-Score", color="tab:green")
    plt.axhline(2, color="tab:green", linestyle=":", label="+2")
    plt.axhline(-2, color="tab:orange", linestyle=":", label="-2")
    plt.axhline(0, color="black", linestyle="-", linewidth=1, label="0")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.title("20-Day Rolling Z-Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Saved z-score plot to {figure_path}")


def main() -> None:
    """Run the full z-score workflow for the spread data."""
    spread_data = load_spread_data(INPUT_PATH)
    zscore_data = calculate_zscore(spread_data)
    save_zscore_data(zscore_data, OUTPUT_PATH)
    plot_spread_with_mean(zscore_data, FIGURE_DIR / "spread_with_rolling_mean.png")
    plot_zscore(zscore_data, FIGURE_DIR / "zscore_plot.png")


if __name__ == "__main__":
    main()
