from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "zscore_data.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "signals.csv"
FIGURE_PATH = ROOT_DIR / "figures" / "signals_plot.png"


def load_zscore_data(input_path: Path) -> pd.DataFrame:
    """Load the z-score dataset from disk."""
    return pd.read_csv(input_path, parse_dates=["Date"])


def generate_positions(zscore_data: pd.DataFrame) -> pd.DataFrame:
    """Create trading positions from z-score thresholds and forward-fill them."""
    signal_data = zscore_data.copy()

    raw_positions = pd.Series(index=signal_data.index, dtype=float)
    raw_positions.loc[signal_data["ZScore"] > 2] = -1
    raw_positions.loc[signal_data["ZScore"] < -2] = 1
    raw_positions.loc[signal_data["ZScore"].abs() < 0.5] = 0

    signal_data["Position"] = raw_positions.ffill().fillna(0)
    return signal_data


def save_signals(signal_data: pd.DataFrame, output_path: Path) -> None:
    """Save the signal dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal_data.to_csv(output_path, index=False)
    print(f"Saved trading signals to {output_path}")


def plot_signals(signal_data: pd.DataFrame, figure_path: Path) -> None:
    """Plot the spread with buy and sell markers."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(signal_data["Date"], signal_data["Spread"], label="Spread", color="tab:blue")

    buy_signals = signal_data[signal_data["Position"] == 1]
    sell_signals = signal_data[signal_data["Position"] == -1]

    plt.scatter(
        buy_signals["Date"],
        buy_signals["Spread"],
        marker="^",
        color="tab:green",
        s=80,
        label="Buy",
    )
    plt.scatter(
        sell_signals["Date"],
        sell_signals["Spread"],
        marker="v",
        color="tab:red",
        s=80,
        label="Sell",
    )

    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.title("Spread Trading Signals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Saved signal plot to {figure_path}")


def main() -> None:
    """Run the full signal-generation workflow."""
    zscore_data = load_zscore_data(INPUT_PATH)
    signal_data = generate_positions(zscore_data)
    save_signals(signal_data, OUTPUT_PATH)
    plot_signals(signal_data, FIGURE_PATH)


if __name__ == "__main__":
    main()
