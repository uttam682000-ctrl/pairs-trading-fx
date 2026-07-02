from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "signals.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "backtest.csv"
FIGURE_PATH = ROOT_DIR / "figures" / "equity_curve.png"


def load_signals(input_path: Path) -> pd.DataFrame:
    """Load the trading signals dataset from disk."""
    return pd.read_csv(input_path, parse_dates=["Date"])


def calculate_strategy_returns(signal_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily spread returns and strategy performance."""
    backtest_data = signal_data.copy()
    backtest_data = backtest_data.sort_values("Date").reset_index(drop=True)

    backtest_data["DailySpreadReturn"] = backtest_data["Spread"].pct_change()
    backtest_data["ShiftedPosition"] = backtest_data["Position"].shift(1)
    backtest_data["StrategyReturn"] = backtest_data["ShiftedPosition"] * backtest_data["DailySpreadReturn"]
    backtest_data["CumulativeStrategyReturn"] = (1 + backtest_data["StrategyReturn"]).cumprod() - 1

    return backtest_data


def save_backtest_data(backtest_data: pd.DataFrame, output_path: Path) -> None:
    """Save the backtest dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backtest_data.to_csv(output_path, index=False)
    print(f"Saved backtest data to {output_path}")


def plot_equity_curve(backtest_data: pd.DataFrame, figure_path: Path) -> None:
    """Plot the cumulative strategy return."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(
        backtest_data["Date"],
        backtest_data["CumulativeStrategyReturn"],
        label="Cumulative Strategy Return",
        color="tab:purple",
    )
    plt.axhline(0, color="black", linestyle="-", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Strategy Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Saved equity curve plot to {figure_path}")


def print_summary(backtest_data: pd.DataFrame) -> None:
    """Print key backtest summary metrics."""
    total_return = backtest_data["CumulativeStrategyReturn"].iloc[-1]
    number_of_trades = int((backtest_data["ShiftedPosition"] != 0).sum())

    print("Backtest Summary")
    print("=" * 30)
    print("Total Return:", round(total_return, 6))
    print("Number of trades:", number_of_trades)


def main() -> None:
    """Run the full backtesting workflow."""
    signal_data = load_signals(INPUT_PATH)
    backtest_data = calculate_strategy_returns(signal_data)
    save_backtest_data(backtest_data, OUTPUT_PATH)
    plot_equity_curve(backtest_data, FIGURE_PATH)
    print_summary(backtest_data)


if __name__ == "__main__":
    main()
