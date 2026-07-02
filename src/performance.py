from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "backtest.csv"
FIGURE_PATH = ROOT_DIR / "figures" / "performance.png"


def load_backtest_data(input_path: Path) -> pd.DataFrame:
    """Load the backtest dataset from disk."""
    return pd.read_csv(input_path, parse_dates=["Date"])


def calculate_performance_metrics(backtest_data: pd.DataFrame) -> dict:
    """Compute portfolio performance statistics from strategy returns."""
    strategy_returns = backtest_data["StrategyReturn"].dropna()

    if strategy_returns.empty:
        raise ValueError("No strategy returns available for performance analysis.")

    cumulative_return = (1 + strategy_returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(strategy_returns)) - 1
    annualized_volatility = strategy_returns.std() * (252**0.5)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else float("inf")

    equity_curve = pd.Series(1.0, index=backtest_data.index, dtype=float)
    valid_returns = backtest_data["StrategyReturn"].notna()
    equity_curve.loc[valid_returns] = (1 + strategy_returns).cumprod()

    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    max_drawdown = drawdown.min()
    win_rate = (strategy_returns > 0).mean() * 100

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "equity_curve": equity_curve,
        "drawdown": drawdown,
    }


def plot_performance(backtest_data: pd.DataFrame, metrics: dict, figure_path: Path) -> None:
    """Plot equity curve and drawdown curve in a single figure."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(backtest_data["Date"], metrics["equity_curve"], color="tab:blue")
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Cumulative Equity")
    axes[0].axhline(1, color="black", linestyle="--", linewidth=1)

    axes[1].plot(backtest_data["Date"], metrics["drawdown"], color="tab:red")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Saved performance plot to {figure_path}")


def print_metrics(metrics: dict) -> None:
    """Print performance metrics clearly."""
    print("Performance Metrics")
    print("=" * 30)
    print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")


def main() -> None:
    """Load backtest results, calculate metrics, and create plots."""
    backtest_data = load_backtest_data(INPUT_PATH)
    metrics = calculate_performance_metrics(backtest_data)
    plot_performance(backtest_data, metrics, FIGURE_PATH)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
