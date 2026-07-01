from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller


INPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "spread_data.csv"


def load_spread_data(input_path):
    """Load the spread dataset from disk."""
    return pd.read_csv(input_path, parse_dates=["Date"])


def run_adf_test(spread_data):
    """Run the Augmented Dickey-Fuller test on the Spread column."""
    result = adfuller(spread_data["Spread"].dropna(), autolag="AIC")
    return result


def print_adf_results(result):
    """Print the ADF test statistic, p-value, critical values, and conclusion."""
    statistic, p_value, critical_values = result[0], result[1], result[4]

    print("Augmented Dickey-Fuller Test")
    print("=" * 35)
    print("ADF statistic:", statistic)
    print("p-value:", p_value)
    print("Critical values:")
    for level, value in critical_values.items():
        print(f"  {level}: {value}")

    alpha = 0.05
    if p_value < alpha:
        print("Conclusion: The spread is stationary at the 5% significance level.")
    else:
        print("Conclusion: The spread is not stationary at the 5% significance level.")


def main():
    """Load spread data and run the ADF test."""
    spread_data = load_spread_data(INPUT_PATH)
    result = run_adf_test(spread_data)
    print_adf_results(result)


if __name__ == "__main__":
    main()
