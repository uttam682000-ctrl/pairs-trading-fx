from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


INPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "hedge_ratio_data.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "residual_data.csv"
FIGURE_PATH = Path(__file__).resolve().parents[1] / "figures" / "residuals_plot.png"


def load_merged_data(input_path):
    """Load the merged EUR/USD and NZD/USD dataset from disk."""
    return pd.read_csv(input_path, parse_dates=["Date"])


def fit_ols_model(merged_data):
    """Fit an OLS regression of EURUSD_Close on NZDUSD_Close."""
    x = sm.add_constant(merged_data["NZDUSD_Close"])
    model = sm.OLS(merged_data["EURUSD_Close"], x).fit()
    return model


def add_residuals(merged_data, model):
    """Add regression residuals as a new column to the dataset."""
    merged_data = merged_data.copy()
    merged_data["Residual"] = model.resid
    return merged_data


def run_adf_test(residual_data):
    """Run the ADF test on the residual series."""
    return adfuller(residual_data["Residual"].dropna(), autolag="AIC")


def save_residual_data(residual_data, output_path):
    """Save the residual dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residual_data.to_csv(output_path, index=False)
    print(f"Saved residual data to {output_path}")


def plot_residuals(residual_data, figure_path):
    """Plot the residuals over time."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(residual_data["Date"], residual_data["Residual"], color="tab:blue")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.title("Regression Residuals")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Saved residual plot to {figure_path}")


def print_regression_results(model):
    """Print alpha, beta, and R-squared from the fitted regression."""
    print("Regression parameters")
    print("=" * 30)
    print("alpha:", model.params["const"])
    print("beta:", model.params["NZDUSD_Close"])
    print("R-squared:", model.rsquared)


def print_adf_conclusion(adf_result):
    """Print the ADF test results and whether the residuals are stationary."""
    statistic, p_value = adf_result[0], adf_result[1]
    print("\nAugmented Dickey-Fuller Test")
    print("=" * 35)
    print("ADF statistic:", statistic)
    print("p-value:", p_value)

    alpha = 0.05
    if p_value < alpha:
        print("Conclusion: The residuals are stationary at the 5% significance level.")
    else:
        print("Conclusion: The residuals are not stationary at the 5% significance level.")


def main():
    """Load the merged data, fit the regression, save residuals, plot them, and test stationarity."""
    merged_data = load_merged_data(INPUT_PATH)
    model = fit_ols_model(merged_data)
    print_regression_results(model)

    residual_data = add_residuals(merged_data, model)
    save_residual_data(residual_data, OUTPUT_PATH)
    plot_residuals(residual_data, FIGURE_PATH)

    adf_result = run_adf_test(residual_data)
    print_adf_conclusion(adf_result)


if __name__ == "__main__":
    main()
