import pandas as pd
from statsmodels.tsa.stattools import adfuller


# Load the merged FX dataset
path = "data/merged_data.csv"
df = pd.read_csv(path, parse_dates=["Date"])
df = df.sort_values("Date").dropna()

# Define a helper function to run the ADF test and print the results clearly

def run_adf(series_name: str, series: pd.Series) -> None:
    result = adfuller(series, autolag="AIC")

    print(f"\nADF test for {series_name}")
    print("-" * 40)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value}")

    alpha = 0.05
    if result[1] < alpha:
        print(f"Conclusion: {series_name} is stationary at the 5% significance level.")
    else:
        print(f"Conclusion: {series_name} is not stationary at the 5% significance level.")


# Run the test for each series
run_adf("EURUSD_Close", df["EURUSD_Close"])
run_adf("GBPUSD_Close", df["GBPUSD_Close"])
