import pandas as pd
from statsmodels.tsa.stattools import coint


# Load the merged FX dataset
path = "data/merged_data.csv"
df = pd.read_csv(path, parse_dates=["Date"])
df = df.sort_values("Date").dropna()

# Run the Engle-Granger cointegration test on the two price series
statistic, p_value, critical_values = coint(
    df["EURUSD_Close"],
    df["GBPUSD_Close"],
    trend="c",
    autolag="AIC",
)

print("Engle-Granger Cointegration Test")
print("-" * 40)
print("Cointegration test statistic:", statistic)
print("p-value:", p_value)
print("Critical values:")
for level, value in zip(["1%", "5%", "10%"], critical_values):
    print(f"  {level}: {value}")

alpha = 0.05
if p_value < alpha:
    print("Conclusion: The two currency pairs are cointegrated at the 5% significance level.")
else:
    print("Conclusion: The two currency pairs are not cointegrated at the 5% significance level.")
