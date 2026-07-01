import itertools
from pathlib import Path

import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint


TICKERS = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDCHF=X", "USDCAD=X"]
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"


def download_close_prices(tickers, start_date, end_date):
    """Download daily Close prices for the selected FX symbols and align them by Date."""
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        close_data = data["Close"].copy()
    else:
        close_data = data[["Close"]].copy()
        close_data.columns = [tickers[0]]

    close_data = close_data.reset_index()
    close_data = close_data.rename(columns={"Date": "Date"})
    close_data = close_data.sort_values("Date").set_index("Date")
    close_data = close_data.dropna()
    return close_data


def evaluate_all_pairs(aligned_prices):
    """Run Engle-Granger cointegration tests for every pair and rank them by p-value."""
    results = []

    for pair_1, pair_2 in itertools.combinations(aligned_prices.columns, 2):
        statistic, p_value, _ = coint(
            aligned_prices[pair_1],
            aligned_prices[pair_2],
            trend="c",
            autolag="AIC",
        )
        results.append(
            {
                "Pair 1": pair_1,
                "Pair 2": pair_2,
                "Cointegration statistic": statistic,
                "p-value": p_value,
            }
        )

    ranked_results = pd.DataFrame(results).sort_values("p-value", ascending=True).reset_index(drop=True)
    return ranked_results


def print_results(results):
    """Print the ranked cointegration table and the best candidate pair."""
    print("Engle-Granger Cointegration Results")
    print("=" * 40)
    print(results.to_string(index=False))
    print()

    best_pair = results.iloc[0]
    print("Best pair:")
    print(f"{best_pair['Pair 1']} vs {best_pair['Pair 2']}")
    print(f"p-value: {best_pair['p-value']:.6f}")


def main():
    """Download FX data, evaluate all pairs, and print the ranked results."""
    aligned_prices = download_close_prices(TICKERS, START_DATE, END_DATE)
    results = evaluate_all_pairs(aligned_prices)
    print_results(results)


if __name__ == "__main__":
    main()
