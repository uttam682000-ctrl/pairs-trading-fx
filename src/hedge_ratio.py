from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import yfinance as yf


TICKERS = ["EURUSD=X", "NZDUSD=X"]
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "hedge_ratio_data.csv"


def download_close_prices(ticker, start_date, end_date):
    """Download daily close prices for a single ticker and return a Date/Close frame."""
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    close_prices = data[["Close"]].copy()
    close_prices.index.name = "Date"
    close_prices.columns = [ticker]
    close_prices = close_prices.reset_index()
    return close_prices


def merge_price_series(ticker_a, ticker_b, start_date, end_date):
    """Download both series, keep only Date and Close, and merge on Date."""
    first_series = download_close_prices(ticker_a, start_date, end_date)
    second_series = download_close_prices(ticker_b, start_date, end_date)

    merged = first_series.merge(second_series, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def run_ols_regression(merged_data, dependent_col, independent_col):
    """Regress the dependent series on the independent series using OLS."""
    y = merged_data[dependent_col]
    x = merged_data[independent_col]

    x_with_constant = sm.add_constant(x)
    model = sm.OLS(y, x_with_constant).fit()
    return model


def save_merged_data(merged_data, output_path):
    """Persist the aligned data for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_data.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")


def main():
    """Download, merge, regress, and report the hedge ratio results."""
    merged_data = merge_price_series(TICKERS[0], TICKERS[1], START_DATE, END_DATE)

    renamed_columns = {
        "EURUSD=X": "EURUSD_Close",
        "NZDUSD=X": "NZDUSD_Close",
    }
    merged_data = merged_data.rename(columns=renamed_columns)

    model = run_ols_regression(merged_data, "EURUSD_Close", "NZDUSD_Close")

    print("Regression Summary")
    print("=" * 40)
    print(model.summary())
    print("\nHedge ratio (beta):", model.params["NZDUSD_Close"])
    print("R-squared:", model.rsquared)

    save_merged_data(merged_data, OUTPUT_PATH)


if __name__ == "__main__":
    main()
