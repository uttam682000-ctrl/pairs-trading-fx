import pandas as pd
from pathlib import Path


def load_fx_series(path: str, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Date"],
    )
    return df[["Date", "Close"]].rename(columns={"Close": f"{symbol}_Close"})


if __name__ == "__main__":
    data_dir = Path("data")

    eurusd = load_fx_series(data_dir / "EURUSD.csv", "EURUSD")
    gbpusd = load_fx_series(data_dir / "GBPUSD.csv", "GBPUSD")

    merged = eurusd.merge(gbpusd, on="Date", how="inner").dropna()

    print(merged.head())
    print("\nDataset info:")
    merged.info()

    merged.to_csv(data_dir / "merged_data.csv", index=False)
    print(f"\nSaved merged dataset to {data_dir / 'merged_data.csv'}")
