# src/features/price_features.py

import pandas as pd
import numpy as np

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ─── Coerce numeric columns to floats ─────────────────────────────────────
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where Close is missing (couldn't convert)
    df.dropna(subset=["Close"], inplace=True)

    # ─── Ensure sorted by date ───────────────────────────────────────────────
    df.sort_values("Date", inplace=True)

    # ─── Compute returns ────────────────────────────────────────────────────
    df["return"] = df["Close"].pct_change()

    # Moving averages
    for window in (5, 10, 20):
        df[f"ma_{window}"] = df["Close"].rolling(window).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Volatility (20-day rolling std of returns)
    df["volatility_20"] = df["return"].rolling(20).std()

    return df

if __name__ == "__main__":
    # Quick smoke test
    import pandas as pd
    from pathlib import Path
    test_csv = Path(__file__).parent.parent.parent / "data" / "raw" / "AAPL_price_daily.csv"
    df = pd.read_csv(test_csv, parse_dates=["Date"])
    df2 = compute_technical_indicators(df)
    print(df2[["Date", "ma_5", "rsi_14", "volatility_20"]].dropna().head())
