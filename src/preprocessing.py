# src/preprocessing.py

import pandas as pd
from pathlib import Path

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DEFAULT_TICKER
from src.features.price_features import compute_technical_indicators
from src.features.sentiment_features import compute_sentiment_features

def load_raw_data(ticker: str):
    """Load raw CSVs into DataFrames."""
    base = RAW_DATA_DIR
    price_daily = pd.read_csv(base / f"{ticker}_price_daily.csv", parse_dates=["Date"])
    news = pd.read_csv(base / f"{ticker}_news.csv")
    twitter = pd.read_csv(base / f"{ticker}_twitter.csv")
    return price_daily, news, twitter

def merge_and_process(ticker: str):
    price_df, news_df, twitter_df = load_raw_data(ticker)

    # 1. Price-based features
    price_feat = compute_technical_indicators(price_df)

    # 2. Sentiment-based features
    sentiment_feat = compute_sentiment_features(news_df, twitter_df)
    sentiment_feat.rename(columns={"date": "Date"}, inplace=True)
    sentiment_feat["Date"] = pd.to_datetime(sentiment_feat["Date"])

    # 3. Merge on Date
    df = pd.merge(price_feat, sentiment_feat, on="Date", how="left")

    # 4. Fill any remaining NaNs for sentiment with zeros
    for col in ["news_sentiment", "twitter_sentiment", "combined_sentiment", "sentiment_7d", "sentiment_30d"]:
        if col in df:
            df[col] = df[col].fillna(0)


    # 5. Write out processed features
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / f"{ticker}_features.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved processed features to {out_path}")

if __name__ == "__main__":
    ticker = DEFAULT_TICKER
    merge_and_process(ticker)
