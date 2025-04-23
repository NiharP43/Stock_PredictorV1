# src/features/sentiment_features.py

import pandas as pd

def compute_sentiment_features(
    news_df: pd.DataFrame, twitter_df: pd.DataFrame
) -> pd.DataFrame:
    """
    From raw news/tweet sentiment (date, sentiment):
      - daily avg news_sentiment
      - daily avg twitter_sentiment
      - combined 7-day and 30-day rolling sentiment
    """
    # Ensure date columns are datetime / strings in ISO format
    news_df = news_df.copy()
    twitter_df = twitter_df.copy()
    # Some news_df may have no rows
    if not news_df.empty:
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
    if not twitter_df.empty:
        twitter_df["date"] = pd.to_datetime(twitter_df["date"]).dt.date

    # Average sentiment by day
    news_daily = (
        news_df.groupby("date")["sentiment"]
        .mean()
        .rename("news_sentiment")
    )
    twitter_daily = (
        twitter_df.groupby("date")["sentiment"]
        .mean()
        .rename("twitter_sentiment")
    )

    # Merge into single DataFrame
    df = pd.concat([news_daily, twitter_daily], axis=1).sort_index()
    # Fill missing days with 0 sentiment
    df.fillna(0, inplace=True)

    # Rolling features on combined sentiment
    df["combined_sentiment"] = df[["news_sentiment", "twitter_sentiment"]].mean(axis=1)
    df["sentiment_7d"] = df["combined_sentiment"].rolling(7).mean()
    df["sentiment_30d"] = df["combined_sentiment"].rolling(30).mean()

    return df.reset_index()

if __name__ == "__main__":
    # Quick smoke test
    import pandas as pd
    from pathlib import Path

    raw = Path(__file__).parent.parent.parent / "data" / "raw"
    news = pd.read_csv(raw / "AAPL_news.csv")
    twitter = pd.read_csv(raw / "AAPL_twitter.csv")
    feat = compute_sentiment_features(news, twitter)
    print(feat.head())
