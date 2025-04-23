# src/data_ingest.py

from datetime import datetime

import pandas as pd
import yfinance as yf
import requests
import tweepy
from textblob import TextBlob

from src.config import (
    RAW_DATA_DIR,
    DEFAULT_TICKER,
    START_DATE,
    NEWSAPI_KEY,
    TWITTER_BEARER,
)

# Ensure raw data directory exists
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_price(
    ticker: str = DEFAULT_TICKER,
    start_date: str = START_DATE,
    end_date: str = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data for `ticker` from start_date to end_date.
    interval: '1d' or '1wk'.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    df.reset_index(inplace=True)
    return df


def save_price_csv(ticker: str):
    # Daily
    df_daily = fetch_price(ticker, interval="1d")
    daily_path = RAW_DATA_DIR / f"{ticker}_price_daily.csv"
    df_daily.to_csv(daily_path, index=False)
    print(f"Saved daily price to {daily_path}")

    # Weekly
    df_weekly = fetch_price(ticker, interval="1wk")
    weekly_path = RAW_DATA_DIR / f"{ticker}_price_weekly.csv"
    df_weekly.to_csv(weekly_path, index=False)
    print(f"Saved weekly price to {weekly_path}")


def fetch_news(ticker: str, from_date: str = START_DATE, to_date: str = None) -> pd.DataFrame:
    """
    Fetch headlines+descriptions for `ticker` and compute TextBlob sentiment polarity.
    """
    if to_date is None:
        to_date = datetime.today().strftime("%Y-%m-%d")

    url = (
        "https://newsapi.org/v2/everything"
        f"?q={ticker}&from={from_date}&to={to_date}"
        "&language=en&sortBy=publishedAt"
        f"&apiKey={NEWSAPI_KEY}"
    )
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"⚠️  Warning: NewsAPI HTTPError (skipping news): {e}")
        return pd.DataFrame(columns=["date", "sentiment"])

    articles = resp.json().get("articles", [])

    rows = []
    for art in articles:
        text = art.get("title", "") + " " + (art.get("description") or "")
        polarity = TextBlob(text).sentiment.polarity
        date = art.get("publishedAt", "")[:10]
        rows.append({"date": date, "sentiment": polarity})
    return pd.DataFrame(rows)


def save_news_csv(ticker: str):
    df = fetch_news(ticker)
    path = RAW_DATA_DIR / f"{ticker}_news.csv"
    df.to_csv(path, index=False)
    print(f"Saved news sentiment to {path}")


def fetch_twitter_sentiment(ticker: str, max_tweets: int = 500) -> pd.DataFrame:
    """
    Fetch recent tweets containing TICKER (no $), compute TextBlob polarity.
    """
    client = tweepy.Client(bearer_token=TWITTER_BEARER)
    # Remove the '$' to avoid the invalid cashtag operator error
    query = f"{ticker} lang:en -is:retweet"

    # Wrap in try/except to handle API errors gracefully
    try:
        resp = client.search_recent_tweets(query, max_results=100)
    except tweepy.errors.BadRequest as e:
        print(f"⚠️  Warning: Twitter API BadRequest (skipping Twitter): {e}")
        return pd.DataFrame(columns=["date", "sentiment"])
    except Exception as e:
        print(f"⚠️  Warning: Twitter API error (skipping Twitter): {e}")
        return pd.DataFrame(columns=["date", "sentiment"])

    data = resp.data or []
    rows = []
    for t in data[:max_tweets]:
        polarity = TextBlob(t.text).sentiment.polarity
        date_str = t.created_at.date().isoformat()
        rows.append({"date": date_str, "sentiment": polarity})

    return pd.DataFrame(rows)

def save_twitter_csv(ticker: str):
    df = fetch_twitter_sentiment(ticker)
    path = RAW_DATA_DIR / f"{ticker}_twitter.csv"
    df.to_csv(path, index=False)
    print(f"Saved twitter sentiment to {path}")


if __name__ == "__main__":
    ticker = DEFAULT_TICKER
    print(f"▶️  Ingesting data for {ticker} starting {START_DATE}…\n")

    # 1. Price data
    save_price_csv(ticker)

    # 2. News sentiment
    save_news_csv(ticker)

    # 3. Twitter sentiment
    save_twitter_csv(ticker)

    print("\n✅  Data ingestion complete. Check data/raw/ for CSVs.")
