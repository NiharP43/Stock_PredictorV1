# src/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# ─── Locate and load .env ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
load_dotenv(BASE_DIR / ".env")

# ─── API KEYS ─────────────────────────────────────────────────────────────────
NEWSAPI_KEY      = os.getenv("NEWSAPI_KEY")
TWITTER_BEARER   = os.getenv("TWITTER_BEARER")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")

# ─── DATA DIRECTORIES ──────────────────────────────────────────────────────────
RAW_DATA_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# ─── DEFAULT PARAMETERS ────────────────────────────────────────────────────────
DEFAULT_TICKER = "AAPL"
START_DATE     = "2010-01-01"
END_DATE       = None  # use today’s date if None

if __name__ == "__main__":
    print("🔧 Config loaded successfully!")
    print("  NEWSAPI_KEY:", NEWSAPI_KEY)
    print("  TWITTER_BEARER:", TWITTER_BEARER)
    print("  AlphaVantage:", ALPHAVANTAGE_KEY)
    print("  Raw data dir:", RAW_DATA_DIR)
    print("  Processed data dir:", PROCESSED_DATA_DIR)
