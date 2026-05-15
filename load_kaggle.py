""" 
DESCRIPTION FROM THE KAGGLE PAGE
Earnings call transcripts scraped from https://www.fool.com/earnings-call-transcripts/.

Each row consists of a date, exchange, quarter, ticker and transcript.

Has not yet been cleaned.

Dataset: https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts

"""

from dotenv import load_dotenv
load_dotenv()

import sys
import argparse
import pandas as pd
from pathlib import Path

TRANSCRIPTS_DIR = Path("data/transcripts")
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA",
    "JPM", "GS", "MS", "BAC", "WFC",
    "TSLA", "NFLX", "AMD", "INTC", "QCOM",
    "V", "MA", "PYPL", "CRM", "ADBE", "ORCL",
]

MIN_WORDS = 500


def load_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    ext = path.suffix.lower()
    print(f"Loading {path.name}...")

    if ext == ".pkl":
        df = pd.read_pickle(path)
    elif ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        print(f"Unknown format: {ext}. Trying pickle...")
        df = pd.read_pickle(path)

    return df


def inspect(df: pd.DataFrame):
    """Print what's in the dataframe so we can find the right columns."""
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst row sample:")
    for col in df.columns:
        val = str(df[col].iloc[0])
        print(f"  {col}: {val[:120]}")


def detect_columns(df: pd.DataFrame) -> dict:
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    for candidate in ["ticker", "symbol", "stock", "company_ticker", "tick"]:
        if candidate in cols:
            mapping["ticker"] = cols[candidate]
            break

    for candidate in ["date", "earnings_date", "call_date", "quarter_date", "period", "year"]:
        if candidate in cols:
            mapping["date"] = cols[candidate]
            break

    for candidate in ["transcript", "text", "content", "body", "transcript_text", "article"]:
        if candidate in cols:
            mapping["text"] = cols[candidate]
            break

    return mapping


def clean_date(val) -> str:
    try:
        if val is None:
            return None
        # Strip timezone string like "ET" which pandas can't parse
        val_str = str(val).strip()
        for tz in [", 9:00 p.m. ET", ", 5:00 p.m. ET", ", 4:30 p.m. ET",
                   " ET", " EST", " EDT", " PT", " PST", " PDT"]:
            if val_str.endswith(tz):
                val_str = val_str[: -len(tz)].strip()
        # Also strip time portion entirely — we only need the date
        import re
        val_str = re.sub(r",?\s+\d+:\d+.*$", "", val_str).strip()
        return pd.to_datetime(val_str).strftime("%Y-%m-%d")
    except Exception:
        return None


def extract_transcripts(file_path: str, limit_per_ticker: int = 8):
    df = load_file(file_path)
    inspect(df)

    mapping = detect_columns(df)
    print(f"\nAuto-detected columns: {mapping}")

    if not mapping.get("ticker") or not mapping.get("text"):
        print("\nCould not auto-detect columns. Please check the column names above")
        print("and edit the TARGET_TICKERS and detect_columns function accordingly.")
        return

    saved, skipped = 0, 0
    counts = {}

    for _, row in df.iterrows():
        ticker = str(row.get(mapping["ticker"], "")).strip().upper()
        if not ticker or ticker == "NAN":
            skipped += 1
            continue
        if TARGET_TICKERS and ticker not in TARGET_TICKERS:
            continue
        if counts.get(ticker, 0) >= limit_per_ticker:
            continue

        date = clean_date(row.get(mapping.get("date", ""), None))
        if not date:
            # Try to build date from year/quarter if available
            year = row.get("year", None)
            quarter = row.get("quarter", None)
            if year and quarter:
                q_month = {"Q1": "02", "Q2": "05", "Q3": "08", "Q4": "11",
                           1: "02", 2: "05", 3: "08", 4: "11"}.get(quarter)
                if q_month:
                    date = f"{int(year)}-{q_month}-01"
            if not date:
                skipped += 1
                continue

        text = str(row.get(mapping["text"], "")).strip()
        if len(text.split()) < MIN_WORDS:
            skipped += 1
            continue

        out_path = TRANSCRIPTS_DIR / f"{ticker}_{date}.txt"
        if out_path.exists():
            counts[ticker] = counts.get(ticker, 0) + 1
            continue

        out_path.write_text(text, encoding="utf-8")
        counts[ticker] = counts.get(ticker, 0) + 1
        saved += 1
        if saved % 20 == 0:
            print(f"  {saved} transcripts saved...")

    print(f"\nDone. {saved} new transcripts saved, {skipped} skipped.")
    print(f"\nCoverage:")
    for ticker, count in sorted(counts.items()):
        print(f"  {ticker}: {count} transcripts")

    if saved == 0 and skipped == 0:
        print("\nNo matching tickers found. The dataset may use different ticker formats.")
        print("Check the column values printed above and update TARGET_TICKERS if needed.")

    print(f"\nNext: python3 ingestor.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to .pkl, .csv, or .parquet file")
    parser.add_argument("--limit", type=int, default=8, help="Max transcripts per ticker")
    args = parser.parse_args()
    extract_transcripts(args.file, limit_per_ticker=args.limit)
