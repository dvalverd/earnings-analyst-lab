from dotenv import load_dotenv
load_dotenv()

import json
import time
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path("data")
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
PRICES_DIR = DATA_DIR / "prices"

TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
PRICES_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "GS",
    "AMD", "INTC", "QCOM", "TSLA", "NFLX", "BAC", "WFC",
    "V", "MA", "PYPL", "CRM", "ADBE", "ORCL",
]


def get_earnings_dates(ticker: str, years_back: int = 7) -> list:
    try:
        import pytz
        stock = yf.Ticker(ticker)
        calendar = stock.earnings_dates
        if calendar is None or calendar.empty:
            raise ValueError("empty")
        cutoff = datetime.now() - timedelta(days=365 * years_back)
        cutoff_aware = pytz.timezone("America/New_York").localize(cutoff)
        return calendar[calendar.index > cutoff_aware].index.strftime("%Y-%m-%d").tolist()
    except Exception as e:
        print(f"  Earnings dates unavailable ({e}), using fallback")
        dates = []
        for year in [2022, 2023, 2024]:
            for month in ["02-01", "05-01", "08-01", "11-01"]:
                dates.append(f"{year}-{month}")
        return dates


def save_price_data(ticker: str) -> None:
    end = datetime.now().strftime("%Y-%m-%d")
    start = "2018-01-01"  
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            print(f"  No price data for {ticker}")
            return
     
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        path = PRICES_DIR / f"{ticker}.csv"
        df.to_csv(path)
        print(f"  Saved price data : {path}")
    except Exception as e:
        print(f"  Price download failed for {ticker}: {e}")


def compute_returns(ticker: str, earnings_date: str) -> dict:
    path = PRICES_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    try:
       
        df = pd.read_csv(path, index_col=0)
        
        df = df[~df.index.isin(["Price", "Ticker"])]
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()].sort_index()

   
        if "Close" not in df.columns:
            print(f"  No Close column for {ticker}, columns: {df.columns.tolist()}")
            return None

        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df[df["Close"].notna()]

        date = pd.Timestamp(earnings_date)
        future = df[df.index >= date]
        if len(future) < 6:
            return None

        close_t0 = float(future["Close"].iloc[0])
        close_t1 = float(future["Close"].iloc[1])
        close_t5 = float(future["Close"].iloc[5])

        return {
            "ticker": ticker,
            "earnings_date": earnings_date,
            "return_1d": (close_t1 - close_t0) / close_t0,
            "return_5d": (close_t5 - close_t0) / close_t0,
            "close_t0": close_t0,
        }
    except Exception as e:
        print(f"  Returns error {ticker} {earnings_date}: {e}")
        return None


def build_earnings_dataset() -> list:
    dataset = []
    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")
        save_price_data(ticker)
        time.sleep(0.5)
        dates = get_earnings_dates(ticker)
        print(f"  Found {len(dates)} earnings dates")
        for date in dates:
            ret = compute_returns(ticker, date)
            if ret:
                dataset.append(ret)

    out_path = DATA_DIR / "earnings_returns.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nDataset saved -> {out_path} ({len(dataset)} events)")
    return dataset


if __name__ == "__main__":
    dataset = build_earnings_dataset()
    if dataset:
        df = pd.DataFrame(dataset)
        print("\nSample:")
        print(df.head(5).to_string())
        print(f"\nAvg 1-day return: {df['return_1d'].mean():.4f}")
        print(f"Avg 5-day return: {df['return_5d'].mean():.4f}")
    else:
        print("\nNo return events matched. Check data/prices/ CSV files.")
