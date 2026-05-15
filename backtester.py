from dotenv import load_dotenv
load_dotenv()


import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path("data")

SIGNAL_COLUMNS = [
    "overall_sentiment",
    "management_tone",
    "guidance_direction",
    "revenue_surprise_language",
    "eps_surprise_language",
]


def load_signals() -> pd.DataFrame:
    path = DATA_DIR / "all_signals.json"
    if not path.exists():
        raise FileNotFoundError("Run extractor.py first to generate data/all_signals.json")
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_returns() -> pd.DataFrame:
    path = DATA_DIR / "earnings_returns.json"
    if not path.exists():
        raise FileNotFoundError("Run ingestor.py first to generate data/earnings_returns.json")
    with open(path) as f:
        data = json.load(f)
    if not data:
        print("WARNING: earnings_returns.json is empty — no price returns matched.")
        print("Backtester will use signals only (no ret validation).")
        # Return a dummy dataframe with the right columns so the join produces empty result
        return pd.DataFrame(columns=["ticker", "earnings_date", "return_1d", "return_5d", "close_t0"])
    df = pd.DataFrame(data)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df


def join_signals_returns(signals: pd.DataFrame, returns: pd.DataFrame, window_days: int = 5) -> pd.DataFrame:

    if returns.empty:
        print("No return events — run ingestor.py first.")
        return pd.DataFrame()
    rows = []
    for _, sig in signals.iterrows():
        tr = returns[returns["ticker"] == sig["ticker"]].copy()
        if tr.empty:
            continue
        diffs = (tr["earnings_date"] - sig["date"]).abs()
        idx = diffs.idxmin()
        if diffs[idx].days <= window_days:
            rows.append({**sig.to_dict(), **tr.loc[idx].to_dict()})
    merged = pd.DataFrame(rows)
    print(f"Matched {len(merged)} signal/return pairs out of "
          f"{len(signals)} signals and {len(returns)} return events")
    return merged


def compute_group_alpha(df: pd.DataFrame, signal_col: str, return_col: str) -> pd.DataFrame:
   
    results = []
    groups = df.groupby(signal_col)

    for name, group in groups:
        n = len(group)
        avg_ret = group[return_col].mean()
        std_ret = group[return_col].std()
        hit_rate = (group[return_col] > 0).mean()

        # t-test vs zero
        if n > 2 and std_ret > 0:
            t_stat, p_val = stats.ttest_1samp(group[return_col].dropna(), 0)
        else:
            t_stat, p_val = np.nan, np.nan

        results.append({
            "signal_value": name,
            "n": n,
            "avg_return": avg_ret,
            "std_return": std_ret,
            "hit_rate": hit_rate,
            "t_stat": t_stat,
            "p_value": p_val,
            "significant": p_val < 0.05 if not np.isnan(p_val) else False,
        })

    return pd.DataFrame(results).sort_values("avg_return", ascending=False)


def compute_ic(df: pd.DataFrame, score_col: str, return_col: str) -> dict:
   
    clean = df[[score_col, return_col]].dropna()
    if len(clean) < 5:
        return {"ic": np.nan, "p_value": np.nan, "n": len(clean)}
    ic, p_val = stats.spearmanr(clean[score_col], clean[return_col])
    return {"ic": ic, "p_value": p_val, "n": len(clean)}


def compute_sharpe(returns: pd.Series, annualize: bool = True) -> float:
  
    if returns.std() == 0 or len(returns) < 2:
        return np.nan
    sr = returns.mean() / returns.std()
    if annualize:
        sr *= np.sqrt(4)  
    return sr


def run_backtest() -> dict:
    signals = load_signals()
    returns = load_returns()
    df = join_signals_returns(signals, returns)

    if df.empty:
        print("No matched pairs found. Check ")
        return {}

    results = {"n_events": len(df), "by_signal": {}, "ic": {}, "overall": {}}

    # --- Per-signal-type alpha ---
    for sig_col in SIGNAL_COLUMNS:
        if sig_col not in df.columns:
            continue
        print(f"\n--- {sig_col} ---")
        for ret_col in ["return_1d", "return_5d"]:
            alpha_df = compute_group_alpha(df, sig_col, ret_col)
            key = f"{sig_col}__{ret_col}"
            results["by_signal"][key] = alpha_df.to_dict(orient="records")
            print(alpha_df[["signal_value", "n", "avg_return", "hit_rate", "significant"]].to_string(index=False))

 
    if "sentiment_score" in df.columns:
        print("\n--- IC: sentiment_score vs returns ---")
        for ret_col in ["return_1d", "return_5d"]:
            ic_result = compute_ic(df, "sentiment_score", ret_col)
            results["ic"][ret_col] = ic_result
            print(f"  {ret_col}: IC={ic_result['ic']:.4f}  p={ic_result['p_value']:.4f}  n={ic_result['n']}")


    if "overall_sentiment" in df.columns:
        long_ret = df[df["overall_sentiment"] == "bullish"]["return_5d"]
        short_ret = df[df["overall_sentiment"] == "bearish"]["return_5d"]
        ls_returns = pd.concat([long_ret, -short_ret])  

        results["overall"]["long_short_sharpe"] = compute_sharpe(ls_returns)
        results["overall"]["bullish_avg_5d"] = long_ret.mean()
        results["overall"]["bearish_avg_5d"] = short_ret.mean()
        results["overall"]["total_events"] = len(df)

        print(f" Long/Short Simulation  ")
        print(f"  Bullish avg 5d return:  {long_ret.mean():.4f}")
        print(f"  Bearish avg 5d return:  {short_ret.mean():.4f}")
        print(f"  L/S annualized Sharpe:  {results['overall']['long_short_sharpe']:.4f}")


    out_path = DATA_DIR / "backtest_results.json"


    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nBacktest results saved -> {out_path}")

    return results


if __name__ == "__main__":
    run_backtest()
