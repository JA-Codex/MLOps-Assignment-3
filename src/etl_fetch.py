"""Pull daily BTC-USD candles and store them as raw CSV (incremental)."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume"]
DEFAULT_START = "2018-01-01"


def default_raw_path() -> str:
    return str(Path(os.environ.get("DATA_DIR", "data")) / "raw" / "btc_usd.csv")


def fetch_yfinance(start: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so the API path stays light

    df = yf.download(
        "BTC-USD", start=start, interval="1d", auto_adjust=False, progress=False
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns=str.lower)
    return df[COLUMNS]


def synthetic(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """Offline geometric random walk, for tests/CI only."""
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0.0004, 0.03, n)))
    end = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1e9, 3e9, n),
        }
    )


def fetch(out_path: str, source: str = "yfinance", full: bool = False) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = None
    start = os.environ.get("BTC_START", DEFAULT_START)
    if out.exists() and not full:
        existing = pd.read_csv(out, parse_dates=["date"])
        start = (existing["date"].max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    new = synthetic() if source == "synthetic" else fetch_yfinance(start)
    new["date"] = pd.to_datetime(new["date"]).dt.tz_localize(None).dt.normalize()
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    new = new[new["date"] < today]  # drop today's incomplete candle

    df = new if existing is None else pd.concat([existing, new])
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    tmp = out.with_suffix(".tmp")
    df[COLUMNS].to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"raw rows={len(df)} last_date={df['date'].max().date()} -> {out}")
    return len(df)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=default_raw_path())
    p.add_argument("--source", default=os.environ.get("BTC_SOURCE", "yfinance"))
    p.add_argument("--full", action="store_true", help="ignore existing file")
    a = p.parse_args()
    fetch(a.out, a.source, a.full)


if __name__ == "__main__":
    main()
