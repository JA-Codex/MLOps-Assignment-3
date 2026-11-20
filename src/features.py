"""Shared feature engineering (used by training, the API and the Airflow DAG)."""

import numpy as np
import pandas as pd

HORIZON = 7
MIN_HISTORY = 31  # 30-day rolling windows need 30 closes + 1 for the diff
FEATURE_COLS = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_7",
    "ma7_ratio",
    "ma30_ratio",
    "vol7",
    "vol30",
]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Type-coerce, drop invalid rows/duplicates, sort by date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    return df.reset_index(drop=True)


def make_features(close: pd.Series) -> pd.DataFrame:
    """Scale-free features (returns, MA ratios, volatility) from closes only."""
    logp = np.log(close)
    daily = logp.diff()
    feats = pd.DataFrame(index=close.index)
    for lag in (1, 2, 3, 5, 7):
        feats[f"ret_{lag}"] = logp.diff(lag)
    feats["ma7_ratio"] = close / close.rolling(7).mean() - 1
    feats["ma30_ratio"] = close / close.rolling(30).mean() - 1
    feats["vol7"] = daily.rolling(7).std()
    feats["vol30"] = daily.rolling(30).std()
    return feats[FEATURE_COLS]


def make_targets(close: pd.Series) -> pd.DataFrame:
    """y_h = log(close[t+h] / close[t]) for h = 1..HORIZON."""
    logp = np.log(close)
    cols = {f"y{h}": logp.shift(-h) - logp for h in range(1, HORIZON + 1)}
    return pd.DataFrame(cols, index=close.index)


def build_dataset(close: pd.Series):
    return make_features(close), make_targets(close)


def latest_features(close: pd.Series) -> pd.DataFrame:
    """One-row feature frame for the most recent close."""
    if len(close) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} closes, got {len(close)}")
    feats = make_features(close.reset_index(drop=True)).iloc[[-1]]
    if feats.isna().any().any():
        raise ValueError("could not compute features from the supplied history")
    return feats
