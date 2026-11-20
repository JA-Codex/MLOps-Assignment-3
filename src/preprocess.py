"""Clean raw candles and write the engineered feature table."""

import argparse
import os
from pathlib import Path

import pandas as pd

from src.features import build_dataset, clean_raw


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def default_clean_path() -> str:
    return str(_data_dir() / "processed" / "btc_clean.csv")


def preprocess(inp: str, out: str, features_out: str = None) -> int:
    df = clean_raw(pd.read_csv(inp))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if features_out:
        X, Y = build_dataset(df.set_index("date")["close"])
        table = X.join(Y).dropna(subset=list(X.columns))
        Path(features_out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(features_out)
    print(f"clean rows={len(df)} -> {out}")
    return len(df)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default=str(_data_dir() / "raw" / "btc_usd.csv"))
    p.add_argument("--out", default=default_clean_path())
    p.add_argument(
        "--features-out", default=str(_data_dir() / "processed" / "features.csv")
    )
    a = p.parse_args()
    preprocess(a.inp, a.out, a.features_out)


if __name__ == "__main__":
    main()
