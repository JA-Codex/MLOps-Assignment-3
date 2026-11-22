"""Train the 7-day BTC close forecaster, gate on validation error, save artifacts.

Model: direct multi-horizon GradientBoosting on scale-free features, predicting
7 log-returns (close[t+h]/close[t]); prices = last_close * exp(prediction).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.features import HORIZON, build_dataset, clean_raw

REJECTED_EXIT_CODE = 3


def _new_model() -> MultiOutputRegressor:
    gbr = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    return MultiOutputRegressor(gbr)


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err / y_true)) * 100),
    }


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def train(data_path, model_dir, test_days=30, max_mape=15.0) -> dict:
    df = clean_raw(pd.read_csv(data_path))
    close = df.set_index("date")["close"]
    X, Y = build_dataset(close)
    ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X, Y = X[ok], Y[ok]
    if len(X) < test_days + HORIZON + 100:
        raise ValueError(f"not enough labelled rows ({len(X)}) to train and validate")

    # Hold out the latest `test_days` forecast origins; purge HORIZON rows of the
    # training set so no training target overlaps the test window.
    split = len(X) - test_days
    X_tr, Y_tr = X.iloc[: split - HORIZON], Y.iloc[: split - HORIZON]
    X_te, Y_te = X.iloc[split:], Y.iloc[split:]

    model = _new_model().fit(X_tr, Y_tr)
    base = close.loc[X_te.index].to_numpy()[:, None]
    actual = base * np.exp(Y_te.to_numpy())
    predicted = base * np.exp(model.predict(X_te))
    metrics = _scores(actual, predicted)
    metrics["naive_mape"] = _scores(actual, np.repeat(base, HORIZON, axis=1))["mape"]
    metrics["n_test_origins"] = int(len(X_te))
    metrics["test_window"] = {
        "first_origin": str(X_te.index[0].date()),
        "last_origin": str(X_te.index[-1].date()),
    }

    latest_origin = X_te.index[-1]
    target_dates = close.loc[close.index > latest_origin].index[:HORIZON]
    metrics["latest_7_days"] = {
        "forecast_origin": str(latest_origin.date()),
        "first_target_date": str(target_dates[0].date()),
        "last_target_date": str(target_dates[-1].date()),
        "days_evaluated": len(target_dates),
        **_scores(actual[-1], predicted[-1]),
    }

    registered = metrics["mape"] <= max_mape
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%d%H%M%S")
    result = {"registered": bool(registered), "version": version, "metrics": metrics}

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write(out / "last_run.json", json.dumps(result, indent=2))
    if not registered:
        return result

    final = _new_model().fit(X, Y)  # refit on everything for serving
    tmp = out / "model.joblib.tmp"
    joblib.dump(final, tmp)
    os.replace(tmp, out / "model.joblib")
    tmp_csv = out / "recent_history.csv.tmp"
    df[["date", "close"]].tail(120).to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, out / "recent_history.csv")
    _atomic_write(
        out / "metrics.json",
        json.dumps(
            {"model_version": version, "evaluated_at": now.isoformat(), **metrics},
            indent=2,
        ),
    )
    meta = {
        "version": version,
        "trained_at": now.isoformat(),
        "last_date": str(df["date"].iloc[-1].date()),
        "last_close": float(df["close"].iloc[-1]),
        "n_rows": int(len(df)),
        "algorithm": "GradientBoosting direct 7-step (log-return targets)",
    }
    # Written LAST: the API watches this file's mtime to hot-reload.
    _atomic_write(out / "meta.json", json.dumps(meta, indent=2))
    return result


def main() -> None:
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(data_dir / "processed" / "btc_clean.csv"))
    p.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models"))
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument(
        "--max-mape", type=float, default=float(os.environ.get("MAX_MAPE", 15))
    )
    a = p.parse_args()
    result = train(a.data, a.model_dir, a.test_days, a.max_mape)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["registered"] else REJECTED_EXIT_CODE)


if __name__ == "__main__":
    main()
