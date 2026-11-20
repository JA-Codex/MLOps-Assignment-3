"""Flask REST API serving the 7-day BTC-USD close forecast."""

import hmac
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from src.features import FEATURE_COLS, HORIZON, MIN_HISTORY, latest_features

log = logging.getLogger("btc-api")
MAX_HISTORY = 5000


class ValidationError(ValueError):
    """Raised for client mistakes -> HTTP 400."""


class ModelStore:
    """Thread-safe model holder that hot-reloads when meta.json changes."""

    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        self._lock = threading.Lock()
        self.model = None
        self.meta = {}
        self.metrics = {}
        self.history = None
        self._stamp = None

    def _disk_stamp(self):
        path = self.dir / "meta.json"
        return path.stat().st_mtime_ns if path.exists() else None

    def load(self) -> bool:
        with self._lock:
            stamp = self._disk_stamp()
            if stamp is None:
                log.warning("no model found in %s", self.dir)
                return False
            try:
                model = joblib.load(self.dir / "model.joblib")
                meta = json.loads((self.dir / "meta.json").read_text())
                metrics = json.loads((self.dir / "metrics.json").read_text())
                hist = pd.read_csv(
                    self.dir / "recent_history.csv", parse_dates=["date"]
                )
            except Exception:  # keep serving the old model on a bad/partial write
                log.exception("model load failed")
                return False
            self.model, self.meta, self.metrics, self.history = (
                model,
                meta,
                metrics,
                hist,
            )
            self._stamp = stamp
            log.info("loaded model version %s", meta.get("version"))
            return True

    def refresh_if_changed(self) -> None:
        stamp = self._disk_stamp()
        if stamp is not None and stamp != self._stamp:
            self.load()

    @property
    def version(self):
        return self.meta.get("version")


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def parse_predict_payload(payload: dict):
    unknown = set(payload) - {"forecast_days", "history"}
    if unknown:
        raise ValidationError(f"unknown field(s): {sorted(unknown)}")
    days = payload.get("forecast_days", HORIZON)
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= HORIZON:
        raise ValidationError(
            f"forecast_days must be an integer between 1 and {HORIZON}"
        )
    history = payload.get("history")
    if history is not None:
        if not isinstance(history, list) or not all(_is_number(v) for v in history):
            raise ValidationError("history must be a list of numbers")
        if not MIN_HISTORY <= len(history) <= MAX_HISTORY:
            raise ValidationError(
                f"history must contain {MIN_HISTORY}-{MAX_HISTORY} daily closes"
            )
        if not all(math.isfinite(v) and v > 0 for v in history):
            raise ValidationError("history values must be finite and > 0")
    return days, history


def create_app(config: dict = None) -> Flask:
    cfg = {
        "MODEL_DIR": os.environ.get("MODEL_DIR", "/app/models"),
        "MODEL_STAGE": os.environ.get("MODEL_STAGE", "production"),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
        "RELOAD_TOKEN": os.environ.get("RELOAD_TOKEN", ""),
    }
    cfg.update(config or {})
    logging.basicConfig(
        level=cfg["LOG_LEVEL"].upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = Flask(__name__)
    store = ModelStore(cfg["MODEL_DIR"])
    store.load()
    started = time.time()

    @app.get("/health")
    def health():
        store.refresh_if_changed()
        body = {
            "status": "ok" if store.model is not None else "degraded",
            "uptime_seconds": round(time.time() - started, 1),
            "model_version": store.version,
            "model_stage": cfg["MODEL_STAGE"],
        }
        return jsonify(body), (200 if store.model is not None else 503)

    @app.get("/metrics")
    def metrics():
        store.refresh_if_changed()
        if store.model is None:
            return jsonify(error="model not loaded"), 503
        return jsonify(store.metrics)

    @app.post("/predict")
    def predict():
        store.refresh_if_changed()
        if store.model is None:
            return jsonify(error="model not loaded"), 503
        if request.get_data().strip():
            payload = request.get_json(force=True, silent=True)
            if not isinstance(payload, dict):
                raise ValidationError("body must be a JSON object")
        else:
            payload = {}
        days, history = parse_predict_payload(payload)

        if history is None:
            close = store.history["close"].reset_index(drop=True)
            base_date = store.history["date"].iloc[-1].date()
            source = "stored"
        else:
            close = pd.Series(history, dtype=float)
            base_date = datetime.now(timezone.utc).date()
            source = "request"
        try:
            feats = latest_features(close)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        log_ret = store.model.predict(feats[FEATURE_COLS])[0]
        base_close = float(close.iloc[-1])
        prices = base_close * np.exp(log_ret)
        forecast = [
            {
                "date": str(base_date + timedelta(days=i + 1)),
                "predicted_close": round(float(prices[i]), 2),
            }
            for i in range(days)
        ]
        return jsonify(
            model_version=store.version,
            generated_at=datetime.now(timezone.utc).isoformat(),
            history_source=source,
            base_date=str(base_date),
            base_close=round(base_close, 2),
            forecast_days=days,
            forecast=forecast,
        )

    @app.post("/reload")
    def reload_model():
        token = cfg["RELOAD_TOKEN"]
        if token and not hmac.compare_digest(
            request.headers.get("X-Reload-Token", ""), token
        ):
            return jsonify(error="unauthorized"), 401
        if not store.load():
            return jsonify(error="no loadable model found"), 503
        return jsonify(reloaded=True, model_version=store.version)

    @app.errorhandler(ValidationError)
    def on_validation(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(HTTPException)
    def on_http(exc):
        return jsonify(error=exc.name, detail=exc.description), exc.code

    @app.errorhandler(Exception)
    def on_error(exc):
        log.exception("unhandled error")
        return jsonify(error="internal server error"), 500

    return app


app = create_app()
