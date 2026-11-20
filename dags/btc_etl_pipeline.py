"""Daily BTC-USD pipeline: fetch -> preprocess -> train/evaluate -> gate -> reload API."""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

sys.path.insert(0, os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))
from src import etl_fetch, preprocess, train_forecast  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", "/opt/airflow/data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/airflow/models"))
RAW = DATA_DIR / "raw" / "btc_usd.csv"
CLEAN = DATA_DIR / "processed" / "btc_clean.csv"
FEATURES = DATA_DIR / "processed" / "features.csv"
MAX_MAPE = float(os.environ.get("MAX_MAPE", "15"))


def fetch_data():
    return etl_fetch.fetch(str(RAW), source=os.environ.get("BTC_SOURCE", "yfinance"))


def preprocess_features():
    return preprocess.preprocess(str(RAW), str(CLEAN), str(FEATURES))


def train_evaluate():
    return train_forecast.train(str(CLEAN), str(MODEL_DIR), max_mape=MAX_MAPE)


def passed_threshold(ti):
    result = ti.xcom_pull(task_ids="train_evaluate")
    print("training result:", result)
    return bool(result and result["registered"])


def reload_api():
    url = os.environ.get("API_RELOAD_URL", "http://api:5000/reload")
    token = os.environ.get("RELOAD_TOKEN", "")
    resp = requests.post(url, headers={"X-Reload-Token": token}, timeout=30)
    resp.raise_for_status()
    print("API reloaded:", resp.json())


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="btc_etl_pipeline",
    description="Daily BTC-USD ETL, retraining, evaluation and API reload",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["btc", "mlops"],
) as dag:
    t_fetch = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    t_prep = PythonOperator(
        task_id="preprocess_features", python_callable=preprocess_features
    )
    t_train = PythonOperator(task_id="train_evaluate", python_callable=train_evaluate)
    t_gate = ShortCircuitOperator(
        task_id="register_if_passed", python_callable=passed_threshold
    )
    t_reload = PythonOperator(task_id="reload_api", python_callable=reload_api)

    t_fetch >> t_prep >> t_train >> t_gate >> t_reload
