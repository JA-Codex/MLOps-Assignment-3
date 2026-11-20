"""Kubeflow Pipelines (KFP v2) spec: ingest -> features -> train.

Compile:  python kubeflow/btc_kfp_pipeline.py            -> btc_kfp_pipeline.yaml
Image:    BTC_IMAGE=docker.io/<user>/btc-forecast:latest (must contain src/)
"""

import argparse
import os

from kfp import compiler, dsl

IMAGE = os.environ.get("BTC_IMAGE", "docker.io/DOCKERHUB_USER/btc-forecast:latest")


@dsl.container_component
def ingest(raw_data: dsl.Output[dsl.Dataset]):
    """Pull BTC-USD candles into a raw CSV artifact."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.etl_fetch"],
        args=["--out", raw_data.path, "--full"],
    )


@dsl.container_component
def features(
    raw_data: dsl.Input[dsl.Dataset],
    clean_data: dsl.Output[dsl.Dataset],
    feature_table: dsl.Output[dsl.Dataset],
):
    """Clean + engineer MA/volatility/return features."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.preprocess"],
        args=[
            "--inp",
            raw_data.path,
            "--out",
            clean_data.path,
            "--features-out",
            feature_table.path,
        ],
    )


@dsl.container_component
def train(
    clean_data: dsl.Input[dsl.Dataset],
    max_mape: float,
    model_dir: dsl.Output[dsl.Model],
):
    """Train the 7-day forecaster; fails the step if validation gate is missed."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.train_forecast"],
        args=[
            "--data",
            clean_data.path,
            "--model-dir",
            model_dir.path,
            "--max-mape",
            max_mape,
        ],
    )


@dsl.pipeline(name="btc-forecast-pipeline", description="Daily BTC 7-day forecaster")
def btc_pipeline(max_mape: float = 15.0):
    ingest_task = ingest()
    feat_task = features(raw_data=ingest_task.outputs["raw_data"])
    train_task = train(clean_data=feat_task.outputs["clean_data"], max_mape=max_mape)
    for t in (ingest_task, feat_task, train_task):
        t.set_caching_options(False)


def compile_pipeline(path: str = "btc_kfp_pipeline.yaml") -> str:
    compiler.Compiler().compile(btc_pipeline, package_path=path)
    return path


def submit_recurring(host: str, package: str, cron: str = "0 0 2 * * *") -> None:
    """Upload + schedule daily at 02:00 (KFP cron has a seconds field). Untested helper."""
    import kfp

    client = kfp.Client(host=host)
    exp = client.create_experiment("btc-forecast")
    client.create_recurring_run(
        experiment_id=exp.experiment_id,
        job_name="btc-daily-train",
        cron_expression=cron,
        pipeline_package_path=package,
        params={"max_mape": 15.0},
        enable_caching=False,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="btc_kfp_pipeline.yaml")
    ap.add_argument("--schedule-on", metavar="KFP_HOST", help="also create a daily run")
    a = ap.parse_args()
    out = compile_pipeline(a.out)
    print("compiled ->", out)
    if a.schedule_on:
        submit_recurring(a.schedule_on, out)
