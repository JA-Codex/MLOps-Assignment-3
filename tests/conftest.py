import pytest

from app import create_app
from src import etl_fetch, preprocess, train_forecast


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("btc")
    raw, clean = root / "raw.csv", root / "clean.csv"
    etl_fetch.fetch(str(raw), source="synthetic")
    preprocess.preprocess(str(raw), str(clean))
    result = train_forecast.train(str(clean), root / "models", max_mape=100)
    assert result["registered"]
    return root / "models"


@pytest.fixture()
def client(model_dir):
    app = create_app({"MODEL_DIR": str(model_dir), "RELOAD_TOKEN": "secret"})
    return app.test_client()


@pytest.fixture()
def empty_client(tmp_path):
    return create_app({"MODEL_DIR": str(tmp_path)}).test_client()
