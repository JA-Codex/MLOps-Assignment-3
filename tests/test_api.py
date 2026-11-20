def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["model_version"]
    assert body["uptime_seconds"] >= 0


def test_health_degraded_without_model(empty_client):
    r = empty_client.get("/health")
    assert r.status_code == 503
    assert r.get_json()["status"] == "degraded"


def test_predict_default_is_7_days(client):
    r = client.post("/predict")
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["forecast"]) == 7
    dates = [f["date"] for f in body["forecast"]]
    assert dates == sorted(dates)
    assert all(f["predicted_close"] > 0 for f in body["forecast"])


def test_predict_forecast_days(client):
    r = client.post("/predict", json={"forecast_days": 3})
    assert r.status_code == 200
    assert len(r.get_json()["forecast"]) == 3


def test_predict_custom_history(client):
    history = [30000 + 10 * i for i in range(60)]
    r = client.post("/predict", json={"history": history})
    assert r.status_code == 200
    assert r.get_json()["history_source"] == "request"


def test_predict_validation_errors(client):
    bad = [
        {"forecast_days": 0},
        {"forecast_days": 8},
        {"forecast_days": "7"},
        {"forecast_days": True},
        {"history": [1, 2, 3]},
        {"history": ["a"] * 40},
        {"history": [-5] * 40},
        {"unknown": 1},
    ]
    for payload in bad:
        r = client.post("/predict", json=payload)
        assert r.status_code == 400, payload
        assert "error" in r.get_json()


def test_predict_invalid_json(client):
    r = client.post("/predict", data="{not json", content_type="application/json")
    assert r.status_code == 400
    r = client.post("/predict", json=[1, 2, 3])
    assert r.status_code == 400


def test_predict_503_without_model(empty_client):
    assert empty_client.post("/predict").status_code == 503


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.get_json()
    assert {"mae", "rmse", "mape", "model_version"} <= set(body)


def test_reload_requires_token(client):
    assert client.post("/reload").status_code == 401
    r = client.post("/reload", headers={"X-Reload-Token": "secret"})
    assert r.status_code == 200
    assert r.get_json()["reloaded"] is True


def test_unknown_route_is_json_404(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert "error" in r.get_json()
