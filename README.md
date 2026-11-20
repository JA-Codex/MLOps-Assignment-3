# BTC-USD 7-day forecast — Flask · Docker · Airflow · Kubeflow · Kubernetes

```
yfinance -> etl_fetch -> preprocess -> train_forecast -> /app/models (PVC / volume)
                                              |                  |
              Airflow DAG (daily) / KFP pipeline            Flask+Gunicorn (3 pods, HPA 2-8)
                                              |                  ^
                                   POST /reload (token) ---------+      Ingress btc-forecast.local
GitHub Actions: black+flake8+pytest -> docker build -> push :sha + :latest
```
Run order is in the assignment guide. Quick local test: `pip install -r requirements-dev.txt && pytest`.
