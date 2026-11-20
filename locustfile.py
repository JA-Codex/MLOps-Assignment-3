"""Optional load test (see guide):

locust -f locustfile.py --headless -u 100 -r 10 -t 3m --host http://btc-forecast.local
"""

from locust import HttpUser, between, task


class ApiUser(HttpUser):
    wait_time = between(0.01, 0.1)

    @task(5)
    def predict(self):
        self.client.post("/predict", json={"forecast_days": 7})

    @task(1)
    def health(self):
        self.client.get("/health")
