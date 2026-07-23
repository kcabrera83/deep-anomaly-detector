import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert data["version"] == "1.0.0"


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "models" in data
    assert "metadata" in data


def test_api_docs(client):
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.0"
    assert "/api/detect" in data["paths"]
    assert "/api/forecast" in data["paths"]
    assert "/api/compare" in data["paths"]


def test_detect(client):
    response = client.post("/api/detect", json={"n_points": 200})
    assert response.status_code == 200
    data = response.get_json()
    assert "detections" in data
    assert "sensors" in data
    assert "true_anomaly_mask" in data
    assert "true_anomaly_types" in data


def test_detect_default_params(client):
    response = client.post("/api/detect", json={})
    assert response.status_code == 200
    data = response.get_json()
    assert "detections" in data


def test_forecast(client):
    response = client.post("/api/forecast", json={"n_points": 200})
    assert response.status_code == 200
    data = response.get_json()
    assert "forecasts" in data
    assert "forecast_length" in data
    assert "sensor_data" in data


def test_forecast_default_params(client):
    response = client.post("/api/forecast", json={})
    assert response.status_code == 200
    data = response.get_json()
    assert "forecasts" in data


def test_compare(client):
    response = client.post("/api/compare", json={"n_points": 300})
    assert response.status_code == 200
    data = response.get_json()
    assert "comparison" in data
    assert "n_samples" in data


def test_compare_contains_models(client):
    response = client.post("/api/compare", json={"n_points": 200})
    assert response.status_code == 200
    data = response.get_json()
    comparison = data["comparison"]
    for model_name, metrics in comparison.items():
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "accuracy" in metrics
