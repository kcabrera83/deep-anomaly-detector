"""Test suite for the Deep Anomaly Detector API endpoints."""

import os
import sys
import json
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient


def test_training():
    """Verify models exist after training."""
    print("\n--- Test: Training Artifacts ---")
    base = os.path.join(os.path.dirname(__file__), "outputs", "models")
    for fname in ["autoencoder.pkl", "lstm_predictor.pkl", "isolation_forest.pkl", "metadata.json"]:
        path = os.path.join(base, fname)
        exists = os.path.exists(path)
        status = "PASS" if exists else "FAIL"
        print(f"  [{status}] {fname}")
        assert exists, f"Missing: {path}"

    with open(os.path.join(base, "metadata.json")) as f:
        meta = json.load(f)
    print(f"  Metadata: {meta['n_samples']} samples, {meta['n_features']} features")
    print(f"  AE threshold: {meta['thresholds']['autoencoder']:.6f}")
    print(f"  LSTM threshold: {meta['thresholds']['lstm']:.6f}")


def test_api_endpoints():
    """Test all FastAPI API endpoints."""
    from app import app, _load_models

    _load_models()
    client = TestClient(app)

    print("\n--- Test: GET /api/health ---")
    r = client.get("/api/health")
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Models loaded: {data['models_loaded']}")
    assert r.status_code == 200
    assert len(data["models_loaded"]) == 3
    print("  [PASS] Health check")

    print("\n--- Test: GET /api/models ---")
    r = client.get("/api/models")
    data = r.json()
    print(f"  Status: {r.status_code}")
    for name, info in data["models"].items():
        print(f"  {name}: threshold={info['threshold']:.6f}")
    assert r.status_code == 200
    print("  [PASS] Models info")

    print("\n--- Test: POST /api/detect ---")
    r = client.post("/api/detect", json={"n_points": 500})
    data = r.json()
    print(f"  Status: {r.status_code}")
    for name, det in data.get("detections", {}).items():
        print(f"  {name}: {det['anomalies_detected']} anomalies detected")
    n_true = sum(1 for x in data.get("true_anomaly_mask", []) if x)
    print(f"  True anomalies in data: {n_true}")
    assert r.status_code == 200
    assert "autoencoder" in data["detections"]
    print("  [PASS] Detection endpoint")

    print("\n--- Test: POST /api/forecast ---")
    r = client.post("/api/forecast", json={"n_points": 200})
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Forecast steps: {data['forecast_length']}")
    assert r.status_code == 200
    assert data["forecast_length"] > 0
    print("  [PASS] Forecast endpoint")

    print("\n--- Test: POST /api/compare ---")
    r = client.post("/api/compare", json={"n_points": 500})
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Compared {len(data['comparison'])} models on {data['n_samples']} samples")
    for name, m in data["comparison"].items():
        print(f"  {name}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1_score']:.3f}")
    assert r.status_code == 200
    print("  [PASS] Compare endpoint")

    print("\n" + "=" * 50)
    print("  ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    test_training()
    test_api_endpoints()
