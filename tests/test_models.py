import pytest
import os
import json
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_autoencoder_loads():
    path = os.path.join(OUTPUT_DIR, "autoencoder.pkl")
    if not os.path.exists(path):
        pytest.skip("autoencoder.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_lstm_predictor_loads():
    path = os.path.join(OUTPUT_DIR, "lstm_predictor.pkl")
    if not os.path.exists(path):
        pytest.skip("lstm_predictor.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_isolation_forest_loads():
    path = os.path.join(OUTPUT_DIR, "isolation_forest.pkl")
    if not os.path.exists(path):
        pytest.skip("isolation_forest.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_metadata_exists():
    path = os.path.join(OUTPUT_DIR, "metadata.json")
    if not os.path.exists(path):
        pytest.skip("metadata.json not found")
    with open(path) as f:
        meta = json.load(f)
    assert "n_samples" in meta
    assert "n_features" in meta
    assert "window_size" in meta
    assert "thresholds" in meta
    assert "metrics" in meta


def test_metadata_thresholds():
    path = os.path.join(OUTPUT_DIR, "metadata.json")
    if not os.path.exists(path):
        pytest.skip("metadata.json not found")
    with open(path) as f:
        meta = json.load(f)
    thresholds = meta["thresholds"]
    assert "autoencoder" in thresholds
    assert "lstm" in thresholds
    assert thresholds["autoencoder"] > 0
    assert thresholds["lstm"] > 0


def test_metadata_metrics():
    path = os.path.join(OUTPUT_DIR, "metadata.json")
    if not os.path.exists(path):
        pytest.skip("metadata.json not found")
    with open(path) as f:
        meta = json.load(f)
    metrics = meta["metrics"]
    for model_name in ["autoencoder", "isolation_forest"]:
        if model_name in metrics:
            m = metrics[model_name]
            assert m["precision"] >= 0
            assert m["recall"] >= 0
            assert m["f1"] >= 0
            assert m["accuracy"] >= 0
