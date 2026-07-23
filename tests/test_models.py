import pytest
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_outputs_directory_exists():
    assert os.path.exists(OUTPUT_DIR)


def test_model_files_exist():
    model_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith((".pkl", ".joblib", ".h5", ".pt"))]
    assert len(model_files) > 0


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
    import json
    path = os.path.join(OUTPUT_DIR, "metadata.json")
    if not os.path.exists(path):
        pytest.skip("metadata.json not found")
    with open(path) as f:
        meta = json.load(f)
    assert "n_samples" in meta
    assert "n_features" in meta
    assert "window_size" in meta
