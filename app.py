"""Flask API for Deep Anomaly Detector - Oil & Gas sensor monitoring."""

import os
import sys
import json
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template
from deep_anomaly.data_generator import SensorDataGenerator
from deep_anomaly.utils.sequence_processor import SequenceProcessor
from deep_anomaly.models.autoencoder import SimpleAutoencoder
from deep_anomaly.models.lstm_predictor import SimpleLSTMPredictor
from deep_anomaly.models.isolation_forest import IsolationForestDetector

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
proc = SequenceProcessor(window_size=30, stride=1)
gen = SensorDataGenerator(seed=42)

_models = {}


def _load_models():
    if not _models:
        ae_path = os.path.join(MODELS_DIR, "autoencoder.pkl")
        lstm_path = os.path.join(MODELS_DIR, "lstm_predictor.pkl")
        if_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
        if os.path.exists(ae_path):
            _models["autoencoder"] = SimpleAutoencoder.load(ae_path)
        if os.path.exists(lstm_path):
            _models["lstm"] = SimpleLSTMPredictor.load(lstm_path)
        if os.path.exists(if_path):
            _models["isolation_forest"] = IsolationForestDetector.load(if_path)


def _get_metadata():
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    _load_models()
    return jsonify({
        "status": "healthy",
        "models_loaded": list(_models.keys()),
        "version": "1.0.0",
    })


@app.route("/api/models", methods=["GET"])
def models_info():
    _load_models()
    info = {}
    for name, model in _models.items():
        info[name] = {
            "type": type(model).__name__,
            "threshold": model.threshold,
        }
    meta = _get_metadata()
    return jsonify({"models": info, "metadata": meta})


@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        _load_models()
        data = request.get_json(silent=True) or {}
        n_points = data.get("n_points", 500)
        normal, anomalous, true_mask, true_types = gen.generate_dataset(
            n_points=n_points, anomaly_ratio=0.05
        )

        windows = proc.prepare_train_data(anomalous, window_size=30)
        flat = windows.reshape(windows.shape[0], -1)

        results = {"timestamp": list(range(n_points)), "sensors": {}}
        for k, v in anomalous.items():
            results["sensors"][k] = [round(float(x), 4) for x in v]

        detections = {}
        if "autoencoder" in _models:
            preds, errors = _models["autoencoder"].detect(flat)
            detections["autoencoder"] = {
                "anomalies_detected": int(np.sum(preds == -1)),
                "scores": [round(float(x), 6) for x in errors],
                "threshold": _models["autoencoder"].threshold,
            }
        if "isolation_forest" in _models:
            preds, errors = _models["isolation_forest"].detect(flat)
            detections["isolation_forest"] = {
                "anomalies_detected": int(np.sum(preds == -1)),
                "scores": [round(float(x), 6) for x in errors],
            }

        results["detections"] = detections
        results["true_anomaly_mask"] = [bool(x) for x in true_mask]
        results["true_anomaly_types"] = [str(x) for x in true_types]
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/forecast", methods=["POST"])
def forecast():
    try:
        _load_models()
        data = request.get_json(silent=True) or {}
        n_points = data.get("n_points", 300)

        sensor_data = gen.generate_normal(n_points)
        normed = proc.normalize(sensor_data, fit=True)
        arr = np.column_stack([normed[k] for k in sorted(normed.keys())])

        forecasts = []
        if "lstm" in _models:
            model = _models["lstm"]
            for i in range(30, min(n_points, 30 + 50)):
                seq = arr[i - 30 : i]
                pred = model.predict_next(seq)
                forecasts.append({
                    "step": i,
                    "predicted": [round(float(x), 4) for x in pred],
                    "actual": [round(float(arr[i, j]), 4) for j in range(arr.shape[1])],
                })

        return jsonify({
            "forecast_length": len(forecasts),
            "forecasts": forecasts,
            "sensor_data": {
                k: [round(float(x), 4) for x in v]
                for k, v in sensor_data.items()
            },
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/compare", methods=["POST"])
def compare():
    try:
        _load_models()
        data = request.get_json(silent=True) or {}
        n_points = data.get("n_points", 500)
        _, anomalous, true_mask, _ = gen.generate_dataset(
            n_points=n_points, anomaly_ratio=0.05
        )
        windows = proc.prepare_train_data(anomalous, window_size=30)
        flat = windows.reshape(windows.shape[0], -1)
        true_labels = np.where(true_mask[: len(flat)], -1, 1)

        comparison = {}
        for name in ["autoencoder", "isolation_forest", "lstm"]:
            if name not in _models:
                continue
            model = _models[name]
            if name == "lstm":
                arr = proc.normalize(anomalous, fit=False)
                arr = np.column_stack([arr[k] for k in sorted(arr.keys())])
                seq_len = 30
                seqs = np.array([arr[i : i + seq_len] for i in range(0, len(arr) - seq_len)])
                tgts = np.array([arr[i + seq_len] for i in range(0, len(arr) - seq_len)])
                preds, errors = model.detect(seqs[: len(tgts)], tgts)
            else:
                preds, errors = model.detect(flat)

            tp = int(np.sum((preds == -1) & (true_labels[:len(preds)] == -1)))
            fp = int(np.sum((preds == -1) & (true_labels[:len(preds)] == 1)))
            fn = int(np.sum((preds == 1) & (true_labels[:len(preds)] == -1)))
            tn = int(np.sum((preds == 1) & (true_labels[:len(preds)] == 1)))
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)

            comparison[name] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "accuracy": round((tp + tn) / max(len(preds), 1), 4),
                "detected": int(np.sum(preds == -1)),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }

        return jsonify({"comparison": comparison, "n_samples": len(flat)})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/docs")
def api_docs():
    return jsonify({
        "openapi": "3.0.0",
        "info": {"title": "Deep Anomaly Detector", "version": "1.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/detect": {"post": {"summary": "Detect anomalies in sensor data"}},
            "/api/forecast": {"post": {"summary": "Forecast sensor readings with LSTM"}},
            "/api/compare": {"post": {"summary": "Compare detection models performance"}},
        }
    })


if __name__ == "__main__":
    _load_models()
    app.run(host="0.0.0.0", port=5018, debug=False)
