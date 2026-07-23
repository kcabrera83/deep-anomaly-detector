"""FastAPI application for Deep Anomaly Detector using TensorFlow/Keras + PyTorch."""

import os
import sys
import json
import traceback
import numpy as np
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf
import torch

from deep_anomaly.data_generator import SensorDataGenerator
from deep_anomaly.utils.sequence_processor import SequenceProcessor

app = FastAPI(
    title="Deep Anomaly Detector",
    description="Deep Learning Anomaly Detection for Oil & Gas sensor monitoring (TF/Keras + PyTorch)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
proc = SequenceProcessor(window_size=30, stride=1)
gen = SensorDataGenerator(seed=42)

_models = {}


class AutoEncoderWrapper:
    def __init__(self, keras_model, threshold):
        self.keras_model = keras_model
        self.threshold = threshold

    def detect(self, flat_data):
        tensor = tf.convert_to_tensor(flat_data, dtype=tf.float32)
        reconstructed = self.keras_model(tensor).numpy()
        errors = np.mean(np.abs(flat_data - reconstructed), axis=1)
        preds = np.where(errors > self.threshold, -1, 1)
        return preds, errors


class LSTMDetectorWrapper:
    def __init__(self, torch_model, threshold):
        self.torch_model = torch_model
        self.threshold = threshold
        self.torch_model.eval()

    def predict_next(self, seq):
        with torch.no_grad():
            x = torch.FloatTensor(seq).unsqueeze(0)
            pred = self.torch_model(x)
        return pred.squeeze(0).numpy()

    def detect(self, seqs, targets):
        errors = []
        with torch.no_grad():
            for seq, tgt in zip(seqs, targets):
                x = torch.FloatTensor(seq).unsqueeze(0)
                pred = self.torch_model(x).squeeze(0).numpy()
                err = float(np.mean(np.abs(pred - tgt)))
                errors.append(err)
        errors = np.array(errors)
        preds = np.where(errors > self.threshold, -1, 1)
        return preds, errors


def _build_ae_model(input_dim, latent_dim=16):
    from deep_anomaly.train import AutoEncoder
    model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
    dummy = tf.zeros((1, input_dim))
    model(dummy)
    return model


def _build_lstm_model(input_size):
    from deep_anomaly.train import LSTMDetector
    model = LSTMDetector(input_size=input_size, hidden_size=64, num_layers=2)
    return model


def _load_models():
    if _models:
        return

    ae_path = os.path.join(MODELS_DIR, "autoencoder.pkl")
    lstm_path = os.path.join(MODELS_DIR, "lstm_predictor.pkl")

    if os.path.exists(ae_path + "_weights.weights.h5"):
        input_dim = 4 * 30
        keras_model = _build_ae_model(input_dim)
        keras_model.load_weights(ae_path + "_weights.weights.h5")
        with open(ae_path + "_meta.json") as f:
            meta = json.load(f)
        _models["autoencoder"] = AutoEncoderWrapper(keras_model, meta["threshold"])
        print("  Loaded TF/Keras Autoencoder")

    if os.path.exists(lstm_path + "_state.pth"):
        input_size = 4
        torch_model = _build_lstm_model(input_size)
        torch_model.load_state_dict(torch.load(lstm_path + "_state.pth", weights_only=True))
        with open(lstm_path + "_meta.json") as f:
            meta = json.load(f)
        _models["lstm"] = LSTMDetectorWrapper(torch_model, meta["threshold"])
        print("  Loaded PyTorch LSTM")


def _get_metadata():
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


class DetectRequest(BaseModel):
    pressure: Optional[List[float]] = None
    temperature: Optional[List[float]] = None
    flow_rate: Optional[List[float]] = None
    vibration: Optional[List[float]] = None
    n_points: int = 500


class ForecastRequest(BaseModel):
    pressure: Optional[List[float]] = None
    temperature: Optional[List[float]] = None
    flow_rate: Optional[List[float]] = None
    vibration: Optional[List[float]] = None
    n_points: int = 300


class CompareRequest(BaseModel):
    pressure: Optional[List[float]] = None
    temperature: Optional[List[float]] = None
    flow_rate: Optional[List[float]] = None
    vibration: Optional[List[float]] = None
    n_points: int = 500


@app.on_event("startup")
async def startup_event():
    _load_models()


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": list(_models.keys()),
        "version": "2.0.0",
        "frameworks": ["tensorflow", "pytorch", "tsfresh"],
    }


@app.get("/api/models")
async def models_info():
    info = {}
    for name, model in _models.items():
        info[name] = {
            "type": type(model).__name__,
            "threshold": model.threshold,
        }
    meta = _get_metadata()
    return {"models": info, "metadata": meta}


def _build_sensor_data(request_data: Dict[str, Optional[List[float]]], n_points: int):
    """Build sensor data dict from request arrays or generate synthetic fallback."""
    provided = {k: v for k, v in request_data.items() if v is not None}
    if len(provided) == 4:
        sensor_data = {}
        for key in ["pressure", "temperature", "flow_rate", "vibration"]:
            sensor_data[key] = np.array(provided[key], dtype=np.float64)
        return sensor_data, False
    return gen.generate_normal(n_points), True


@app.post("/api/detect")
async def detect(request: DetectRequest):
    try:
        request_dict = {
            "pressure": request.pressure,
            "temperature": request.temperature,
            "flow_rate": request.flow_rate,
            "vibration": request.vibration,
        }
        sensor_data, is_synthetic = _build_sensor_data(request_dict, request.n_points)

        normal, anomalous, true_mask, true_types = gen.generate_dataset(
            n_points=len(next(iter(sensor_data.values()))), anomaly_ratio=0.05
        )

        if not is_synthetic:
            for k in sensor_data:
                if k in anomalous:
                    anomalous[k] = sensor_data[k].copy()

        windows = proc.prepare_train_data(anomalous, window_size=30)
        flat = windows.reshape(windows.shape[0], -1)

        n_actual = len(next(iter(anomalous.values())))
        results: Dict[str, Any] = {"timestamp": list(range(n_actual)), "sensors": {}, "data_source": "user_provided" if not is_synthetic else "synthetic"}
        for k, v in anomalous.items():
            results["sensors"][k] = [round(float(x), 4) for x in v]

        detections = {}
        if "autoencoder" in _models:
            preds, errors = _models["autoencoder"].detect(flat)
            detections["autoencoder"] = {
                "anomalies_detected": int(np.sum(preds == -1)),
                "scores": [round(float(x), 6) for x in errors],
                "threshold": _models["autoencoder"].threshold,
                "framework": "tensorflow",
            }

        results["detections"] = detections
        results["true_anomaly_mask"] = [bool(x) for x in true_mask]
        results["true_anomaly_types"] = [str(x) for x in true_types]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})


@app.post("/api/forecast")
async def forecast(request: ForecastRequest):
    try:
        request_dict = {
            "pressure": request.pressure,
            "temperature": request.temperature,
            "flow_rate": request.flow_rate,
            "vibration": request.vibration,
        }
        sensor_data, is_synthetic = _build_sensor_data(request_dict, request.n_points)

        normed = proc.normalize(sensor_data, fit=True)
        arr = np.column_stack([normed[k] for k in sorted(normed.keys())])

        forecasts = []
        if "lstm" in _models:
            model = _models["lstm"]
            n_pts = len(arr)
            for i in range(30, min(n_pts, 30 + 50)):
                seq = arr[i - 30: i]
                pred = model.predict_next(seq)
                forecasts.append({
                    "step": i,
                    "predicted": [round(float(x), 4) for x in pred],
                    "actual": [round(float(arr[i, j]), 4) for j in range(arr.shape[1])],
                })

        return {
            "forecast_length": len(forecasts),
            "forecasts": forecasts,
            "data_source": "user_provided" if not is_synthetic else "synthetic",
            "sensor_data": {
                k: [round(float(x), 4) for x in v]
                for k, v in sensor_data.items()
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})


@app.post("/api/compare")
async def compare(request: CompareRequest):
    try:
        request_dict = {
            "pressure": request.pressure,
            "temperature": request.temperature,
            "flow_rate": request.flow_rate,
            "vibration": request.vibration,
        }
        sensor_data, is_synthetic = _build_sensor_data(request_dict, request.n_points)

        normal, anomalous, true_mask, _ = gen.generate_dataset(
            n_points=len(next(iter(sensor_data.values()))), anomaly_ratio=0.05
        )

        if not is_synthetic:
            for k in sensor_data:
                if k in anomalous:
                    anomalous[k] = sensor_data[k].copy()

        windows = proc.prepare_train_data(anomalous, window_size=30)
        flat = windows.reshape(windows.shape[0], -1)
        true_labels = np.where(true_mask[:len(flat)], -1, 1)

        comparison = {}
        for name in ["autoencoder", "lstm"]:
            if name not in _models:
                continue
            model = _models[name]
            if name == "lstm":
                arr = proc.normalize(anomalous, fit=False)
                arr = np.column_stack([arr[k] for k in sorted(arr.keys())])
                seq_len = 30
                seqs = np.array([arr[i: i + seq_len] for i in range(0, len(arr) - seq_len)])
                tgts = np.array([arr[i + seq_len] for i in range(0, len(arr) - seq_len)])
                preds, errors = model.detect(seqs[:len(tgts)], tgts)
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

        return {"comparison": comparison, "n_samples": len(flat), "data_source": "user_provided" if not is_synthetic else "synthetic"}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})


@app.get("/api/docs")
async def api_docs():
    return {
        "openapi": "3.0.0",
        "info": {"title": "Deep Anomaly Detector", "version": "2.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/detect": {"post": {"summary": "Detect anomalies in sensor data"}},
            "/api/forecast": {"post": {"summary": "Forecast sensor readings with LSTM"}},
            "/api/compare": {"post": {"summary": "Compare detection models performance"}},
        }
    }


if __name__ == "__main__":
    import uvicorn
    _load_models()
    uvicorn.run(app, host="0.0.0.0", port=5018)
