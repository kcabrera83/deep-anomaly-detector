# API Documentation - Deep Anomaly Detector

## Base URL

```
http://localhost:5018
```

## Endpoints

### GET /

Serves the web dashboard (HTML).

**Response:** HTML page

---

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": ["autoencoder", "isolation_forest", "lstm"],
  "version": "1.0.0"
}
```

---

### GET /api/models

Returns model metadata and thresholds.

**Response:**
```json
{
  "models": {
    "autoencoder": {
      "type": "SimpleAutoencoder",
      "threshold": 0.0234
    },
    "isolation_forest": {
      "type": "IsolationForestDetector",
      "threshold": -0.5
    },
    "lstm": {
      "type": "SimpleLSTMPredictor",
      "threshold": 0.0156
    }
  },
  "metadata": {
    "n_samples": 5000,
    "n_features": 4,
    "window_size": 30,
    "anomaly_ratio": 0.05,
    "total_anomalies": 250,
    "thresholds": {
      "autoencoder": 0.0234,
      "lstm": 0.0156
    }
  }
}
```

---

### POST /api/detect

Run anomaly detection on generated sensor data.

**Request:**
```json
{
  "n_points": 500
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| n_points | integer | No | 500 | Number of data points to generate and analyze |

**Response:**
```json
{
  "timestamp": [0, 1, 2, ...],
  "sensors": {
    "pressure": [100.5, 101.2, 99.8, ...],
    "temperature": [75.3, 76.1, 74.9, ...],
    "flow_rate": [250.0, 248.5, 252.1, ...],
    "vibration": [0.5, 0.6, 0.4, ...]
  },
  "detections": {
    "autoencoder": {
      "anomalies_detected": 25,
      "scores": [0.0023, 0.0456, 0.0019, ...],
      "threshold": 0.0234
    },
    "isolation_forest": {
      "anomalies_detected": 30,
      "scores": [0.123, -0.567, 0.089, ...]
    }
  },
  "true_anomaly_mask": [false, false, true, ...],
  "true_anomaly_types": ["normal", "normal", "spike", ...]
}
```

---

### POST /api/forecast

Forecast sensor readings using LSTM model.

**Request:**
```json
{
  "n_points": 300
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| n_points | integer | No | 300 | Number of normal data points to generate |

**Response:**
```json
{
  "forecast_length": 50,
  "forecasts": [
    {
      "step": 30,
      "predicted": [100.5, 75.3, 250.0, 0.5],
      "actual": [100.2, 75.1, 251.0, 0.48]
    },
    ...
  ],
  "sensor_data": {
    "pressure": [100.5, 101.2, ...],
    "temperature": [75.3, 76.1, ...],
    "flow_rate": [250.0, 248.5, ...],
    "vibration": [0.5, 0.6, ...]
  }
}
```

---

### POST /api/compare

Compare all detection models on the same dataset.

**Request:**
```json
{
  "n_points": 500
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| n_points | integer | No | 500 | Number of data points for comparison |

**Response:**
```json
{
  "comparison": {
    "autoencoder": {
      "precision": 0.8500,
      "recall": 0.7800,
      "f1_score": 0.8137,
      "accuracy": 0.9650,
      "detected": 25,
      "true_positives": 20,
      "false_positives": 5,
      "false_negatives": 6
    },
    "isolation_forest": {
      "precision": 0.7333,
      "recall": 0.8800,
      "f1_score": 0.8000,
      "accuracy": 0.9520,
      "detected": 30,
      "true_positives": 22,
      "false_positives": 8,
      "false_negatives": 3
    }
  },
  "n_samples": 470
}
```

---

### GET /api/docs

Returns OpenAPI 3.0.0 specification.

**Response:**
```json
{
  "openapi": "3.0.0",
  "info": {"title": "Deep Anomaly Detector", "version": "1.0.0"},
  "paths": {
    "/api/health": {"get": {"summary": "Health check"}},
    "/api/models": {"get": {"summary": "Model info"}},
    "/api/detect": {"post": {"summary": "Detect anomalies in sensor data"}},
    "/api/forecast": {"post": {"summary": "Forecast sensor readings with LSTM"}},
    "/api/compare": {"post": {"summary": "Compare detection models performance"}}
  }
}
```

## Sensor Types

| Sensor | Description |
|--------|-------------|
| pressure | Wellhead pressure (PSI) |
| temperature | Operating temperature (deg F) |
| flow_rate | Production flow rate (bbl/day) |
| vibration | Equipment vibration level |

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request - missing or invalid parameters |
| 500 | Internal server error - includes traceback |
