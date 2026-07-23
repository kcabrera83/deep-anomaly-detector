# User Guide - Deep Anomaly Detector

## Overview

The Deep Anomaly Detector is an AI-based anomaly detection system for oil & gas sensor monitoring. It uses autoencoders, LSTM time-series forecasting, and Isolation Forest to detect anomalies in pressure, temperature, flow rate, and vibration sensor data.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd deep-anomaly-detector
pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

Generates 5000 timesteps of synthetic sensor data and trains all three models (autoencoder, LSTM, Isolation Forest). Models are saved to `outputs/models/`.

### Start the API Server

```bash
python app.py
```

The server starts on `http://localhost:5018`.

### Open the Dashboard

Navigate to `http://localhost:5018` in your browser.

### Run Tests

```bash
python test_api.py
```

## Dashboard Features

- **Anomaly Detection**: Run detection on generated sensor data with visual charts
- **LSTM Forecasting**: See predicted vs actual sensor readings
- **Model Comparison**: Compare precision, recall, F1, and accuracy across models
- **Model Statistics**: View thresholds and training metadata

## How It Works

### Autoencoder
- Trained on normal sensor data only
- High reconstruction error = anomaly
- Threshold set at 95th percentile of normal data reconstruction errors

### LSTM Predictor
- Predicts next sensor reading from previous 30 timesteps
- High prediction error = anomaly
- Threshold set at 95th percentile of normal prediction errors

### Isolation Forest
- Unsupervised ensemble method
- Isolates anomalies by random feature splits
- Contamination parameter at 5%

## API Usage

### Using curl

**Detect anomalies:**
```bash
curl -X POST http://localhost:5018/api/detect \
  -H "Content-Type: application/json" \
  -d '{"n_points": 500}'
```

**Forecast sensor readings:**
```bash
curl -X POST http://localhost:5018/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"n_points": 300}'
```

**Compare models:**
```bash
curl -X POST http://localhost:5018/api/compare \
  -H "Content-Type: application/json" \
  -d '{"n_points": 500}'
```

### Using Python

```python
import requests

# Detect anomalies
response = requests.post("http://localhost:5018/api/detect", json={"n_points": 500})
data = response.json()
for model, det in data["detections"].items():
    print(f"{model}: {det['anomalies_detected']} anomalies detected")

# Forecast
response = requests.post("http://localhost:5018/api/forecast", json={"n_points": 300})
forecasts = response.json()["forecasts"]
print(f"Forecast steps: {len(forecasts)}")
print(f"First prediction: {forecasts[0]['predicted']}")

# Compare models
response = requests.post("http://localhost:5018/api/compare", json={"n_points": 500})
comparison = response.json()["comparison"]
for model, metrics in comparison.items():
    print(f"{model}: F1={metrics['f1_score']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
```

### Using JavaScript

```javascript
// Detect anomalies
const response = await fetch("http://localhost:5018/api/detect", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ n_points: 500 })
});
const data = await response.json();
Object.entries(data.detections).forEach(([model, det]) => {
  console.log(`${model}: ${det.anomalies_detected} anomalies`);
});
```
