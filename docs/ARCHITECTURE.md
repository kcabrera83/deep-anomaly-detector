# Architecture - Deep Anomaly Detector

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Dashboard (HTML)                   │
│                    Port 5018 /                           │
├─────────────────────────────────────────────────────────┤
│                    Flask API Layer                       │
│      /api/detect  /api/forecast  /api/compare           │
├──────────┬──────────────┬──────────────┬─────────────────┤
│ Auto-    │ LSTM         │ Isolation    │ Sequence        │
│ encoder  │ Predictor    │ Forest       │ Processor       │
│ (NumPy)  │ (NumPy)      │ (sklearn)    │ (windowing)     │
├──────────┴──────────────┴──────────────┴─────────────────┤
│           Sensor Data Generator (4 sensors)               │
│    pressure / temperature / flow_rate / vibration         │
└─────────────────────────────────────────────────────────┘
```

## Components

### Data Layer

- **Sensor Data Generator**: Creates synthetic time-series data for 4 sensors
  - pressure, temperature, flow_rate, vibration
- **Anomaly Injection**: Adds spikes, drops, drifts, and noise anomalies at 5% ratio
- **Normal Data**: Baseline sensor readings with realistic patterns
- **Dataset**: 5000 timesteps, 4 sensors, window_size=30

### Model Layer

#### Simple Autoencoder (`SimpleAutoencoder`)
- **Algorithm**: Neural network autoencoder (NumPy implementation)
- **Architecture**: Input(120) → Hidden(64) → Encoding(16) → Hidden(64) → Output(120)
- **Input**: Flattened 30-step windows of 4 sensors (30×4=120 features)
- **Training**: Minimize reconstruction error on normal data
- **Detection**: Anomaly if reconstruction error > threshold
- **Threshold**: 95th percentile of training reconstruction errors
- **Epochs**: 80, Learning Rate: 0.002, Batch Size: 64
- **Persistence**: Pickle (.pkl)

#### SimpleLSTMPredictor (`SimpleLSTMPredictor`)
- **Algorithm**: LSTM-based next-step predictor (NumPy implementation)
- **Architecture**: LSTM(input_dim=4, hidden_dim=32) → Dense(output_dim=4)
- **Input**: Sequence of 30 timesteps of 4 sensors
- **Output**: Predicted next timestep (4 values)
- **Training**: Minimize prediction error on normal data
- **Detection**: Anomaly if prediction error > threshold
- **Threshold**: 95th percentile of training prediction errors
- **Epochs**: 30, Learning Rate: 0.005
- **Persistence**: Pickle (.pkl)

#### IsolationForestDetector (`IsolationForestDetector`)
- **Algorithm**: Isolation Forest (scikit-learn)
- **Parameters**: contamination=0.05, n_estimators=100
- **Input**: Flattened 30-step windows
- **Output**: Binary label (-1=anomaly, 1=normal) + anomaly score
- **Persistence**: Pickle (.pkl)

#### SequenceProcessor
- **Windowing**: Creates sliding windows from time-series data
- **Normalization**: Min-max normalization with fit/transform pattern
- **Parameters**: window_size=30, stride=1

### API Layer

- **Framework**: Flask (Python)
- **Serialization**: JSON request/response
- **Model Loading**: Lazy loading from `outputs/models/` on first request
- **Port**: 5018

### Dashboard Layer

- **Frontend**: HTML/CSS/JavaScript (single page)
- **Visualization**: Chart.js for sensor plots, detection results, forecasts
- **Style**: Dark-themed responsive UI

## Data Flow

### Anomaly Detection Pipeline

```
1. Generate / Input Sensor Data (n_points × 4 sensors)
   ↓
2. Create Sliding Windows (window_size=30)
   ↓
3. Flatten Windows (30×4 = 120 features)
   ↓
4. Model Detection
   ├── Autoencoder: reconstruction error > threshold → anomaly
   ├── Isolation Forest: predict → -1 = anomaly
   └── LSTM: prediction error > threshold → anomaly
   ↓
5. Results (labels, scores, metrics)
   ↓
6. Dashboard Visualization
```

### LSTM Forecasting Pipeline

```
1. Generate Normal Sensor Data
   ↓
2. Normalize
   ↓
3. Sliding Window (30 steps)
   ↓
4. LSTM predict_next(sequence) → predicted next values
   ↓
5. Compare predicted vs actual
   ↓
6. Forecast Results
```

### Model Comparison Pipeline

```
1. Generate Anomalous Dataset
   ↓
2. Create Windows + Flatten
   ↓
3. Run All Models on Same Data
   ↓
4. Compute TP, FP, FN, TN against true labels
   ↓
5. Calculate Precision, Recall, F1, Accuracy
   ↓
6. Comparison Results
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Web Framework | Flask |
| Deep Learning | Custom NumPy implementations |
| ML Library | scikit-learn (Isolation Forest) |
| Numerical | NumPy |
| Model Persistence | Pickle |
| Frontend | HTML/CSS/JS + Chart.js |

## Model Artifacts

| File | Description |
|------|-------------|
| `outputs/models/autoencoder.pkl` | Trained autoencoder model |
| `outputs/models/lstm_predictor.pkl` | Trained LSTM predictor |
| `outputs/models/isolation_forest.pkl` | Trained Isolation Forest |
| `outputs/models/metadata.json` | Training metadata and metrics |
