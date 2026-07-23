# Deep Anomaly Detector

AI-based anomaly detection system for oil & gas operations using autoencoders and LSTM time-series forecasting.

## Overview

This project implements multiple anomaly detection approaches for industrial sensor monitoring:

- **Autoencoder**: Reconstruction-based detection using a neural network trained on normal data. High reconstruction error indicates anomalous behavior.
- **LSTM Predictor**: Next-step forecasting model that flags time-steps with high prediction error as anomalies.
- **Isolation Forest**: Ensemble-based statistical method used as a baseline for comparison.

## Features

- Synthetic time-series sensor data generation (pressure, temperature, flow rate, vibration)
- Sliding window sequence processing with normalization
- Training pipeline with evaluation metrics (precision, recall, F1, accuracy)
- REST API with Flask for real-time detection, forecasting, and model comparison
- Web dashboard with Chart.js visualization

## Project Structure

```
deep-anomaly-detector/
  deep_anomaly/
    data_generator.py      # Sensor data generation
    models/
      autoencoder.py       # NumPy autoencoder
      lstm_predictor.py    # NumPy LSTM predictor
      isolation_forest.py  # Scikit-learn Isolation Forest
    utils/
      sequence_processor.py # Windowing, normalization
  templates/
    index.html             # Dashboard
  outputs/models/          # Trained model artifacts
  train.py                 # Training script
  app.py                   # Flask API
  test_api.py              # Test suite
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Train all models:
```bash
python train.py
```

2. Start the API server:
```bash
python app.py
```

3. Open the dashboard at `http://localhost:5018`

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web dashboard |
| `/api/health` | GET | Health check and model status |
| `/api/models` | GET | Model metadata and thresholds |
| `/api/detect` | POST | Run anomaly detection (body: `{"n_points": 500}`) |
| `/api/forecast` | POST | Run LSTM forecasting |
| `/api/compare` | POST | Compare all models on metrics |

## Training Results

After running `python train.py`, the models achieve the following on synthetic data:

- Autoencoder: reconstruction error threshold set at 95th percentile of normal data
- LSTM: prediction error threshold set at 95th percentile of normal forecasts
- Isolation Forest: contamination parameter at 5%

## Testing

```bash
python test_api.py
```

## License

Internal project - not for distribution.

## Author

Elaborado por Ing. Kelvin Cabrera
