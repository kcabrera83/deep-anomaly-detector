# Deployment Guide - Deep Anomaly Detector

## Docker Deployment

### Build the Image

```bash
cd deep-anomaly-detector
docker build -t deep-anomaly-detector .
```

### Run the Container

```bash
docker run -p 5018:5018 deep-anomaly-detector
```

### With Model Training

```bash
docker run -p 5018:5018 deep-anomaly-detector bash -c "python train.py && python app.py"
```

## Docker Compose

```yaml
version: '3.8'
services:
  deep-anomaly:
    build: .
    ports:
      - "5018:5018"
    volumes:
      - ./outputs:/app/outputs
    environment:
      - FLASK_ENV=production
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PYTHONUNBUFFERED | Disable Python output buffering | 1 |
| PORT | Server port (hardcoded in app.py) | 5018 |

## Manual Deployment

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: Flask, NumPy, scikit-learn

### Train Models

```bash
python train.py
```

### Run with Gunicorn (Production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5018 app:app
```

### Run with Flask Development Server

```bash
python app.py
```

## Production Considerations

- Use **gunicorn** with multiple workers for production deployments
- `debug=False` is already set in `app.py`
- Configure proper logging for request/error tracking
- Place behind a reverse proxy (nginx/Apache) for SSL termination
- Models are lazy-loaded on first request - pre-train with `train.py`
- Autoencoder and LSTM use NumPy implementations (no GPU required)
- Memory usage scales with number of concurrent requests

## Health Check

```bash
curl http://localhost:5018/api/health
```

Expected response:
```json
{"status": "healthy", "models_loaded": ["autoencoder", "isolation_forest", "lstm"], "version": "1.0.0"}
```

## Ports

| Service | Port |
|---------|------|
| Deep Anomaly Detector | 5018 |
