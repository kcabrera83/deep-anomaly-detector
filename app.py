
import os, json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Deep Anomaly Detector")

class AnalysisRequest(BaseModel):
    features: dict
    pipeline: str = "default"

class AnalysisResponse(BaseModel):
    prediction: float
    pipeline: str
    model_type: str

PIPELINES = {}
for f in os.listdir("outputs/models"):
    if f.endswith(".pkl"):
        PIPELINES[f.replace(".pkl", "")] = joblib.load(os.path.join("outputs/models", f))

@app.get("/")
def root():
    return dict(
        service="Deep Anomaly Detector",
        version="2.0",
        pipelines=list(PIPELINES.keys()),
        endpoints=["/analyze/{pipeline}"]
    )

@app.post("/analyze/{pipeline}")
def analyze(pipeline: str, req: AnalysisRequest):
    pipe = PIPELINES.get(pipeline)
    if not pipe:
        raise HTTPException(404, f"Pipeline {pipeline} unavailable")
    feats = pipe.get("feature_names", list(req.features.keys()))
    X = np.array([req.features.get(f, 0) for f in feats]).reshape(1, -1)
    scaler = pipe.get("scaler")
    if scaler:
        X = scaler.transform(X)
    pred = pipe["model"].predict(X)[0]
    return AnalysisResponse(prediction=float(pred), pipeline=pipeline, model_type=type(pipe["model"]).__name__)
