import streamlit as st
import pickle
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Deep Anomaly Detector", layout="wide")
st.title("Deep Anomaly Detector")
st.markdown("Detect anomalies in industrial sensor data using ensemble methods.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: pickle.load(open(d / v, "rb")) for k, v in [("detector", "anomaly_detector.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
pressure_bar = st.sidebar.slider("Pressure Bar", 10, 100, 55)
temperature_c = st.sidebar.slider("Temperature C", 20, 200, 110)
flow_rate_m3h = st.sidebar.slider("Flow Rate M3H", 10, 500, 255)
vibration_mm_s = st.sidebar.slider("Vibration Mm S", 0, 10, 5)
rpm = st.sidebar.slider("Rpm", 500, 5000, 2750)
current_a = st.sidebar.slider("Current A", 5, 100, 52)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[pressure_bar, temperature_c, flow_rate_m3h, vibration_mm_s, rpm, current_a]])
        m = models["detector"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Detector", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")