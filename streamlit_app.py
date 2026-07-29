import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Deep Anomaly Detector", layout="wide")
st.title("Deep Anomaly Detector")
st.markdown("Detect anomalies in industrial sensor data.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'anomaly': joblib.load(d / 'anomaly_detector.pkl')}

st.sidebar.header("Input Parameters")
pressure = st.sidebar.slider('Pressure', 10, 100, 55)
temperature = st.sidebar.slider('Temperature', 20, 200, 110)
flow_rate = st.sidebar.slider('Flow Rate', 10, 500, 255)
vibration = st.sidebar.slider('Vibration', 0, 10, 5)
rpm = st.sidebar.slider('Rpm', 500, 5000, 2750)
current = st.sidebar.slider('Current', 5, 100, 52)

if st.sidebar.button("Run"):
    try:
        x = np.array([[pressure, temperature, flow_rate, vibration, rpm, current]])
        cols = st.columns(1)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))