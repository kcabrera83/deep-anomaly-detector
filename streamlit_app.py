import streamlit as st, joblib, numpy as np
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Anomaly Detector", page_icon="\U0001f4ca")
st.header("Anomaly Detector")

p = Path(__file__).parent / 'outputs' / 'models'
models = {'score': joblib.load(p / 'anomaly_detector.pkl')}

with st.sidebar:
    st.write('Configure parameters below')
    c = st.columns(2)
    pressure = c[0].slider('Pressure', 10, 100, 55)
    temp = c[1].slider('Temp', 20, 200, 110)
    c = st.columns(2)
    flow = c[0].slider('Flow', 10, 500, 255)
    vibration = c[1].slider('Vibration', 0, 10, 5)
    c = st.columns(2)
    rpm = c[0].slider('Rpm', 500, 5000, 2750)
    current = c[1].slider('Current', 5, 100, 52)
    run = st.button('Analyze', use_container_width=True)

if run:
    x = np.array([[pressure, temp, flow, vibration, rpm, current]])
    st.divider()
    m = models['score']
    if isinstance(m, dict):
        X = m['scaler'].transform(x)
        p = m['model'].predict(X)
        v = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else f'{p[0]:.2f}'
    else:
        v = f'{m.predict(x)[0]:.2f}'
    st.metric('Score', v)