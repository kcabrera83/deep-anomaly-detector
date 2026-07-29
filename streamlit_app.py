
import streamlit as st
import numpy as np
import joblib, os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deep Anomaly Detector", layout="wide")
st.title(":microscope: Deep Anomaly Detector")
st.caption("Unsupervised anomaly detection on time-series sensor data")

models = {}
for f in os.listdir("outputs/models"):
    if f.endswith(".pkl"):
        models[f.replace(".pkl", "")] = joblib.load(os.path.join("outputs/models", f))

view = st.sidebar.radio("View", ["Inference", "Model Analysis", "Data Explorer"])

if view == "Inference":
    sel = st.selectbox("Model", list(models.keys()))
    m = models[sel]
    feats = m.get("feature_names", [f"x{i}" for i in range(4)])
    cols = st.columns(2)
    inp = [cols[i%2].number_input(f, value=0.0) for i, f in enumerate(feats)]
    if st.button("Infer"):
        X = np.array(inp).reshape(1, -1)
        if m.get("scaler"):
            X = m["scaler"].transform(X)
        pred = m["model"].predict(X)[0]
        st.metric("Result", f"{pred:.3f}")

elif view == "Model Analysis":
    sel = st.selectbox("Model", list(models.keys()))
    m = models[sel]
    st.json({k: str(type(v)) for k, v in m.items() if k != "model"})
    st.write("Model type:", type(m["model"]).__name__)
    if hasattr(m["model"], "feature_importances_"):
        fig, ax = plt.subplots()
        ax.bar(range(len(m["model"].feature_importances_)), m["model"].feature_importances_)
        st.pyplot(fig)

elif view == "Data Explorer":
    n = st.slider("Samples", 10, 1000, 100)
    X = np.random.randn(n, 4)
    fig, ax = plt.subplots()
    ax.plot(X)
    st.pyplot(fig)
