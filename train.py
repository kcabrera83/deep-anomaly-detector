"""Training script for all anomaly detection models."""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from deep_anomaly.data_generator import SensorDataGenerator
from deep_anomaly.utils.sequence_processor import SequenceProcessor
from deep_anomaly.models.autoencoder import SimpleAutoencoder
from deep_anomaly.models.lstm_predictor import SimpleLSTMPredictor
from deep_anomaly.models.isolation_forest import IsolationForestDetector

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Deep Anomaly Detector - Training Pipeline")
    print("  Oil & Gas Sensor Data")
    print("=" * 60)

    # --- Data Generation ---
    print("\n[1/6] Generating synthetic sensor data...")
    gen = SensorDataGenerator(seed=42)
    normal, anomalous, anomaly_mask, anomaly_types = gen.generate_dataset(
        n_points=5000, anomaly_ratio=0.05
    )
    total_anomalies = int(anomaly_mask.sum())
    print(f"  Generated 5000 timesteps, 4 sensors")
    print(f"  Injected anomalies: {total_anomalies} ({total_anomalies/5000*100:.1f}%)")
    type_counts = {}
    for t in anomaly_types[anomaly_types != "normal"]:
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Anomaly types: {type_counts}")

    # --- Preprocessing ---
    print("\n[2/6] Preprocessing sequences...")
    processor = SequenceProcessor(window_size=30, stride=1)
    windows = processor.prepare_train_data(anomalous, window_size=30)
    print(f"  Created {windows.shape[0]} windows of shape {windows.shape[1:]}")
    flat = windows.reshape(windows.shape[0], -1)
    n_features = windows.shape[2]

    # Split normal windows for training AE
    normal_data = processor.normalize(gen.generate_normal(5000), fit=True)
    normal_arr = np.column_stack([normal_data[k] for k in sorted(normal_data.keys())])
    normal_windows = processor.create_windows(normal_arr)
    normal_flat = normal_windows.reshape(normal_windows.shape[0], -1)

    # --- Train Autoencoder ---
    print("\n[3/6] Training Autoencoder...")
    ae = SimpleAutoencoder(input_dim=n_features * 30, encoding_dim=16, hidden_dim=64, learning_rate=0.002)
    ae_losses = ae.fit(normal_flat, epochs=80, batch_size=64, verbose=True)
    ae_path = os.path.join(OUTPUT_DIR, "autoencoder.pkl")
    ae.save(ae_path)
    print(f"  Autoencoder saved to {ae_path}")
    print(f"  Final loss: {ae_losses[-1]:.6f}, Threshold: {ae.threshold:.6f}")

    # --- Train LSTM ---
    print("\n[4/6] Training LSTM Predictor...")
    seq_len = 30
    lstm = SimpleLSTMPredictor(input_dim=n_features, hidden_dim=32, output_dim=n_features, learning_rate=0.005)
    normal_data_dict = processor.normalize(gen.generate_normal(5000), fit=False)
    normal_data = np.column_stack([normal_data_dict[k] for k in sorted(normal_data_dict.keys())])
    train_seqs = normal_data[:len(normal_data) - 1]
    train_targets = normal_data[1:]
    n_train = len(train_seqs) - seq_len
    lstm_seqs = np.array([train_seqs[i : i + seq_len] for i in range(0, n_train, seq_len)])
    lstm_targets = np.array([train_targets[i + seq_len - 1] for i in range(0, n_train, seq_len)])
    lstm_losses = lstm.fit(lstm_seqs, lstm_targets, epochs=30, verbose=True)
    lstm_path = os.path.join(OUTPUT_DIR, "lstm_predictor.pkl")
    lstm.save(lstm_path)
    print(f"  LSTM saved to {lstm_path}")
    print(f"  Final loss: {lstm_losses[-1]:.6f}, Threshold: {lstm.threshold:.6f}")

    # --- Train Isolation Forest ---
    print("\n[5/6] Training Isolation Forest...")
    iforest = IsolationForestDetector(contamination=0.05, n_estimators=100)
    iforest.fit(normal_flat)
    if_path = os.path.join(OUTPUT_DIR, "isolation_forest.pkl")
    iforest.save(if_path)
    print(f"  Isolation Forest saved to {if_path}")

    # --- Evaluation ---
    print("\n[6/6] Evaluating all models on test data...")
    ae_preds, ae_errors = ae.detect(flat)
    if_preds, if_errors = iforest.detect(flat)

    true_labels = np.where(anomaly_mask[:len(ae_preds)], -1, 1)

    def metrics(preds, name):
        tp = int(np.sum((preds == -1) & (true_labels == -1)))
        fp = int(np.sum((preds == -1) & (true_labels == 1)))
        fn = int(np.sum((preds == 1) & (true_labels == -1)))
        tn = int(np.sum((preds == 1) & (true_labels == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        acc = (tp + tn) / len(preds)
        print(f"\n  [{name}]")
        print(f"    Detected: {int(np.sum(preds == -1))} anomalies")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall:    {recall:.3f}")
        print(f"    F1-Score:  {f1:.3f}")
        print(f"    Accuracy:  {acc:.3f}")
        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc, "detected": int(np.sum(preds == -1))}

    results = {}
    results["autoencoder"] = metrics(ae_preds, "Autoencoder")
    results["isolation_forest"] = metrics(if_preds, "Isolation Forest")

    # Save metadata
    meta = {
        "n_samples": 5000,
        "n_features": n_features,
        "window_size": 30,
        "anomaly_ratio": 0.05,
        "total_anomalies": total_anomalies,
        "thresholds": {"autoencoder": ae.threshold, "lstm": lstm.threshold},
        "metrics": results,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("  Training complete. All models saved to outputs/models/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
