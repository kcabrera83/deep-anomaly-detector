"""Training script for anomaly detection models using TensorFlow/Keras Autoencoder, PyTorch LSTM, and tsfresh."""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from deep_anomaly.data_generator import SensorDataGenerator
from deep_anomaly.utils.sequence_processor import SequenceProcessor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")


class AutoEncoder(keras.Model):
    """Deep Autoencoder for anomaly detection"""
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(latent_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTMDetector(nn.Module):
    """LSTM for sequence anomaly detection"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class ModelWrapper:
    """Wrapper for autoencoder to match API interface"""
    def __init__(self, keras_model, threshold):
        self.keras_model = keras_model
        self.threshold = threshold

    def detect(self, flat_data):
        tensor = tf.convert_to_tensor(flat_data, dtype=tf.float32)
        reconstructed = self.keras_model(tensor).numpy()
        errors = np.mean(np.abs(flat_data - reconstructed), axis=1)
        preds = np.where(errors > self.threshold, -1, 1)
        return preds, errors

    def save(self, path):
        self.keras_model.save_weights(path + "_weights.weights.h5")
        with open(path + "_meta.json", "w") as f:
            json.dump({"threshold": float(self.threshold)}, f)

    @classmethod
    def load(cls, path, input_dim):
        model = AutoEncoder(input_dim=input_dim)
        dummy = tf.zeros((1, input_dim))
        model(dummy)
        model.load_weights(path + "_weights.weights.h5")
        with open(path + "_meta.json") as f:
            meta = json.load(f)
        return cls(model, meta["threshold"])


class LSTMWrapper:
    """Wrapper for PyTorch LSTM to match API interface"""
    def __init__(self, torch_model, threshold):
        self.torch_model = torch_model
        self.threshold = threshold
        self.torch_model.eval()

    def predict_next(self, seq):
        with torch.no_grad():
            x = torch.FloatTensor(seq).unsqueeze(0)
            pred = self.torch_model(x)
        return pred.squeeze(0).numpy()

    def detect(self, seqs, targets):
        errors = []
        with torch.no_grad():
            for seq, tgt in zip(seqs, targets):
                x = torch.FloatTensor(seq).unsqueeze(0)
                pred = self.torch_model(x).squeeze(0).numpy()
                err = float(np.mean(np.abs(pred - tgt)))
                errors.append(err)
        errors = np.array(errors)
        preds = np.where(errors > self.threshold, -1, 1)
        return preds, errors

    def save(self, path):
        torch.save(self.torch_model.state_dict(), path + "_state.pth")
        with open(path + "_meta.json", "w") as f:
            json.dump({"threshold": float(self.threshold)}, f)

    @classmethod
    def load(cls, path, input_size):
        model = LSTMDetector(input_size=input_size)
        model.load_state_dict(torch.load(path + "_state.pth", weights_only=True))
        with open(path + "_meta.json") as f:
            meta = json.load(f)
        return cls(model, meta["threshold"])


def extract_tsfresh_features(df):
    """Extract time series features using tsfresh"""
    settings = MinimalFCParameters()
    features = extract_features(df, column_id="id", column_sort="time",
                                default_fc_parameters=settings, disable_progressbar=True)
    return features


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Deep Anomaly Detector - Training Pipeline")
    print("  TensorFlow/Keras Autoencoder + PyTorch LSTM + tsfresh")
    print("=" * 60)

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

    print("\n[2/6] Preprocessing sequences...")
    processor = SequenceProcessor(window_size=30, stride=1)
    windows = processor.prepare_train_data(anomalous, window_size=30)
    print(f"  Created {windows.shape[0]} windows of shape {windows.shape[1:]}")
    flat = windows.reshape(windows.shape[0], -1)
    n_features = windows.shape[2]

    normal_data = processor.normalize(gen.generate_normal(5000), fit=True)
    normal_arr = np.column_stack([normal_data[k] for k in sorted(normal_data.keys())])
    normal_windows = processor.create_windows(normal_arr)
    normal_flat = normal_windows.reshape(normal_windows.shape[0], -1)

    print("\n[3/6] Training TensorFlow/Keras Autoencoder...")
    input_dim = n_features * 30
    ae_model = AutoEncoder(input_dim=input_dim, latent_dim=16)
    ae_model.compile(optimizer='adam', loss='mse')
    ae_model.fit(normal_flat, normal_flat, epochs=80, batch_size=64, validation_split=0.2, verbose=1)
    reconstructed = ae_model(tf.convert_to_tensor(normal_flat, dtype=tf.float32)).numpy()
    ae_errors = np.mean(np.abs(normal_flat - reconstructed), axis=1)
    ae_threshold = float(np.percentile(ae_errors, 95))
    ae_wrapper = ModelWrapper(ae_model, ae_threshold)
    ae_path = os.path.join(OUTPUT_DIR, "autoencoder.pkl")
    ae_wrapper.save(ae_path)
    print(f"  Autoencoder saved to {ae_path}")
    print(f"  Threshold: {ae_threshold:.6f}")

    print("\n[4/6] Training PyTorch LSTM Predictor...")
    seq_len = 30
    lstm_model = LSTMDetector(input_size=n_features, hidden_size=64, num_layers=2)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    normal_data_dict = processor.normalize(gen.generate_normal(5000), fit=False)
    normal_data_np = np.column_stack([normal_data_dict[k] for k in sorted(normal_data_dict.keys())])
    train_seqs = normal_data_np[:len(normal_data_np) - 1]
    train_targets = normal_data_np[1:]
    n_train = len(train_seqs) - seq_len
    lstm_seqs = np.array([train_seqs[i: i + seq_len] for i in range(0, n_train, seq_len)])
    lstm_targets = np.array([train_targets[i + seq_len - 1] for i in range(0, n_train, seq_len)])
    lstm_model.train()
    for epoch in range(30):
        epoch_loss = 0.0
        for batch_x, batch_y in zip(lstm_seqs, lstm_targets):
            x = torch.FloatTensor(batch_x).unsqueeze(0)
            y = torch.FloatTensor(batch_y).unsqueeze(0)
            output = lstm_model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30, Loss: {epoch_loss/len(lstm_seqs):.6f}")
    lstm_model.eval()
    lstm_errors = []
    with torch.no_grad():
        for seq, tgt in zip(lstm_seqs[:200], lstm_targets[:200]):
            x = torch.FloatTensor(seq).unsqueeze(0)
            pred = lstm_model(x).squeeze(0).numpy()
            lstm_errors.append(float(np.mean(np.abs(pred - tgt))))
    lstm_threshold = float(np.percentile(lstm_errors, 95))
    lstm_wrapper = LSTMWrapper(lstm_model, lstm_threshold)
    lstm_path = os.path.join(OUTPUT_DIR, "lstm_predictor.pkl")
    lstm_wrapper.save(lstm_path)
    print(f"  LSTM saved to {lstm_path}")
    print(f"  Threshold: {lstm_threshold:.6f}")

    print("\n[5/6] Extracting tsfresh features...")
    ts_df = gen.generate_normal(500)
    ts_features = extract_tsfresh_features(ts_df)
    print(f"  Extracted {ts_features.shape[1]} features from tsfresh")

    print("\n[6/6] Evaluating all models on test data...")
    ae_preds, ae_errors = ae_wrapper.detect(flat)

    seqs_for_lstm = np.array([normal_arr[i: i + seq_len] for i in range(0, len(normal_arr) - seq_len)])
    tgts_for_lstm = np.array([normal_arr[i + seq_len] for i in range(0, len(normal_arr) - seq_len)])
    lstm_preds, lstm_errors = lstm_wrapper.detect(seqs_for_lstm[:len(tgts_for_lstm)], tgts_for_lstm)

    true_labels = np.where(anomaly_mask[:len(ae_preds)], -1, 1)

    def metrics(preds, name):
        tp = int(np.sum((preds == -1) & (true_labels == -1)))
        fp = int(np.sum((preds == -1) & (true_labels == 1)))
        fn = int(np.sum((preds == 1) & (true_labels == -1)))
        tn = int(np.sum((preds == 1) & (true_labels == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        acc = (tp + tn) / max(len(preds), 1)
        print(f"\n  [{name}]")
        print(f"    Detected: {int(np.sum(preds == -1))} anomalies")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall:    {recall:.3f}")
        print(f"    F1-Score:  {f1:.3f}")
        print(f"    Accuracy:  {acc:.3f}")
        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc, "detected": int(np.sum(preds == -1))}

    results = {}
    results["autoencoder"] = metrics(ae_preds, "Autoencoder (TF/Keras)")

    meta = {
        "n_samples": 5000,
        "n_features": n_features,
        "window_size": 30,
        "anomaly_ratio": 0.05,
        "total_anomalies": total_anomalies,
        "thresholds": {"autoencoder": ae_threshold, "lstm": lstm_threshold},
        "tsfresh_features": int(ts_features.shape[1]),
        "metrics": results,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("  Training complete. Models saved to outputs/models/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
