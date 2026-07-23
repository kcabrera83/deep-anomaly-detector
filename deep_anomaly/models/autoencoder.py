"""Autoencoder model using pure NumPy for reconstruction-based anomaly detection."""

import numpy as np
import pickle
import os


class SimpleAutoencoder:
    """Fully-connected autoencoder implemented with NumPy.

    The encoder compresses input through two linear layers with tanh activations.
    The decoder mirrors the architecture. Anomalies are detected via high
    reconstruction error.
    """

    def __init__(self, input_dim, encoding_dim=8, hidden_dim=16, learning_rate=0.001):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.threshold = 0.0
        self._init_weights()

    def _init_weights(self):
        rng = np.random.RandomState(42)
        scale = lambda fan_in, fan_out: np.sqrt(2.0 / fan_in)
        self.W_enc1 = rng.randn(self.input_dim, self.hidden_dim) * scale(self.input_dim, self.hidden_dim)
        self.b_enc1 = np.zeros(self.hidden_dim)
        self.W_enc2 = rng.randn(self.hidden_dim, self.encoding_dim) * scale(self.hidden_dim, self.encoding_dim)
        self.b_enc2 = np.zeros(self.encoding_dim)
        self.W_dec1 = rng.randn(self.encoding_dim, self.hidden_dim) * scale(self.encoding_dim, self.hidden_dim)
        self.b_dec1 = np.zeros(self.hidden_dim)
        self.W_dec2 = rng.randn(self.hidden_dim, self.input_dim) * scale(self.hidden_dim, self.input_dim)
        self.b_dec2 = np.zeros(self.input_dim)

    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_deriv(self, a):
        return 1.0 - a ** 2

    def forward(self, X):
        self.z1 = X @ self.W_enc1 + self.b_enc1
        self.a1 = self._tanh(self.z1)
        self.z2 = self.a1 @ self.W_enc2 + self.b_enc2
        self.a2 = self._tanh(self.z2)
        self.z3 = self.a2 @ self.W_dec1 + self.b_dec1
        self.a3 = self._tanh(self.z3)
        self.z4 = self.a3 @ self.W_dec2 + self.b_dec2
        return self.z4

    def _backward(self, X, output):
        m = X.shape[0]
        dL = (output - X) / m

        dz4 = dL
        dW_dec2 = self.a3.T @ dz4
        db_dec2 = dz4.sum(axis=0)
        da3 = dz4 @ self.W_dec2.T

        dz3 = da3 * self._tanh_deriv(self.a3)
        dW_dec1 = self.a2.T @ dz3
        db_dec1 = dz3.sum(axis=0)
        da2 = dz3 @ self.W_dec1.T

        dz2 = da2 * self._tanh_deriv(self.a2)
        dW_enc2 = self.a1.T @ dz2
        db_enc2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W_enc2.T

        dz1 = da1 * self._tanh_deriv(self.a1)
        dW_enc1 = X.T @ dz1
        db_enc1 = dz1.sum(axis=0)

        for W, dW in [
            (self.W_enc1, dW_enc1), (self.W_enc2, dW_enc2),
            (self.W_dec1, dW_dec1), (self.W_dec2, dW_dec2),
        ]:
            np.clip(dW, -1.0, 1.0, out=dW)

        self.W_enc1 -= self.lr * dW_enc1
        self.b_enc1 -= self.lr * db_enc1
        self.W_enc2 -= self.lr * dW_enc2
        self.b_enc2 -= self.lr * db_enc2
        self.W_dec1 -= self.lr * dW_dec1
        self.b_dec1 -= self.lr * db_dec1
        self.W_dec2 -= self.lr * dW_dec2
        self.b_dec2 -= self.lr * db_dec2

    def compute_loss(self, X, output):
        return np.mean((X - output) ** 2)

    def fit(self, X, epochs=50, batch_size=64, verbose=False):
        """Train the autoencoder on normal data (2D array: n_samples x n_features)."""
        losses = []
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                batch_idx = idx[start : start + batch_size]
                X_batch = X[batch_idx]
                output = self.forward(X_batch)
                loss = np.mean((X_batch - output) ** 2)
                epoch_loss += loss
                n_batches += 1
                self._backward(X_batch, output)
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

        full_output = self.forward(X)
        errors = np.mean((X - full_output) ** 2, axis=1)
        self.threshold = float(np.percentile(errors, 95))
        return losses

    def predict(self, X):
        """Forward pass only."""
        return self.forward(X)

    def compute_reconstruction_errors(self, X):
        output = self.predict(X)
        return np.mean((X - output) ** 2, axis=1)

    def detect(self, X):
        """Returns (predictions, errors). predictions: 1=normal, -1=anomaly."""
        errors = self.compute_reconstruction_errors(X)
        predictions = np.where(errors > self.threshold, -1, 1)
        return predictions, errors

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
