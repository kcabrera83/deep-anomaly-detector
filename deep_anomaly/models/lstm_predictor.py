"""LSTM-like predictor implemented in pure NumPy for time-series forecasting."""

import numpy as np
import pickle


class SimpleLSTMPredictor:
    """A simplified LSTM-inspired model using NumPy.

    Implements a single-cell LSTM layer followed by a linear output layer.
    Used for next-step prediction; anomalies are points with high prediction error.
    """

    def __init__(self, input_dim, hidden_dim=32, output_dim=4, learning_rate=0.005):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.threshold = 0.0
        self._init_weights()

    def _init_weights(self):
        rng = np.random.RandomState(42)
        s = lambda fi, fo: np.sqrt(2.0 / fi)
        hd, id_ = self.hidden_dim, self.input_dim
        self.Wf = rng.randn(id_ + hd, hd) * s(id_ + hd, hd)
        self.bf = np.zeros(hd)
        self.Wi = rng.randn(id_ + hd, hd) * s(id_ + hd, hd)
        self.bi = np.zeros(hd)
        self.Wc = rng.randn(id_ + hd, hd) * s(id_ + hd, hd)
        self.bc = np.zeros(hd)
        self.Wo = rng.randn(id_ + hd, hd) * s(id_ + hd, hd)
        self.bo = np.zeros(hd)
        self.Wy = rng.randn(hd, self.output_dim) * s(hd, self.output_dim)
        self.by = np.zeros(self.output_dim)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward_step(self, x_t, h_prev, c_prev):
        combined = np.concatenate([x_t, h_prev])
        f = self._sigmoid(combined @ self.Wf + self.bf)
        i = self._sigmoid(combined @ self.Wi + self.bi)
        c_tilde = np.tanh(combined @ self.Wc + self.bc)
        o = self._sigmoid(combined @ self.Wo + self.bo)
        c_new = f * c_prev + i * c_tilde
        h_new = o * np.tanh(c_new)
        return h_new, c_new, (f, i, o, c_tilde, combined)

    def forward_sequence(self, X_seq):
        """X_seq: (seq_len, input_dim). Returns list of (h, c) per step."""
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        states = []
        for t in range(X_seq.shape[0]):
            h, c, _ = self.forward_step(X_seq[t], h, c)
            states.append((h.copy(), c.copy()))
        return states

    def predict_next(self, X_seq):
        """Predict next-step value from a sequence."""
        states = self.forward_sequence(X_seq)
        h_last = states[-1][0]
        return h_last @ self.Wy + self.by

    def _backward_simple(self, X_seq, y_true, states):
        """Simplified truncated BPTT - only update output layer and approximate LSTM gradients."""
        h_last = states[-1][0]
        y_pred = h_last @ self.Wy + self.by
        y_true_flat = y_true.ravel()
        d_out = y_pred - y_true_flat

        dWy = np.outer(h_last, d_out)
        dby = d_out.copy()

        self.Wy -= self.lr * np.clip(dWy, -1, 1)
        self.by -= self.lr * np.clip(dby, -1, 1)

        approx_dh = d_out @ self.Wy.T

        seq_len = X_seq.shape[0]
        for t in range(min(5, seq_len)):
            h_t = states[t][0]
            combined = np.concatenate([X_seq[t], h_t])
            lr_scale = self.lr * 0.1 / (t + 1)
            for W, b in [
                (self.Wf, self.bf), (self.Wi, self.bi),
                (self.Wc, self.bc), (self.Wo, self.bo),
            ]:
                W -= lr_scale * np.clip(np.outer(combined, approx_dh) * 0.01, -0.5, 0.5)
                b -= lr_scale * np.clip(approx_dh * 0.01, -0.5, 0.5)

    def fit(self, sequences, targets, epochs=30, verbose=False):
        """Train on sequences.

        Parameters
        ----------
        sequences : np.ndarray (n_samples, seq_len, input_dim)
        targets : np.ndarray (n_samples, output_dim)
        """
        losses = []
        for epoch in range(epochs):
            idx = np.random.permutation(len(sequences))
            epoch_loss = 0.0
            for i in idx:
                states = self.forward_sequence(sequences[i])
                h_last = states[-1][0]
                y_pred = h_last @ self.Wy + self.by
                loss = np.mean((y_pred - targets[i]) ** 2)
                epoch_loss += loss
                self._backward_simple(sequences[i], targets[i : i + 1], states)
            avg = epoch_loss / len(sequences)
            losses.append(avg)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg:.6f}")

        all_preds = []
        for i in range(len(sequences)):
            all_preds.append(self.predict_next(sequences[i]))
        all_preds = np.array(all_preds)
        errors = np.mean((all_preds - targets) ** 2, axis=1)
        self.threshold = float(np.percentile(errors, 95))
        return losses

    def detect(self, sequences, targets):
        """Predict and flag anomalies based on prediction error."""
        preds = np.array([self.predict_next(s) for s in sequences])
        errors = np.mean((preds - targets) ** 2, axis=1)
        predictions = np.where(errors > self.threshold, -1, 1)
        return predictions, errors

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
