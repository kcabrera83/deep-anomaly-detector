import numpy as np
import pickle
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """Wraps scikit-learn IsolationForest for comparison with deep learning models."""

    def __init__(self, contamination=0.05, n_estimators=100, random_state=2024):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.threshold = 0.0

    def fit(self, X):
        """Fit on 2D array (n_samples, n_features)."""
        self.model.fit(X)
        scores = -self.model.score_samples(X)
        self.threshold = float(np.percentile(scores, (1 - self.contamination) * 100))
        return self

    def detect(self, X):
        """Return (predictions, anomaly_scores). predictions: 1=normal, -1=anomaly."""
        predictions = self.model.predict(X)
        scores = -self.model.score_samples(X)
        return predictions, scores

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
