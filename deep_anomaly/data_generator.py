import numpy as np


class SensorDataGenerator:
    """Generates realistic oil & gas sensor time-series with injected anomalies."""

    SENSOR_COLS = ["pressure", "temperature", "flow_rate", "vibration"]

    # Normal operating ranges per sensor
    RANGES = {
        "pressure": (200.0, 350.0),
        "temperature": (60.0, 120.0),
        "flow_rate": (50.0, 150.0),
        "vibration": (0.5, 3.0),
    }

    def __init__(self, seed=2024):
        self.rng = np.random.RandomState(seed)

    def _smooth(self, n, freq=0.02, phase=None):
        t = np.arange(n, dtype=np.float64)
        if phase is None:
            phase = self.rng.uniform(0, 2 * np.pi, size=3)
        signal = np.zeros(n, dtype=np.float64)
        for i, p in enumerate(phase):
            signal += np.sin(2 * np.pi * (i + 1) * freq * t + p)
        return signal / 3.0

    def generate_normal(self, n_points=5000, noise_pct=0.03):
        """Generate normal baseline sensor readings."""
        data = {}
        for name, (lo, hi) in self.RANGES.items():
            base = self._smooth(n_points, phase=self.rng.uniform(0, 2 * np.pi, 3))
            mid = (lo + hi) / 2.0
            amp = (hi - lo) / 2.0
            vals = mid + base * amp * 0.4
            noise = self.rng.normal(0, amp * noise_pct, n_points)
            data[name] = vals + noise
        return data

    def inject_anomalies(self, data, anomaly_ratio=0.05, seed=None):
        """Inject anomalies into a copy of the data.

        Returns (modified_data, anomaly_mask, anomaly_types).
        """
        rng = self.rng if seed is None else np.random.RandomState(seed)
        n = len(next(iter(data.values())))
        n_anom = max(1, int(n * anomaly_ratio))

        out = {k: v.copy() for k, v in data.items()}
        mask = np.zeros(n, dtype=bool)
        atypes = np.empty(n, dtype="U20")
        atypes[:] = "normal"

        indices = rng.choice(n, size=n_anom, replace=False)
        for idx in indices:
            kind = rng.choice(["spike", "drift", "dropout", "oscillation"])
            mask[idx] = True
            atypes[idx] = kind
            sensor = rng.choice(self.SENSOR_COLS)
            lo, hi = self.RANGES[sensor]
            amp = (hi - lo)

            if kind == "spike":
                out[sensor][idx] += rng.choice([-1, 1]) * amp * rng.uniform(2, 4)
            elif kind == "drift":
                drift_len = min(30, n - idx)
                out[sensor][idx : idx + drift_len] += np.linspace(
                    0, amp * rng.uniform(1.5, 3.0), drift_len
                )
            elif kind == "dropout":
                out[sensor][idx] = lo * rng.uniform(0.01, 0.05)
            elif kind == "oscillation":
                osc_len = min(20, n - idx)
                out[sensor][idx : idx + osc_len] += (
                    amp * 1.5 * np.sin(np.linspace(0, 6 * np.pi, osc_len))
                )

        return out, mask, atypes

    def generate_dataset(self, n_points=5000, anomaly_ratio=0.05, seed=2024):
        """Full pipeline: normal data + anomalies."""
        normal = self.generate_normal(n_points)
        anomalous, mask, atypes = self.inject_anomalies(normal, anomaly_ratio, seed=seed)
        return normal, anomalous, mask, atypes
