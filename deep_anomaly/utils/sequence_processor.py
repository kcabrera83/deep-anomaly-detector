import numpy as np


class SequenceProcessor:
    """Utilities for preparing time-series sequences for model training and inference."""

    def __init__(self, window_size=30, stride=1):
        self.window_size = window_size
        self.stride = stride
        self._stats = {}

    def normalize(self, data, method="minmax", fit=True):
        """Normalize sensor data using min-max or z-score.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples,) or dict of arrays
        method : str, 'minmax' or 'zscore'
        fit : bool, whether to learn stats from this data

        Returns
        -------
        np.ndarray normalized data (same shape as input)
        """
        is_dict = isinstance(data, dict)
        if is_dict:
            keys = list(data.keys())
            stacked = np.column_stack([data[k] for k in keys])
        else:
            stacked = data.reshape(-1, 1) if data.ndim == 1 else data

        if method == "minmax":
            if fit:
                mins = stacked.min(axis=0)
                maxs = stacked.max(axis=0)
                self._stats["method"] = "minmax"
                self._stats["mins"] = mins
                self._stats["maxs"] = maxs
            mins = self._stats["mins"]
            maxs = self._stats["maxs"]
            denom = maxs - mins
            denom[denom == 0] = 1.0
            normed = (stacked - mins) / denom
        else:
            if fit:
                mu = stacked.mean(axis=0)
                sigma = stacked.std(axis=0)
                sigma[sigma == 0] = 1.0
                self._stats["method"] = "zscore"
                self._stats["mu"] = mu
                self._stats["sigma"] = sigma
            normed = (stacked - self._stats["mu"]) / self._stats["sigma"]

        if is_dict:
            return {k: normed[:, i] for i, k in enumerate(keys)}
        return normed if normed.shape[1] > 1 else normed.ravel()

    def denormalize(self, data, method=None):
        """Reverse the normalization."""
        if method is None:
            method = self._stats.get("method", "minmax")

        is_dict = isinstance(data, dict)
        if is_dict:
            keys = list(data.keys())
            stacked = np.column_stack([data[k] for k in keys])
        else:
            stacked = data.reshape(-1, 1) if data.ndim == 1 else data

        if method == "minmax":
            mins = self._stats["mins"]
            maxs = self._stats["maxs"]
            denom = maxs - mins
            denom[denom == 0] = 1.0
            orig = stacked * denom + mins
        else:
            orig = stacked * self._stats["sigma"] + self._stats["mu"]

        if is_dict:
            return {k: orig[:, i] for i, k in enumerate(keys)}
        return orig if orig.shape[1] > 1 else orig.ravel()

    def create_windows(self, data, window_size=None):
        """Create sliding windows from a 2D array (n_samples, n_features).

        Returns np.ndarray of shape (n_windows, window_size, n_features).
        """
        ws = window_size or self.window_size
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n = data.shape[0]
        n_features = data.shape[1]
        n_windows = max(0, (n - ws) // self.stride + 1)
        if n_windows == 0:
            return np.empty((0, ws, n_features))
        windows = np.empty((n_windows, ws, n_features), dtype=data.dtype)
        for i in range(n_windows):
            start = i * self.stride
            windows[i] = data[start : start + ws]
        return windows

    def pad_sequence(self, seq, max_len=None, pad_value=0.0):
        """Pad a 1D or 2D sequence to a fixed length."""
        if max_len is None:
            return seq
        if seq.ndim == 1:
            pad_len = max_len - len(seq)
            if pad_len <= 0:
                return seq[:max_len]
            return np.concatenate([seq, np.full(pad_len, pad_value)])
        else:
            pad_len = max_len - seq.shape[0]
            if pad_len <= 0:
                return seq[:max_len]
            pad_arr = np.full((pad_len, seq.shape[1]), pad_value, dtype=seq.dtype)
            return np.concatenate([seq, pad_arr], axis=0)

    def prepare_train_data(self, data_dict, window_size=None):
        """Stack dict of sensor arrays into windows ready for model training."""
        arr = np.column_stack([data_dict[k] for k in sorted(data_dict.keys())])
        arr = self.normalize(arr, fit=True)
        return self.create_windows(arr, window_size)
