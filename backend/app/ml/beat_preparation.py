"""Shared, bounded-context inputs for the corrected ECG hierarchy.

Input is a resampled (128 Hz), unfiltered waveform. Filtering and scaling use
only the four-second model window; no recording-wide statistics are fitted.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

PREPROCESSING_VERSION = "beat-local-bandpass-zscore-v1"
FS = 128
WINDOW = 512
R_BEFORE = 200
_SOS = butter(4, [0.5, 50.0], btype="bandpass", fs=FS, output="sos")


def prepare_beat(window: np.ndarray) -> np.ndarray:
    values = np.asarray(window, dtype=np.float64)
    if values.shape != (WINDOW,) or not np.isfinite(values).all():
        raise ValueError("A beat must contain 512 finite samples")
    if np.ptp(values) < 1e-8:
        raise ValueError("Flat ECG window cannot be classified")
    filtered = sosfiltfilt(_SOS, values)
    scale = float(filtered.std())
    if scale < 1e-8:
        raise ValueError("ECG window has no usable variation")
    return ((filtered - filtered.mean()) / (scale + 1e-8)).astype(np.float32)


def base_rr_at(peaks: np.ndarray, index: int, fs: float = FS) -> np.ndarray:
    """Seven RR features, requiring a preceding and following observed beat."""
    peaks = np.asarray(peaks)
    if index < 1 or index >= len(peaks) - 1:
        raise ValueError("Previous and next R peaks are required")
    history = np.diff(peaks[max(0, index - 10):index + 1]) / float(fs)
    previous = float(history[-1])
    following = float(peaks[index + 1] - peaks[index]) / float(fs)
    if np.any(history <= 0) or following <= 0:
        raise ValueError("R peaks must be strictly increasing")
    mean = float(history.mean())
    return np.asarray([previous, following, mean, previous / mean,
                       following / mean, float(history.std()) / mean,
                       60.0 / previous], dtype=np.float32)
