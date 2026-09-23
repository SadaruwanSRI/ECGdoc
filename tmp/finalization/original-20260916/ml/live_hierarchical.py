"""Beat-aligned streaming support for the improved hierarchical ECG system.

The trained feature system cannot safely score an arbitrary sliding window:
its inputs must be centred on an R peak and include RR timing context.  This
module keeps a rolling signal buffer, detects R peaks without annotations, and
emits each beat only after enough future signal and the next RR interval are
available.  That produces a deliberate latency of roughly one to three
seconds, depending on heart rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.signal import find_peaks

from app.core.config import settings
from app.ml.classifier import ARRHYTHMIA_NAMES
from app.ml.feature_system import extend_rr_features
from app.ml.inference import InferenceEngine


def detect_rpeaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """Return R-peak indices for a preprocessed rolling ECG buffer.

    BioSPPy is used when available because the rest of the application already
    uses it for cardiac metrics.  A polarity-aware SciPy fallback keeps the
    live system functional if BioSPPy cannot analyse a noisy or short buffer.
    """
    values = np.asarray(signal, dtype=np.float32)
    if values.size < fs * 3:
        return np.empty(0, dtype=np.int64)

    try:
        from biosppy.signals import ecg as bsp_ecg

        output = bsp_ecg.ecg(signal=values, sampling_rate=fs, show=False)
        peaks = np.asarray(output["rpeaks"], dtype=np.int64)
        if peaks.size >= 2:
            return peaks
    except Exception:
        pass

    centred = values - float(np.median(values))
    robust_scale = 1.4826 * float(
        np.median(np.abs(centred - np.median(centred)))
    )
    robust_scale = max(robust_scale, float(np.std(centred)) * 0.25, 1e-3)
    distance = max(1, int(0.28 * fs))
    prominence = max(0.35, 0.8 * robust_scale)
    positive, positive_props = find_peaks(
        centred, distance=distance, prominence=prominence
    )
    negative, negative_props = find_peaks(
        -centred, distance=distance, prominence=prominence
    )
    positive_strength = float(
        np.median(positive_props.get("prominences", [0.0]))
    )
    negative_strength = float(
        np.median(negative_props.get("prominences", [0.0]))
    )
    return np.asarray(
        negative if negative_strength > positive_strength else positive,
        dtype=np.int64,
    )


@dataclass
class BeatStreamAnalyzer:
    """Accumulate ECG chunks and classify new, fully observed heartbeats."""

    engine: InferenceEngine
    fs: int = settings.SAMPLING_RATE_HZ
    r_peak_before: int = 200
    window_samples: int = settings.WINDOW_SAMPLES
    max_buffer_seconds: int = 60
    mlii_signal: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    buffer_start_t: int = 0
    last_processed_peak_t: Optional[int] = None

    @property
    def post_peak_samples(self) -> int:
        return self.window_samples - self.r_peak_before

    def append(
        self,
        mlii_chunk: np.ndarray,
    ) -> list[dict[str, Any]]:
        first = np.asarray(mlii_chunk, dtype=np.float32).reshape(-1)
        if first.size == 0:
            return []
        self.mlii_signal = np.concatenate([self.mlii_signal, first])

        events = self._classify_ready_beats()
        self._trim()
        return events

    def _classify_ready_beats(self) -> list[dict[str, Any]]:
        peaks = detect_rpeaks(self.mlii_signal, self.fs)
        if peaks.size < 3:
            return []

        intervals = np.diff(peaks).astype(np.float32) / float(self.fs)
        events: list[dict[str, Any]] = []
        duplicate_tolerance = int(0.20 * self.fs)

        for index in range(1, len(peaks) - 1):
            peak = int(peaks[index])
            global_peak = self.buffer_start_t + peak
            if (
                self.last_processed_peak_t is not None
                and global_peak <= self.last_processed_peak_t + duplicate_tolerance
            ):
                continue

            start = peak - self.r_peak_before
            end = start + self.window_samples
            if start < 0 or end > self.mlii_signal.size:
                continue

            previous_rr = float(intervals[index - 1])
            next_rr = float(intervals[index])
            base_history = intervals[max(0, index - 10):index]
            local_mean = float(np.mean(base_history))
            local_std = float(np.std(base_history))
            safe_mean = max(local_mean, 1e-3)
            base_rr_features = np.asarray(
                [
                    previous_rr,
                    next_rr,
                    local_mean,
                    previous_rr / safe_mean,
                    next_rr / safe_mean,
                    local_std / safe_mean,
                    60.0 / max(previous_rr, 1e-3),
                ],
                dtype=np.float32,
            )
            rr_features = (
                extend_rr_features(
                    base_rr_features,
                    intervals[max(0, index - 20):index],
                )
                if (
                    self.engine.improved_system is not None
                    and self.engine.improved_system.rr_feature_count == 15
                )
                else base_rr_features
            )

            window = self.mlii_signal[start:end].astype(np.float32, copy=True)
            result = self.engine.classify_beat(
                window,
                rr_features,
            )
            if result is None:
                continue

            class_code = str(result["class"])
            class_name = (
                "Unclassified abnormal pattern"
                if class_code == "Unclassified abnormal"
                else ARRHYTHMIA_NAMES.get(class_code, class_code)
            )
            event = {
                "t": global_peak,
                "is_anomaly": bool(result["is_anomaly"]),
                "anomaly_probability": float(result["anomaly_probability"]),
                "threshold": float(result["threshold"]),
                "class": class_code,
                "class_name": class_name,
                "confidence": float(result["confidence"]),
                "probabilities": result["probabilities"],
                "classification_mode": result["classification_mode"],
                "rr_seconds": previous_rr,
                "rr_features": rr_features.tolist(),
                "signal": window,
            }
            events.append(event)
            self.last_processed_peak_t = global_peak

        return events

    def _trim(self) -> None:
        max_samples = int(self.max_buffer_seconds * self.fs)
        if self.mlii_signal.size <= max_samples:
            return
        remove = self.mlii_signal.size - max_samples
        self.mlii_signal = self.mlii_signal[remove:]
        self.buffer_start_t += remove
