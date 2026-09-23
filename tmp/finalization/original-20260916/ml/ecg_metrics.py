"""ECG vital signs computation using biosppy.

Computes clinical ECG metrics from a window of ECG signal:
- BPM (beats per minute) from R-peak detection
- HRV (heart rate variability, RMSSD)
- PR interval, QRS duration, QT interval (from mean beat template)
- Rhythm classification (NSR, bradycardia, tachycardia, etc.)
- Mean beat template (average of all detected beats, aligned to R-peak)

Used by the Live Analysis tab to show real-time vital signs alongside
the anomaly detection chart.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings


def compute_ecg_metrics(signal: np.ndarray,
                         fs: int = settings.SAMPLING_RATE_HZ,
                         min_beats: int = 3) -> Dict[str, Any]:
    """Compute clinical ECG metrics from a signal window.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG signal (already preprocessed: resampled, bandpass, z-scored).
    fs : int
        Sampling frequency in Hz.
    min_beats : int
        Minimum number of R-peaks required to compute metrics. Below this,
        returns an empty metrics dict (caller should keep the previous values).

    Returns
    -------
    dict with keys:
        bpm: float | None            — beats per minute (None if too few beats)
        hrv_rmssd_ms: float | None   — RMSSD heart rate variability in ms
        pr_ms: float | None          — PR interval in ms
        qrs_ms: float | None         — QRS duration in ms
        qt_ms: float | None          — QT interval in ms
        qt_corrected_ms: float | None — QTc (Bazett's formula)
        rhythm: str                  — rhythm classification
        mean_beat: dict | None       — mean beat template + interval markers
        n_beats: int                 — number of beats detected
        rpeaks: list[int]            — sample indices of R-peaks
    """
    result: Dict[str, Any] = {
        "bpm": None,
        "hrv_rmssd_ms": None,
        "pr_ms": None,
        "qrs_ms": None,
        "qt_ms": None,
        "qt_corrected_ms": None,
        "rhythm": "Unknown",
        "mean_beat": None,
        "n_beats": 0,
        "rpeaks": [],
    }

    if len(signal) < fs * 3:  # need at least 3 seconds
        return result

    try:
        import biosppy.signals.ecg as bsp_ecg
        out = bsp_ecg.ecg(signal=signal, sampling_rate=fs, show=False)
        rpeaks = out["rpeaks"]
        templates = out["templates"]  # shape: (n_beats, samples_per_beat)
        heart_rate = out["heart_rate"]
    except Exception as e:
        print(f"[ecg_metrics] biosppy analysis failed: {e}")
        return result

    if len(rpeaks) < min_beats or templates is None or len(templates) == 0:
        return result

    result["n_beats"] = len(rpeaks)
    result["rpeaks"] = rpeaks.tolist()

    # ---------- BPM ----------
    if len(heart_rate) > 0:
        result["bpm"] = float(np.mean(heart_rate[-5:]))  # average of last 5 HR samples
    else:
        # Fallback: compute from R-R intervals
        rr_intervals_s = np.diff(rpeaks) / fs
        if len(rr_intervals_s) > 0:
            mean_rr = np.mean(rr_intervals_s)
            if mean_rr > 0:
                result["bpm"] = float(60.0 / mean_rr)

    # ---------- HRV (RMSSD) ----------
    rr_intervals_s = np.diff(rpeaks) / fs
    if len(rr_intervals_s) >= 2:
        rr_ms = rr_intervals_s * 1000.0
        successive_diffs = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        result["hrv_rmssd_ms"] = float(rmssd)

    # ---------- Mean beat + intervals ----------
    mean_beat_data = _compute_mean_beat(templates, fs)
    if mean_beat_data:
        result["mean_beat"] = mean_beat_data
        result["pr_ms"] = mean_beat_data.get("pr_ms")
        result["qrs_ms"] = mean_beat_data.get("qrs_ms")
        result["qt_ms"] = mean_beat_data.get("qt_ms")
        if result["bpm"] and result["qt_ms"]:
            # Bazett's formula: QTc = QT / sqrt(RR in seconds)
            rr_s = 60.0 / result["bpm"]
            result["qt_corrected_ms"] = float(result["qt_ms"] / np.sqrt(rr_s))

    # ---------- Rhythm classification ----------
    result["rhythm"] = _classify_rhythm(result["bpm"], result["hrv_rmssd_ms"],
                                          len(rpeaks), len(signal) / fs)

    return result


def _compute_mean_beat(templates: np.ndarray, fs: int) -> Optional[Dict[str, Any]]:
    """Compute the mean beat template and detect interval markers.

    biosppy aligns each beat so that the R-peak is at a fixed position
    within the template (typically around 25-30% into the template).
    We use the mean beat to detect:
    - P-wave onset (before QRS)
    - QRS onset and end
    - T-wave end (after QRS)
    And compute PR, QRS, QT intervals.

    The R-peak position is found as the index of the maximum amplitude
    in the mean beat.
    """
    if templates is None or len(templates) == 0:
        return None

    # Average all beat templates → mean beat
    mean_beat = np.mean(templates, axis=0)
    n = len(mean_beat)

    # Find R-peak (max amplitude point)
    r_idx = int(np.argmax(mean_beat))
    r_amp = mean_beat[r_idx]

    # ---------- QRS detection ----------
    # QRS is the sharp spike around R. Define thresholds relative to R amplitude.
    qrs_threshold = 0.3 * r_amp  # 30% of R peak

    # Search backwards from R for QRS onset (where signal first drops below threshold)
    qrs_onset = r_idx
    for i in range(r_idx, max(0, r_idx - int(0.1 * fs)), -1):
        if abs(mean_beat[i]) < qrs_threshold:
            qrs_onset = i
            break

    # Search forwards from R for QRS end
    qrs_end = r_idx
    for i in range(r_idx, min(n, r_idx + int(0.1 * fs))):
        if abs(mean_beat[i]) < qrs_threshold:
            qrs_end = i
            break

    qrs_ms = (qrs_end - qrs_onset) / fs * 1000.0

    # ---------- P-wave detection ----------
    # P-wave is a small bump before QRS. Search in the window [qrs_onset - 250ms, qrs_onset - 50ms]
    p_search_start = max(0, qrs_onset - int(0.25 * fs))
    p_search_end = max(p_search_start + 1, qrs_onset - int(0.05 * fs))
    p_region = mean_beat[p_search_start:p_search_end]

    p_onset = None
    p_peak = None
    if len(p_region) > 5:
        # P-wave is a positive bump. Find the peak in this region.
        p_peak_local = int(np.argmax(p_region))
        p_peak = p_search_start + p_peak_local
        p_amp = p_region[p_peak_local]

        # Only accept as P-wave if amplitude is > 5% of R and positive
        if p_amp > 0.05 * r_amp and p_amp > 0:
            # P-onset: search backwards from P-peak for where signal rises above baseline
            baseline = np.median(mean_beat[max(0, p_search_start - 10):p_search_start]) if p_search_start > 10 else 0
            p_onset = p_peak
            for i in range(p_peak, p_search_start, -1):
                if mean_beat[i] <= baseline + 0.02 * r_amp:
                    p_onset = i
                    break

    pr_ms = None
    if p_onset is not None:
        pr_ms = (qrs_onset - p_onset) / fs * 1000.0

    # ---------- T-wave detection ----------
    # T-wave is a broader bump after QRS. Search in [qrs_end + 100ms, qrs_end + 500ms]
    t_search_start = qrs_end + int(0.10 * fs)
    t_search_end = min(n, qrs_end + int(0.50 * fs))
    t_region = mean_beat[t_search_start:t_search_end]

    t_end = None
    if len(t_region) > 5:
        # T-wave is usually positive. Find the peak.
        t_peak_local = int(np.argmax(t_region))
        t_peak = t_search_start + t_peak_local
        t_amp = t_region[t_peak_local]

        if t_amp > 0.05 * r_amp:
            # T-end: search forwards from T-peak for where signal returns to baseline
            baseline = np.median(mean_beat[-20:]) if n > 20 else 0
            t_end = t_peak
            for i in range(t_peak, min(n, t_peak + int(0.2 * fs))):
                if mean_beat[i] <= baseline + 0.02 * r_amp:
                    t_end = i
                    break
            if t_end >= n:
                t_end = n - 1

    qt_ms = None
    if t_end is not None:
        qt_ms = (t_end - qrs_onset) / fs * 1000.0

    return {
        "samples": mean_beat.tolist(),
        "fs": fs,
        "n_samples": n,
        "r_idx": r_idx,
        "p_onset": p_onset,
        "p_peak": p_peak,
        "qrs_onset": qrs_onset,
        "qrs_end": qrs_end,
        "t_peak": t_peak if t_end else None,
        "t_end": t_end,
        "pr_ms": float(pr_ms) if pr_ms is not None else None,
        "qrs_ms": float(qrs_ms),
        "qt_ms": float(qt_ms) if qt_ms is not None else None,
    }


def _classify_rhythm(bpm: Optional[float], hrv: Optional[float],
                      n_beats: int, duration_s: float) -> str:
    """Simple rhythm classification based on BPM and HRV."""
    if bpm is None:
        return "Analyzing..."
    if n_beats < 3:
        return "Insufficient data"

    # Check for irregular rhythm (high HRV suggests AFib or frequent ectopy)
    if hrv is not None and hrv > 80:
        return "Irregular (possible AFib)"

    if bpm < 50:
        return "Bradycardia"
    elif bpm > 100:
        return "Tachycardia"
    elif 60 <= bpm <= 100:
        return "Normal Sinus Rhythm"
    elif 50 <= bpm < 60:
        return "Sinus Bradycardia (borderline)"
    else:
        return "Unknown rhythm"


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, "/home/z/my-project/backend")
    from app.ml.data import synthetic_normal_ecg, synthetic_arrhythmia_ecg

    print("=== Normal ECG (10s) ===")
    sig = synthetic_normal_ecg(10, seed=42)
    m = compute_ecg_metrics(sig)
    print(f"  BPM: {m['bpm']:.1f}" if m['bpm'] else "  BPM: None")
    print(f"  HRV (RMSSD): {m['hrv_rmssd_ms']:.1f} ms" if m['hrv_rmssd_ms'] else "  HRV: None")
    print(f"  PR: {m['pr_ms']:.1f} ms" if m['pr_ms'] else "  PR: None")
    print(f"  QRS: {m['qrs_ms']:.1f} ms" if m['qrs_ms'] else "  QRS: None")
    print(f"  QT: {m['qt_ms']:.1f} ms" if m['qt_ms'] else "  QT: None")
    print(f"  Rhythm: {m['rhythm']}")
    print(f"  Beats: {m['n_beats']}")

    print()
    print("=== Arrhythmia ECG (10s, PVCs) ===")
    sig = synthetic_arrhythmia_ecg(10, seed=42)
    m = compute_ecg_metrics(sig)
    print(f"  BPM: {m['bpm']:.1f}" if m['bpm'] else "  BPM: None")
    print(f"  HRV (RMSSD): {m['hrv_rmssd_ms']:.1f} ms" if m['hrv_rmssd_ms'] else "  HRV: None")
    print(f"  Rhythm: {m['rhythm']}")
    print(f"  Beats: {m['n_beats']}")
