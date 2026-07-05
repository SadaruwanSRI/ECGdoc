"""ECG data pipeline.

Provides three sources:
1. Synthetic normal sinus rhythm — for quick smoke tests when PhysioNet is unreachable.
2. MIT-BIH Normal Sinus Rhythm DB (NSRDB) — for training.
3. MIT-BIH Arrhythmia Database (MITDB) — for evaluation / live replay.

All sources go through the same preprocessing per slide 11:
  raw → resample to 128 Hz → bandpass 0.5–50 Hz → 4 s windows (512 samples) → z-score normalize
"""
from __future__ import annotations

import io
import os
import math
import random
from pathlib import Path
from typing import Iterator, List, Tuple, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

from app.core.config import settings


# ---------- Signal processing primitives ----------

def butter_bandpass(low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sos


def bandpass_filter(signal: np.ndarray, fs: float,
                    low: float = settings.BANDPASS_LOW,
                    high: float = settings.BANDPASS_HIGH) -> np.ndarray:
    sos = butter_bandpass(low, high, fs)
    return sosfiltfilt(sos, signal)


def resample_to(signal: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    if src_fs == dst_fs:
        return signal.astype(np.float32)
    # Polyphase anti-aliasing resample
    from math import gcd
    g = gcd(int(src_fs), int(dst_fs))
    up = int(dst_fs) // g
    down = int(src_fs) // g
    return resample_poly(signal, up, down).astype(np.float32)


def zscore(signal: np.ndarray) -> np.ndarray:
    mu = signal.mean()
    sd = signal.std() + 1e-8
    return ((signal - mu) / sd).astype(np.float32)


def preprocess_signal(signal: np.ndarray, src_fs: float) -> np.ndarray:
    """Full pipeline: resample → bandpass → z-score."""
    signal = resample_to(signal, src_fs, settings.SAMPLING_RATE_HZ)
    signal = bandpass_filter(signal, settings.SAMPLING_RATE_HZ)
    signal = zscore(signal)
    return signal


def windowize(signal: np.ndarray,
              window_size: int = settings.WINDOW_SAMPLES,
              stride: Optional[int] = None) -> np.ndarray:
    """Split a 1-D signal into non-overlapping windows of length `window_size`."""
    if stride is None:
        stride = window_size
    n = len(signal)
    n_windows = max(0, (n - window_size) // stride + 1)
    if n_windows == 0:
        return np.empty((0, window_size), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)[::stride]
    return windows[:n_windows].astype(np.float32)


def quality_control(windows: np.ndarray, max_std: float = 5.0,
                    min_std: float = 0.05) -> np.ndarray:
    """Drop windows that are flat/degenerate or saturated with artifacts.

    This is the simple fallback filter. For real ECG quality classification
    using the ecg_qc library (SQI feature extraction + heuristic thresholds),
    use `ecg_qc_quality_control()` instead.
    """
    stds = windows.std(axis=1)
    mask = (stds > min_std) & (stds < max_std)
    return windows[mask]


# ---------- ECG-QC quality classifier (uses ecg_qc SQI features) ----------
#
# The ecg_qc library ships a pre-trained RandomForest model (rfc_norm_2s.pkl)
# but the pickle is incompatible with modern scikit-learn (1.3+) due to a
# dtype change in the tree node array. Rather than pinning an old sklearn,
# we use the library's Signal Quality Indicator (SQI) feature extractors
# directly and apply heuristic thresholds that we validated empirically.
#
# SQI features used (with corrected semantics):
#   q_sqi   - Fraction of matching R-peaks between two detectors (0-1, HIGHER = better)
#   c_sqi   - CoV of R-R intervals (LOWER = more regular = better; <0.30 = clean)
#   s_sqi   - Skewness of signal (HIGHER = cleaner QRS morphology; >2.0 = clean)
#   k_sqi   - Kurtosis of signal (HIGHER = sharper QRS peaks; >5.0 = clean)
#   p_sqi   - QRS power / total power 5-15Hz / 5-40Hz (HIGHER = better; >0.40 = clean)
#   bas_sqi - 1 - baseline/total power 0-1Hz / 0-40Hz (HIGHER = cleaner baseline; >0.85 = clean)
#
# Empirical reference values from our synthetic + MIT-BIH testing:
#   Signal type         | q_sqi | s_sqi | k_sqi  | p_sqi | bas_sqi
#   --------------------|-------|-------|--------|-------|--------
#   Clean normal ECG    | 0.57  | 3.10  | 12.7   | 0.46  | 0.95
#   Arrhythmia (PVC)    | 0.76  | 2.93  | 11.5   | 0.58  | 0.92
#   Light noise (0.5σ)  | 0.67  | 2.16  | 8.1    | 0.37  | 0.96
#   Heavy noise (2σ)    | 0.26  | 0.20  | 0.39   | 0.30  | 0.97
#   Pure noise          | 0.37  | 0.09  | -0.13  | 0.29  | 0.97
#   Flat line           | 0.00  | NaN   | NaN    | NaN   | NaN
#   Motion artifact     | 0.67  | 0.58  | 0.94   | 0.46  | 0.92
#
# Notes:
# - bas_sqi is NOT a good discriminator on its own (stays ~0.95 for clean and noisy)
#   but is included in the score for completeness.
# - s_sqi and k_sqi together are the strongest discriminators (skewness + kurtosis
#   of clean ECG are much higher than noise because QRS peaks are sharp & asymmetric).
# - We classify each window into:
#     0 = bad         (any critical SQI fails)
#     1 = medium-low  (1+ SQI marginal)
#     2 = medium-high (all SQIs pass)
#     3 = excellent   (all SQIs strongly pass)
#
# Default: keep windows with score >= 2.

# Lazy-loaded module references (only imported when ecg_qc_quality_control is called)
_ecg_qc_sqi = None


def _load_ecg_qc_sqi():
    """Lazily import the SQI functions from ecg_qc. Returns None if not installed."""
    global _ecg_qc_sqi
    if _ecg_qc_sqi is None:
        try:
            from ecg_qc.sqi_computing.sqi_rr_intervals import csqi, qsqi
            from ecg_qc.sqi_computing.sqi_frequency_distribution import ssqi, ksqi
            from ecg_qc.sqi_computing.sqi_power_spectrum import bassqi, psqi
            _ecg_qc_sqi = {
                'qsqi': qsqi, 'csqi': csqi,
                'ssqi': ssqi, 'ksqi': ksqi,
                'psqi': psqi, 'bassqi': bassqi,
            }
        except ImportError:
            _ecg_qc_sqi = {}
    return _ecg_qc_sqi


def _compute_sqi_scores(segment: np.ndarray, fs: int) -> dict:
    """Compute all 6 SQI scores for a single ECG segment.

    Returns a dict with keys q_sqi, c_sqi, s_sqi, k_sqi, p_sqi, bas_sqi.
    On any error (flat line, too short, no R-peaks detected), returns
    a dict with all scores set to 0 / NaN — caller must handle.
    """
    sqi = _load_ecg_qc_sqi()
    if not sqi:
        return {}

    seg_list = list(segment)
    try:
        return {
            'q_sqi':   float(sqi['qsqi'](seg_list, fs)),
            'c_sqi':   float(sqi['csqi'](seg_list, fs)),
            's_sqi':   float(sqi['ssqi'](segment)),
            'k_sqi':   float(sqi['ksqi'](segment)),
            'p_sqi':   float(sqi['psqi'](seg_list, fs)),
            'bas_sqi': float(sqi['bassqi'](segment, fs)),
        }
    except Exception:
        # Flat line, NaN values, or no R-peaks detected.
        # Return values that will trigger class 0 (bad) in _classify_sqi:
        # q_sqi=0 (R-peak fail), k_sqi=0 (no peaks), bas_sqi=0 (no signal)
        return {
            'q_sqi': 0.0, 'c_sqi': 1.0, 's_sqi': 0.0,
            'k_sqi': 0.0, 'p_sqi': 0.0, 'bas_sqi': 0.0,
        }


def _classify_sqi(scores: dict) -> int:
    """Map SQI scores to a quality class 0-3.

    Thresholds based on empirical testing against clean normal ECG,
    arrhythmia ECG (PVCs - should be classified as clean), white noise,
    flat line, and motion artifacts (see reference table in the comment
    block above).

    Note: c_sqi (CoV of R-R intervals) is naturally higher on short 4s
    windows because there are only ~5 R-peaks per window. We therefore
    use more lenient thresholds on c_sqi and rely more heavily on
    s_sqi (skewness) and k_sqi (kurtosis), which are the strongest
    discriminators for short windows.

    Returns:
        3 = excellent (all SQIs strongly pass)
        2 = medium-high (all SQIs pass)
        1 = medium-low (1+ SQI marginal)
        0 = bad (any critical SQI fails)
    """
    q   = scores.get('q_sqi',   0.0)   # higher = better
    c   = scores.get('c_sqi',   1.0)   # lower  = better
    s   = scores.get('s_sqi',   0.0)   # higher = better
    k   = scores.get('k_sqi',   0.0)   # higher = better
    p   = scores.get('p_sqi',   0.0)   # higher = better
    bas = scores.get('bas_sqi', 0.0)   # higher = better

    # NaN guard (flat line produces NaN skewness/kurtosis)
    if any(v != v for v in (q, c, s, k, p, bas)):  # NaN check
        return 0

    # Critical failures -> score 0
    # (R-peak detector fails, no sharp peaks, or QRS band has no energy)
    if q < 0.30: return 0       # R-peak detector can't find regular beats
    if k < 1.0: return 0        # No sharp peaks (pure noise / flat)
    if p < 0.25: return 0       # QRS band has almost no energy
    if bas < 0.80: return 0     # Baseline noise dominates (>20% of total power)
    if c > 1.00: return 0       # Extremely irregular R-R intervals (very noisy)

    # Marginal -> at best score 1
    # (c_sqi threshold relaxed because short windows naturally have higher CoV)
    marginal = (
        (q < 0.40) or
        (k < 4.0) or
        (p < 0.38) or
        (s < 1.5) or
        (bas < 0.88)
    )
    if marginal:
        return 1

    # Excellent (all strongly pass — c_sqi excluded because of short-window bias)
    if q >= 0.45 and k >= 8.0 and p >= 0.42 and s >= 2.5 and bas >= 0.90:
        return 3

    # Default: medium-high
    return 2


def ecg_qc_quality_control(windows: np.ndarray,
                            fs: int = settings.SAMPLING_RATE_HZ,
                            min_quality: int = 2,
                            verbose: bool = False) -> tuple:
    """Filter ECG windows using the ecg_qc SQI feature extractor + heuristics.

    Each window is scored 0-3 (3 = excellent). Only windows with
    `min_quality` or higher are kept.

    Parameters
    ----------
    windows : np.ndarray
        (N, window_size) array of ECG windows.
    fs : int
        Sampling frequency in Hz (default: 128).
    min_quality : int
        Minimum quality class to keep (0-3). Default: 2.
        - 0 = keep all
        - 1 = drop only bad
        - 2 = drop bad + medium-low (recommended for training)
        - 3 = keep only excellent
    verbose : bool
        If True, prints per-window scores.

    Returns
    -------
    tuple of (kept_windows, stats)
        kept_windows : np.ndarray of shape (M, window_size)
        stats : dict with keys:
            n_in, n_out, n_kept, n_dropped, retention_pct,
            distribution: {0: int, 1: int, 2: int, 3: int},
            quality_labels: list[int] (one per input window)
    """
    if len(windows) == 0:
        empty = np.empty((0, windows.shape[1] if windows.ndim > 1 else 0), dtype=np.float32)
        return empty, {
            'n_in': 0, 'n_out': 0, 'n_kept': 0, 'n_dropped': 0,
            'retention_pct': 0.0, 'distribution': {0: 0, 1: 0, 2: 0, 3: 0},
            'quality_labels': [],
        }

    sqi = _load_ecg_qc_sqi()
    if not sqi:
        # Fallback to simple std-based filter if ecg_qc not installed
        if verbose:
            print("[ecg_qc] library not available, falling back to std-based filter")
        kept = quality_control(windows)
        labels = [2] * len(kept)  # treat all kept as medium-high
        return kept, {
            'n_in': len(windows), 'n_out': len(kept),
            'n_kept': len(kept), 'n_dropped': len(windows) - len(kept),
            'retention_pct': 100.0 * len(kept) / max(1, len(windows)),
            'distribution': {0: len(windows) - len(kept), 1: 0, 2: len(kept), 3: 0},
            'quality_labels': labels,
            'fallback': True,
        }

    labels = []
    distribution = {0: 0, 1: 0, 2: 0, 3: 0}

    for w in windows:
        scores = _compute_sqi_scores(w, fs)
        q_class = _classify_sqi(scores)
        labels.append(q_class)
        distribution[q_class] += 1
        if verbose:
            print(f"  [ecg_qc] class={q_class} q={scores.get('q_sqi',0):.3f} "
                  f"k={scores.get('k_sqi',0):.3f} p={scores.get('p_sqi',0):.3f} "
                  f"bas={scores.get('bas_sqi',0):.3f}")

    labels_arr = np.array(labels)
    mask = labels_arr >= min_quality
    kept = windows[mask]

    stats = {
        'n_in': len(windows),
        'n_out': len(kept),
        'n_kept': len(kept),
        'n_dropped': len(windows) - len(kept),
        'retention_pct': 100.0 * len(kept) / max(1, len(windows)),
        'distribution': distribution,
        'quality_labels': labels,
        'fallback': False,
    }
    return kept, stats


# ---------- Synthetic ECG generator (always available, no network needed) ----------

def _gaussian(x: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def synthetic_normal_ecg(duration_s: float = 30.0,
                         fs: int = settings.SAMPLING_RATE_HZ,
                         heart_rate_bpm: float = 72.0,
                         noise_std: float = 0.015,
                         seed: Optional[int] = None) -> np.ndarray:
    """Generate a realistic-looking normal sinus rhythm ECG.

    Each beat is composed of P, QRS, and T Gaussian bumps. Output is already
    z-scored and bandpass-clean, so it can be fed straight to the model.
    """
    if seed is not None:
        np.random.seed(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    period = 60.0 / heart_rate_bpm
    n_beats = int(duration_s / period) + 2

    signal = np.zeros(n, dtype=np.float32)
    for k in range(n_beats):
        start = k * period
        if start > duration_s:
            break
        # P wave
        signal += _gaussian(t, start + 0.10, 0.025, 0.15)
        # QRS complex (Q, R, S)
        signal += _gaussian(t, start + 0.20, 0.008, -0.10)
        signal += _gaussian(t, start + 0.22, 0.010,  1.00)
        signal += _gaussian(t, start + 0.24, 0.008, -0.25)
        # T wave
        signal += _gaussian(t, start + 0.40, 0.045, 0.30)

    # Add subtle baseline wander + noise
    signal += 0.05 * np.sin(2 * np.pi * 0.5 * t)
    signal += np.random.normal(0, noise_std, n).astype(np.float32)

    # Post-process like a real recording
    return preprocess_signal(signal, fs)


def synthetic_arrhythmia_ecg(duration_s: float = 30.0,
                             fs: int = settings.SAMPLING_RATE_HZ,
                             kind: str = "pvc",
                             seed: Optional[int] = None) -> np.ndarray:
    """Generate an ECG containing abnormal beats (PVCs / SVT / AFib-ish)."""
    if seed is not None:
        np.random.seed(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    base_hr = 72.0
    period = 60.0 / base_hr
    n_beats = int(duration_s / period) + 2

    signal = np.zeros(n, dtype=np.float32)
    for k in range(n_beats):
        start = k * period
        if start > duration_s:
            break

        # Every 3rd beat becomes abnormal
        is_abnormal = (k % 3 == 0)
        if is_abnormal:
            # Premature ventricular contraction: wide & tall QRS, no P wave
            signal += _gaussian(t, start + 0.22, 0.030, 1.80)   # wide R
            signal += _gaussian(t, start + 0.27, 0.040, -0.60)  # deep S
            signal += _gaussian(t, start + 0.50, 0.060, -0.40)  # inverted T
        else:
            signal += _gaussian(t, start + 0.10, 0.025, 0.15)
            signal += _gaussian(t, start + 0.20, 0.008, -0.10)
            signal += _gaussian(t, start + 0.22, 0.010,  1.00)
            signal += _gaussian(t, start + 0.24, 0.008, -0.25)
            signal += _gaussian(t, start + 0.40, 0.045, 0.30)

    signal += 0.05 * np.sin(2 * np.pi * 0.5 * t)
    signal += np.random.normal(0, 0.015, n).astype(np.float32)
    return preprocess_signal(signal, fs)


# ---------- MIT-BIH loaders (download from PhysioNet on first use) ----------

NSRDB_RECORDS = [
    "16265", "16272", "16420", "16483", "16539", "16773", "16786", "16795",
    "17052", "17453", "18177", "18184", "19088", "19090", "19093", "19140",
    "19830", "19840",
]

MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def _download_record(record_name: str, db_slug: str) -> Tuple[np.ndarray, int]:
    """Load a record from local files, PhysioNet cache, or download from PhysioNet.

    Search order:
    1. Local user-provided files in storage/datasets/{db_slug}/ (e.g. 100.dat + 100.hea)
    2. Cached .npy file from a previous download
    3. Download from PhysioNet (requires internet)

    To use your own MIT-BIH files: copy the .dat and .hea files to
    storage/datasets/nsrdb/ (for NSR DB) or storage/datasets/mitdb/ (for
    Arrhythmia DB). The system will use them instead of downloading.
    """
    cache_dir = settings.DATASET_DIR / db_slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check for local user-provided WFDB files (.dat + .hea)
    local_dat = cache_dir / f"{record_name}.dat"
    local_hea = cache_dir / f"{record_name}.hea"
    if local_dat.exists() and local_hea.exists():
        print(f"[ok] Loading {record_name} from local files ({cache_dir})")
        import wfdb
        record = wfdb.rdrecord(str(cache_dir / record_name))
        signal = record.p_signal[:, 0].astype(np.float32)
        fs = record.fs
        # Cache as .npy for faster subsequent loads
        cache_file = cache_dir / f"{record_name}.npy"
        fs_cache_file = cache_dir / f"{record_name}.fs"
        np.save(cache_file, signal)
        fs_cache_file.write_text(str(fs))
        return signal, fs

    # 2. Check for cached .npy file
    cache_file = cache_dir / f"{record_name}.npy"
    fs_cache_file = cache_dir / f"{record_name}.fs"
    if cache_file.exists() and fs_cache_file.exists():
        signal = np.load(cache_file)
        fs = int(fs_cache_file.read_text().strip())
        return signal, fs

    # 3. Download from PhysioNet
    print(f"[net] Downloading {record_name} from PhysioNet ({db_slug})...")
    import wfdb
    # Note: the parameter was renamed from 'pb_dir' to 'pn_dir' in wfdb 4.0+.
    try:
        record = wfdb.rdrecord(record_name, pn_dir=db_slug)
    except TypeError:
        record = wfdb.rdrecord(record_name, pb_dir=db_slug)
    # Use lead 0 (MLII in MITDB, ECG1 in NSRDB)
    signal = record.p_signal[:, 0].astype(np.float32)
    fs = record.fs
    np.save(cache_file, signal)
    fs_cache_file.write_text(str(fs))
    return signal, fs


def load_mitbih_nsr(records: Optional[List[str]] = None,
                    max_records: int = 4,
                    duration_s_per_record: float = 60.0,
                    use_ecg_qc: bool = True,
                    min_quality: int = 2) -> np.ndarray:
    """Load MIT-BIH NSR DB records, preprocess, windowize, and QC.

    Returns a (N, 512) array of normal ECG windows for training.

    Parameters
    ----------
    records : list of str, optional
        NSRDB record names to load. Defaults to the first `max_records` from NSRDB_RECORDS.
    max_records : int
        Number of records to use (only when `records` is None).
    duration_s_per_record : float
        Seconds of each record to load (default 60s — increase for more training data).
    use_ecg_qc : bool
        If True (default), use the ecg_qc SQI-based quality classifier.
        If False, fall back to the simple std-based filter.
    min_quality : int
        Minimum quality class (0-3) to keep when using ecg_qc. Default: 2.
    """
    records = records or NSRDB_RECORDS[:max_records]
    all_windows: List[np.ndarray] = []
    total_in = 0
    total_kept = 0
    for rec in records:
        try:
            signal, src_fs = _download_record(rec, "nsrdb")
        except Exception as e:
            print(f"[warn] Could not load NSRDB record {rec}: {e}")
            continue
        # Trim to requested duration
        max_samples = int(duration_s_per_record * src_fs)
        signal = signal[:max_samples]
        signal = preprocess_signal(signal, src_fs)
        windows = windowize(signal)
        total_in += len(windows)
        if use_ecg_qc:
            windows, stats = ecg_qc_quality_control(windows, fs=settings.SAMPLING_RATE_HZ,
                                                     min_quality=min_quality)
            print(f"[ok] NSRDB {rec}: {stats['n_kept']}/{stats['n_in']} windows kept "
                  f"(dist: {stats['distribution']})")
        else:
            windows = quality_control(windows)
            print(f"[ok] NSRDB {rec}: {len(windows)} windows (simple QC)")
        total_kept += len(windows)
        all_windows.append(windows)
    print(f"[qc] Total: {total_kept}/{total_in} windows kept ({100.0*total_kept/max(1,total_in):.1f}%)")
    if not all_windows:
        return np.empty((0, settings.WINDOW_SAMPLES), dtype=np.float32)
    return np.concatenate(all_windows, axis=0)


def load_mitbih_arrhythmia(records: Optional[List[str]] = None,
                           max_records: int = 4) -> List[Tuple[str, np.ndarray, int]]:
    """Load MIT-BIH Arrhythmia DB records for live replay / testing.

    Returns a list of (record_name, raw_signal, src_fs).
    """
    records = records or MITDB_RECORDS[:max_records]
    out = []
    for rec in records:
        try:
            signal, src_fs = _download_record(rec, "mitdb")
        except Exception as e:
            print(f"[warn] Could not load MITDB record {rec}: {e}")
            continue
        out.append((rec, signal, src_fs))
    return out


def stream_mitbih_record(record_name: str,
                         chunk_seconds: float = 4.0) -> Iterator[np.ndarray]:
    """Stream a MIT-BIH arrhythmia record in chunks for live UI playback."""
    signal, src_fs = _download_record(record_name, "mitdb")
    signal = preprocess_signal(signal, src_fs)
    chunk_size = int(chunk_seconds * settings.SAMPLING_RATE_HZ)
    for i in range(0, len(signal) - chunk_size, chunk_size):
        yield signal[i:i + chunk_size]


# ---------- Final tensor form ----------

def windows_to_tensor(windows: np.ndarray) -> "torch.Tensor":
    """(N, 512) → (N, 1, 512) torch.FloatTensor."""
    import torch
    return torch.from_numpy(windows).float().unsqueeze(1)


# ---------- Uploaded dataset loader ----------

def load_uploaded_dataset(upload_dir: Path,
                           max_records: int = 20,
                           duration_s_per_record: float = 60.0,
                           use_ecg_qc: bool = True,
                           min_quality: int = 2) -> np.ndarray:
    """Load ECG records from a user-uploaded dataset directory.

    Supports two formats:
    1. WFDB format (.dat + .hea pairs) — the native MIT-BIH format
       downloaded from PhysioNet. Each pair of files (e.g. 100.dat + 100.hea)
       represents one ECG record.
    2. CSV format (.csv) — single-column or two-column (time, value) files.
       The signal is assumed to be in millivolts; sampling rate is read from
       the 'fs' field in the .hea file if present, otherwise defaults to 128 Hz.

    Parameters
    ----------
    upload_dir : Path
        Directory containing the uploaded .dat/.hea or .csv files.
    max_records : int
        Maximum number of records to load (default 20).
    duration_s_per_record : float
        Seconds of each record to use (default 60s).
    use_ecg_qc : bool
        If True, apply ecg_qc SQI-based quality filter.
    min_quality : int
        Minimum quality class (0-3) to keep.

    Returns
    -------
    np.ndarray of shape (N, 512) — the preprocessed, windowed, QC-passed ECG.
    """
    upload_dir = Path(upload_dir)
    if not upload_dir.exists():
        raise FileNotFoundError(f"Upload directory not found: {upload_dir}")

    # Find all WFDB records (.dat files with corresponding .hea)
    dat_files = sorted(upload_dir.glob("*.dat"))
    csv_files = sorted(upload_dir.glob("*.csv"))

    all_windows: List[np.ndarray] = []
    records_loaded = 0

    # Load WFDB format records
    for dat_file in dat_files:
        if records_loaded >= max_records:
            break
        record_name = dat_file.stem  # e.g. "100" from "100.dat"
        hea_file = upload_dir / f"{record_name}.hea"
        if not hea_file.exists():
            print(f"[warn] {record_name}: .hea file missing, skipping")
            continue

        try:
            import wfdb
            record = wfdb.rdrecord(str(upload_dir / record_name))
            signal = record.p_signal[:, 0].astype(np.float32)
            src_fs = record.fs
            # Trim to requested duration
            max_samples = int(duration_s_per_record * src_fs)
            signal = signal[:max_samples]
            signal = preprocess_signal(signal, src_fs)
            windows = windowize(signal)
            total_in = len(windows)
            if use_ecg_qc:
                windows, stats = ecg_qc_quality_control(
                    windows, fs=settings.SAMPLING_RATE_HZ,
                    min_quality=min_quality,
                )
                print(f"[ok] Uploaded WFDB {record_name}: {stats['n_kept']}/{stats['n_in']} "
                      f"windows kept (fs={src_fs}Hz, dist={stats['distribution']})")
            else:
                windows = quality_control(windows)
                print(f"[ok] Uploaded WFDB {record_name}: {len(windows)} windows (simple QC)")
            all_windows.append(windows)
            records_loaded += 1
        except Exception as e:
            print(f"[warn] {record_name}: load failed ({e})")
            continue

    # Load CSV format records
    for csv_file in csv_files:
        if records_loaded >= max_records:
            break
        record_name = csv_file.stem
        try:
            import pandas as pd
            # Try to read sampling rate from a matching .hea file
            hea_file = upload_dir / f"{record_name}.hea"
            src_fs = 128  # default
            if hea_file.exists():
                try:
                    with open(hea_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 3 and parts[1].isdigit():
                                src_fs = int(parts[2]) if len(parts) > 2 else src_fs
                                break
                except Exception:
                    pass

            # Read CSV - assume single column of values, or two columns (time, value)
            df = pd.read_csv(csv_file, header=None)
            if df.shape[1] >= 2:
                signal = df.iloc[:, 1].values.astype(np.float32)
            else:
                signal = df.iloc[:, 0].values.astype(np.float32)

            # Trim to requested duration
            max_samples = int(duration_s_per_record * src_fs)
            signal = signal[:max_samples]
            signal = preprocess_signal(signal, src_fs)
            windows = windowize(signal)
            if use_ecg_qc:
                windows, stats = ecg_qc_quality_control(
                    windows, fs=settings.SAMPLING_RATE_HZ,
                    min_quality=min_quality,
                )
                print(f"[ok] Uploaded CSV {record_name}: {stats['n_kept']}/{stats['n_in']} "
                      f"windows kept (fs={src_fs}Hz, dist={stats['distribution']})")
            else:
                windows = quality_control(windows)
                print(f"[ok] Uploaded CSV {record_name}: {len(windows)} windows (simple QC)")
            all_windows.append(windows)
            records_loaded += 1
        except Exception as e:
            print(f"[warn] CSV {record_name}: load failed ({e})")
            continue

    if not all_windows:
        raise RuntimeError(
            f"No valid records found in {upload_dir}. "
            f"Expected .dat + .hea files (WFDB format) or .csv files. "
            f"Found {len(dat_files)} .dat, {len(csv_files)} .csv files."
        )

    print(f"[qc] Uploaded total: {records_loaded} records loaded")
    return np.concatenate(all_windows, axis=0)


def list_uploaded_datasets() -> list:
    """List all uploaded dataset directories in the storage/datasets/uploads/ folder."""
    upload_root = settings.DATASET_DIR / "uploads"
    if not upload_root.exists():
        return []
    datasets = []
    for d in sorted(upload_root.iterdir()):
        if d.is_dir():
            dat_count = len(list(d.glob("*.dat")))
            csv_count = len(list(d.glob("*.csv")))
            hea_count = len(list(d.glob("*.hea")))
            datasets.append({
                "id": d.name,
                "name": d.name,
                "path": str(d),
                "dat_files": dat_count,
                "csv_files": csv_count,
                "hea_files": hea_count,
                "total_records": dat_count + csv_count,
            })
    return datasets


if __name__ == "__main__":
    # Quick test: generate synthetic normal + arrhythmia ECG
    print("Testing synthetic generators...")
    normal = synthetic_normal_ecg(duration_s=10.0, seed=42)
    arrh   = synthetic_arrhythmia_ecg(duration_s=10.0, seed=42)
    print(f"Normal ECG shape   : {normal.shape}, range [{normal.min():.3f}, {normal.max():.3f}]")
    print(f"Arrhythmia shape   : {arrh.shape}, range [{arrh.min():.3f}, {arrh.max():.3f}]")

    w = windowize(normal)
    print(f"Windows from normal: {w.shape}")
