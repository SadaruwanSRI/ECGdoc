"""Rebuild supervised inputs from source ECG for the final audit rerun."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings
from app.ml.data import MITDB_RECORDS, _download_record, resample_to
from app.ml.classifier import ARRHYTHMIA_CLASSES, mitbih_symbol_is_anomaly, mitbih_symbol_to_class
from app.ml.train_hierarchical import _rhythm_lookup
from app.ml.beat_preparation import prepare_beat, base_rr_at
from app.ml.feature_system import extend_rr_features

CACHE = settings.DATASET_DIR / "mitdb/corrected_beat_local_v1.npz"


def extract(record: str, database: str = "mitdb", lead: str = "MLII") -> dict:
    raw, fs = _download_record(record, database, required_lead=lead)
    signal = resample_to(raw, fs, 128)
    prefix = settings.DATASET_DIR / database / record
    ann = (wfdb.rdann(str(prefix), "atr") if prefix.with_suffix(".atr").exists()
           else wfdb.rdann(record, "atr", pn_dir=database))
    peaks_all = (ann.sample * 128 / fs).astype(int)
    rhythm_at = _rhythm_lookup(peaks_all, getattr(ann, "aux_note", []))
    accepted = []
    for peak, symbol in zip(peaks_all, ann.symbol):
        rhythm = rhythm_at(int(peak))
        binary = mitbih_symbol_is_anomaly(symbol, rhythm)
        if binary is not None and peak >= 200 and peak + 312 <= len(signal):
            cls = mitbih_symbol_to_class(symbol, rhythm)
            accepted.append((int(peak), int(binary), ARRHYTHMIA_CLASSES.index(cls) if cls else -1))
    peaks = np.asarray([row[0] for row in accepted], dtype=np.int64)
    rows = {k: [] for k in ("windows", "binary", "classes", "rr_features", "rr15", "peaks", "records")}
    # First/last eligible beat lacks a measured neighbouring RR. No global
    # median or duplicated interval is substituted into the corrected model.
    for i in range(1, len(accepted) - 1):
        peak, binary, cls = accepted[i]
        try:
            window = prepare_beat(signal[peak - 200:peak + 312])
            base = base_rr_at(peaks, i)
        except ValueError:
            continue
        history = np.diff(peaks[max(0, i - 20):i + 1]).astype(np.float64) / 128
        for key, value in zip(rows, (window, binary, cls, base, extend_rr_features(base, history), peak, record)):
            rows[key].append(value)
    return {key: np.asarray(value, dtype="<U3" if key == "records" else
                            np.int64 if key in ("binary", "classes", "peaks") else np.float32)
            for key, value in rows.items()}


def load_dataset() -> dict:
    if CACHE.exists():
        with np.load(CACHE) as cached:
            return {key: cached[key] for key in cached.files}
    pieces = []
    for record in MITDB_RECORDS:
        pieces.append(extract(record))
        print(f"[corrected data] {record}: {len(pieces[-1]['windows'])} beats", flush=True)
    result = {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}
    np.savez_compressed(CACHE, **result)
    return result
