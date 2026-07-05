"""Model evaluation endpoints using annotated MIT-BIH Arrhythmia records."""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from app.api.routes_auth import require_user
from app.api.schemas import ModelEvaluationRequest, OkResponse
from app.core.config import settings
from app.db.session import get_db
from app.ml.classifier import (
    ARRHYTHMIA_CLASSES,
    mitbih_symbol_is_anomaly,
    mitbih_symbol_to_class,
)
from app.ml.data import _download_record, preprocess_signal
from app.ml.inference import get_engine

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


def _load_annotations(record_name: str):
    import wfdb

    cache_dir = settings.DATASET_DIR / "mitdb"
    local_hea = cache_dir / f"{record_name}.hea"
    local_atr = cache_dir / f"{record_name}.atr"
    if local_hea.exists() and local_atr.exists():
        return wfdb.rdann(str(cache_dir / record_name), "atr")

    try:
        return wfdb.rdann(record_name, "atr", pn_dir="mitdb")
    except TypeError:
        return wfdb.rdann(record_name, "atr", pb_dir="mitdb")


def _rhythm_lookup(samples: np.ndarray, aux_notes: list[str]):
    changes = []
    current: Optional[str] = None
    for sample, note in zip(samples, aux_notes):
        note = (note or "").strip()
        if not note.startswith("("):
            continue
        if "AFIB" in note or "AFL" in note:
            current = "AFib"
        else:
            current = None
        changes.append((int(sample), current))

    def rhythm_at(sample: int) -> Optional[str]:
        rhythm = None
        for change_sample, change_rhythm in changes:
            if change_sample > sample:
                break
            rhythm = change_rhythm
        return rhythm

    return rhythm_at


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _detection_metrics(samples: list[dict], threshold: float) -> dict:
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for sample in samples:
        truth = bool(sample["true_anomaly"])
        pred = float(sample["score"]) > threshold
        if truth and pred:
            confusion["tp"] += 1
        elif not truth and not pred:
            confusion["tn"] += 1
        elif not truth and pred:
            confusion["fp"] += 1
        else:
            confusion["fn"] += 1

    total = sum(confusion.values())
    accuracy = _safe_div(confusion["tp"] + confusion["tn"], total)
    sensitivity = _safe_div(confusion["tp"], confusion["tp"] + confusion["fn"])
    specificity = _safe_div(confusion["tn"], confusion["tn"] + confusion["fp"])
    precision = _safe_div(confusion["tp"], confusion["tp"] + confusion["fp"])
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)

    return {
        "threshold": float(threshold),
        "confusion": confusion,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def _threshold_sweep(samples: list[dict], default_threshold: float) -> dict:
    if not samples:
        empty = _detection_metrics(samples, default_threshold)
        return {"best": empty, "default": empty, "candidates": []}

    scores = np.array([float(s["score"]) for s in samples], dtype=np.float32)
    percentiles = np.linspace(5, 95, 37)
    candidate_thresholds = sorted({
        float(default_threshold),
        *[float(np.percentile(scores, p)) for p in percentiles],
        float(scores.min()),
        float(scores.max()),
    })

    candidates = [_detection_metrics(samples, thr) for thr in candidate_thresholds]
    best = max(
        candidates,
        key=lambda m: (m["f1"], m["sensitivity"], m["specificity"], -abs(m["threshold"] - default_threshold)),
    )
    default = _detection_metrics(samples, default_threshold)
    return {
        "best": best,
        "default": default,
        "candidates": candidates,
    }


@router.post("/mitbih", response_model=OkResponse)
def evaluate_mitbih(req: ModelEvaluationRequest, user=Depends(require_user)):
    """Evaluate models against MIT-BIH Arrhythmia beat annotations.

    The anomaly model is tested as a binary detector:
    normal beat -> non-anomaly, abnormal beat/rhythm -> anomaly.

    If a classifier is selected, classification is measured on abnormal beats.
    """
    records = [r.strip() for r in req.records if r.strip()]
    if not records:
        raise HTTPException(400, "At least one MIT-BIH record is required")
    if req.max_beats < 1:
        raise HTTPException(400, "max_beats must be positive")

    with get_db() as db:
        model = db.execute(text("""
            SELECT modelPath, threshold, status FROM ModelVersion WHERE id = :id
        """), {"id": req.model_id}).fetchone()
        if not model or model[2] != "ready" or not model[0]:
            raise HTTPException(404, "Ready anomaly model not found")

        classifier_path = None
        if req.classifier_model_id:
            clf = db.execute(text("""
                SELECT modelPath, status FROM ModelVersion WHERE id = :id
            """), {"id": req.classifier_model_id}).fetchone()
            if not clf or clf[1] != "ready" or not clf[0]:
                raise HTTPException(404, "Ready classifier model not found")
            classifier_path = clf[0]

    threshold = req.threshold_override if req.threshold_override is not None else model[1]
    engine = get_engine(model[0], threshold=threshold, classifier_path=classifier_path)

    class_confusion = {
        cls: {pred: 0 for pred in ARRHYTHMIA_CLASSES}
        for cls in ARRHYTHMIA_CLASSES
    }
    per_class_support = defaultdict(int)
    score_rows = []
    scored_samples = []
    processed = 0
    record_summaries = []
    win = settings.WINDOW_SAMPLES

    for record_name in records:
        if processed >= req.max_beats:
            break

        try:
            raw_signal, src_fs = _download_record(record_name, "mitdb")
            ann = _load_annotations(record_name)
        except Exception as exc:
            record_summaries.append({
                "record": record_name,
                "status": "failed",
                "error": str(exc),
                "beats": 0,
            })
            continue

        signal = preprocess_signal(raw_signal, src_fs)
        scale = settings.SAMPLING_RATE_HZ / src_fs
        peaks = (ann.sample * scale).astype(int)
        rhythm_at = _rhythm_lookup(peaks, getattr(ann, "aux_note", []))
        record_count = 0

        for peak, symbol in zip(peaks, ann.symbol):
            if processed >= req.max_beats:
                break
            rhythm = rhythm_at(int(peak))
            is_true_anomaly = mitbih_symbol_is_anomaly(symbol, rhythm=rhythm)
            if is_true_anomaly is None:
                continue

            true_class = mitbih_symbol_to_class(symbol, rhythm=rhythm) or "OTHER"

            start = int(peak) - req.r_peak_before
            end = start + win
            if start < 0 or end > len(signal):
                continue

            window = signal[start:end].astype(np.float32)
            score, _, _ = engine.score_window(window)
            scored_samples.append({
                "score": float(score),
                "true_anomaly": bool(is_true_anomaly),
            })

            classification = None
            if engine.classifier is not None and true_class in ARRHYTHMIA_CLASSES:
                classification = engine.classify_window(window)
                pred_class = classification["class"]
                if pred_class in class_confusion[true_class]:
                    class_confusion[true_class][pred_class] += 1

            per_class_support[true_class] += 1
            if len(score_rows) < 200:
                score_rows.append({
                    "record": record_name,
                    "sample": int(peak),
                    "symbol": symbol,
                    "true_class": true_class,
                    "true_anomaly": is_true_anomaly,
                    "score": float(score),
                    "predicted_anomaly": bool(score > engine.threshold),
                    "predicted_class": classification["class"] if classification else None,
                    "confidence": classification["confidence"] if classification else None,
                })

            processed += 1
            record_count += 1

        record_summaries.append({
            "record": record_name,
            "status": "ok",
            "beats": record_count,
        })

    sweep = _threshold_sweep(scored_samples, engine.threshold)
    selected_detection = sweep["best"] if req.optimize_threshold else sweep["default"]
    selected_threshold = selected_detection["threshold"]
    for row in score_rows:
        row["predicted_anomaly"] = bool(row["score"] > selected_threshold)

    classifier_metrics = None
    if engine.classifier is not None:
        per_class = {}
        correct = 0
        class_total = 0
        f1_values = []
        for cls in ARRHYTHMIA_CLASSES:
            tp = class_confusion[cls][cls]
            fp = sum(class_confusion[other][cls] for other in ARRHYTHMIA_CLASSES if other != cls)
            fn = sum(v for pred, v in class_confusion[cls].items() if pred != cls)
            support = sum(class_confusion[cls].values())
            correct += tp
            class_total += support
            cls_precision = _safe_div(tp, tp + fp)
            cls_recall = _safe_div(tp, tp + fn)
            cls_f1 = _safe_div(2 * cls_precision * cls_recall, cls_precision + cls_recall)
            if support:
                f1_values.append(cls_f1)
            per_class[cls] = {
                "precision": cls_precision,
                "recall": cls_recall,
                "f1": cls_f1,
                "support": support,
            }
        classifier_metrics = {
            "accuracy": _safe_div(correct, class_total),
            "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
            "confusion_matrix": class_confusion,
            "per_class": per_class,
        }

    return OkResponse(ok=True, data={
        "threshold": float(selected_threshold),
        "default_threshold": float(engine.threshold),
        "optimized_threshold": sweep["best"]["threshold"],
        "threshold_optimization": {
            "enabled": req.optimize_threshold,
            "selected": sweep["best"] if req.optimize_threshold else sweep["default"],
            "default": sweep["default"],
            "best": sweep["best"],
            "candidates": sweep["candidates"],
        },
        "records": record_summaries,
        "n_beats": processed,
        "class_support": dict(per_class_support),
        "detection": selected_detection,
        "classifier": classifier_metrics,
        "examples": score_rows,
    })
