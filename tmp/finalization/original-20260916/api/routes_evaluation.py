"""Model evaluation endpoints using annotated ECG datasets."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import threading
from typing import Callable, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
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
from app.ml.data import INCART_RECORDS, MITDB_RECORDS, _download_record, preprocess_signal
from app.ml.feature_system import extend_rr_features
from app.ml.inference import get_engine

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


EVALUATION_JOBS: dict[str, dict] = {}
EVALUATION_JOBS_LOCK = threading.Lock()
ProgressCallback = Callable[[dict], None]
EVALUATION_RESULT_DIR = settings.STORAGE_DIR / "evaluations"


def _user_identity(user) -> tuple[Optional[str], Optional[str]]:
    if isinstance(user, tuple):
        return str(user[0]) if len(user) > 0 else None, str(user[1]) if len(user) > 1 else None
    if isinstance(user, dict):
        return user.get("id"), user.get("email")
    return getattr(user, "id", None), getattr(user, "email", None)


def _save_evaluation_result(result: dict) -> None:
    """Persist completed evaluation results for later report generation."""
    dataset = result.get("dataset", {}) or {}
    dataset_id = str(dataset.get("id") or "unknown")
    saved_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "saved_at": saved_at,
        "dataset_id": dataset_id,
        "validation_kind": dataset.get("validation_kind"),
        "result": result,
    }
    EVALUATION_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = saved_at.replace(":", "").replace("-", "").split(".")[0]
    path = EVALUATION_RESULT_DIR / f"{dataset_id}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (EVALUATION_RESULT_DIR / f"latest_{dataset_id}.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    if dataset.get("validation_kind") == "external":
        (EVALUATION_RESULT_DIR / "latest_external_evaluation.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )


EVALUATION_DATASETS = {
    "mit-bih-arrhythmia": {
        "id": "mit-bih-arrhythmia",
        "name": "MIT-BIH Arrhythmia Database",
        "physionet_slug": "mitdb",
        "records": MITDB_RECORDS,
        "default_records": ["100", "101", "103"],
        "default_lead": "MLII",
        "supported_leads": ["MLII"],
        "sampling_rate_hz": 360,
        "record_count": len(MITDB_RECORDS),
        "annotation_extension": "atr",
        "validation_kind": "internal-source",
        "description": (
            "MIT-BIH Arrhythmia records containing MLII. Useful for pipeline "
            "regression, but not a fully external dataset for this project."
        ),
        "warning": (
            "This dataset is from the same MIT-BIH family as the final model "
            "development protocol. Use INCART for external validation."
        ),
    },
    "incartdb": {
        "id": "incartdb",
        "name": "St Petersburg INCART 12-lead Arrhythmia Database",
        "physionet_slug": "incartdb",
        "records": INCART_RECORDS,
        "default_records": ["I01", "I02", "I03"],
        "default_lead": "II",
        "supported_leads": ["II"],
        "sampling_rate_hz": 257,
        "record_count": len(INCART_RECORDS),
        "annotation_extension": "atr",
        "validation_kind": "external",
        "description": (
            "75 unseen 30-minute 12-lead Holter records with beat annotations. "
            "Lead II is used as the closest practical external counterpart to MLII."
        ),
        "warning": (
            "External validation: INCART lead II is not identical to MIT-BIH MLII, "
            "so results measure dataset, device, and lead-domain shift."
        ),
    },
}


@router.get("/datasets", response_model=OkResponse)
def list_evaluation_datasets(user=Depends(require_user)):
    """Return annotated datasets supported by the performance evaluator."""
    return OkResponse(ok=True, data={"datasets": list(EVALUATION_DATASETS.values())})


@router.get("/final-study", response_model=OkResponse)
def get_final_study(user=Depends(require_user)):
    """Return the saved final MLII temporal-holdout benchmark."""
    study_path = (
        Path(__file__).resolve().parents[3]
        / "Final_Report_Research"
        / "experiments"
        / "final_temporal_holdout.json"
    )
    if not study_path.exists():
        raise HTTPException(404, "The final MLII study is not available")
    try:
        study = json.loads(study_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"Could not read the final MLII study: {exc}") from exc
    return OkResponse(ok=True, data={
        "protocol": study.get("protocol", {}),
        "test_binary": study.get("test_binary", {}),
        "test_subtype": study.get("test_subtype_on_true_supported_arrhythmias", {}),
        "test_whole_system": study.get("test_whole_system", {}),
        "clustered_uncertainty": study.get("clustered_uncertainty", {}),
    })


def _load_annotations(record_name: str, db_slug: str, extension: str = "atr"):
    import wfdb

    cache_dir = settings.DATASET_DIR / db_slug
    local_hea = cache_dir / f"{record_name}.hea"
    local_ann = cache_dir / f"{record_name}.{extension}"
    if local_hea.exists() and local_ann.exists():
        return wfdb.rdann(str(cache_dir / record_name), extension)

    try:
        return wfdb.rdann(record_name, extension, pn_dir=db_slug)
    except TypeError:
        return wfdb.rdann(record_name, extension, pb_dir=db_slug)


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
    truth = np.asarray(
        [bool(sample["true_anomaly"]) for sample in samples],
        dtype=np.int64,
    )
    scores = np.asarray(
        [float(sample["score"]) for sample in samples],
        dtype=np.float64,
    )
    predicted = scores > threshold
    has_both_classes = len(np.unique(truth)) == 2

    return {
        "threshold": float(threshold),
        "confusion": confusion,
        "accuracy": accuracy,
        "balanced_accuracy": (
            float(balanced_accuracy_score(truth, predicted))
            if has_both_classes
            else None
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auroc": (
            float(roc_auc_score(truth, scores))
            if has_both_classes
            else None
        ),
        "auprc": (
            float(average_precision_score(truth, scores))
            if np.any(truth == 1)
            else None
        ),
        "brier": (
            float(brier_score_loss(truth, scores))
            if (
                len(truth)
                and np.isfinite(scores).all()
                and np.all((scores >= 0.0) & (scores <= 1.0))
            )
            else None
        ),
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
        key=lambda m: (
            m["balanced_accuracy"]
            if m["balanced_accuracy"] is not None
            else -1.0,
            m["f1"],
            m["sensitivity"],
            m["specificity"],
            -abs(m["threshold"] - default_threshold),
        ),
    )
    default = _detection_metrics(samples, default_threshold)
    return {
        "best": best,
        "default": default,
        "candidates": candidates,
    }


@router.post("/run", response_model=OkResponse)
def evaluate_dataset(req: ModelEvaluationRequest, user=Depends(require_user)):
    """Evaluate models against the selected annotated ECG dataset."""
    return _evaluate_annotated_dataset(req)


@router.post("/start", response_model=OkResponse)
def start_evaluation(req: ModelEvaluationRequest, user=Depends(require_user)):
    """Start a background evaluation job and return its progress id."""
    job_id = secrets.token_hex(12)
    dataset = EVALUATION_DATASETS.get(req.dataset_id)
    owner_id, owner_email = _user_identity(user)
    initial_job = {
        "job_id": job_id,
        "owner_id": owner_id,
        "owner_email": owner_email,
        "status": "queued",
        "processed": 0,
        "target": int(req.max_beats),
        "progress": 0.0,
        "current_detection": None,
        "message": "Waiting to start evaluation",
        "dataset": {
            "id": req.dataset_id,
            "name": dataset["name"] if dataset else req.dataset_id,
        },
        "result": None,
        "error": None,
    }
    with EVALUATION_JOBS_LOCK:
        EVALUATION_JOBS[job_id] = initial_job

    def update_progress(update: dict):
        with EVALUATION_JOBS_LOCK:
            job = EVALUATION_JOBS.get(job_id)
            if not job:
                return
            job.update(update)

    def worker():
        try:
            update_progress({"status": "running", "message": "Preparing evaluation"})
            response = _evaluate_annotated_dataset(req, progress_callback=update_progress)
            update_progress({
                "status": "completed",
                "processed": response.data.get("n_beats", 0) if response.data else 0,
                "progress": 100.0,
                "message": "Evaluation complete",
                "result": response.data,
            })
        except HTTPException as exc:
            update_progress({
                "status": "failed",
                "progress": 100.0,
                "message": "Evaluation failed",
                "error": str(exc.detail),
            })
        except Exception as exc:
            update_progress({
                "status": "failed",
                "progress": 100.0,
                "message": "Evaluation failed",
                "error": str(exc),
            })

    threading.Thread(target=worker, daemon=True).start()
    return OkResponse(ok=True, data={"job_id": job_id})


@router.get("/progress/{job_id}", response_model=OkResponse)
def get_evaluation_progress(job_id: str, user=Depends(require_user)):
    """Return a background evaluation job snapshot."""
    with EVALUATION_JOBS_LOCK:
        job = EVALUATION_JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Evaluation job not found")
        owner_id, owner_email = _user_identity(user)
        if (
            (job.get("owner_id") and owner_id and job["owner_id"] != owner_id)
            or (job.get("owner_email") and owner_email and job["owner_email"] != owner_email)
        ):
            raise HTTPException(404, "Evaluation job not found")
        snapshot = dict(job)
    snapshot.pop("owner_id", None)
    snapshot.pop("owner_email", None)
    return OkResponse(ok=True, data=snapshot)


@router.post("/mitbih", response_model=OkResponse)
def evaluate_mitbih(req: ModelEvaluationRequest, user=Depends(require_user)):
    """Backward-compatible MIT-BIH evaluator."""
    req.dataset_id = "mit-bih-arrhythmia"
    req.lead_name = "MLII"
    return _evaluate_annotated_dataset(req)


def _evaluate_annotated_dataset(
    req: ModelEvaluationRequest,
    progress_callback: Optional[ProgressCallback] = None,
):
    """Evaluate models against beat annotations.

    The anomaly model is tested as a binary detector:
    normal beat -> non-anomaly; abnormal beat/rhythm -> anomaly.

    If a classifier is selected, classification is measured on abnormal beats.
    """
    dataset = EVALUATION_DATASETS.get(req.dataset_id)
    if dataset is None:
        raise HTTPException(400, f"Unsupported evaluation dataset: {req.dataset_id}")
    if req.mode != "offline":
        raise HTTPException(400, "Only offline beat-level evaluation is currently supported")

    records = [r.strip() for r in req.records if r.strip()]
    if not records:
        records = list(dataset["default_records"])
    available_records = set(dataset["records"])
    unknown_records = [r for r in records if r not in available_records]
    if unknown_records:
        raise HTTPException(
            400,
            (
                f"{dataset['name']} does not include records: "
                f"{', '.join(unknown_records[:10])}"
            ),
        )
    lead_name = req.lead_name or str(dataset["default_lead"])
    supported_leads = {str(lead).upper() for lead in dataset["supported_leads"]}
    if lead_name.upper() not in supported_leads:
        raise HTTPException(
            400,
            (
                f"Lead {lead_name} is not configured for {dataset['name']}; "
                f"use one of: {', '.join(dataset['supported_leads'])}"
            ),
        )
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
    engine = get_engine(
        model[0],
        threshold=threshold,
        classifier_path=classifier_path,
        improved_threshold=req.threshold_override,
    )
    decision_threshold = engine.decision_threshold
    target_beats = int(req.max_beats)

    def emit_progress(message: str, include_metrics: bool = False):
        if progress_callback is None:
            return
        payload = {
            "status": "running",
            "processed": processed,
            "target": target_beats,
            "progress": min(99.0, _safe_div(processed, target_beats) * 100.0),
            "message": message,
            "dataset": {
                "id": dataset["id"],
                "name": dataset["name"],
                "lead_name": lead_name,
            },
        }
        if include_metrics and scored_samples:
            payload["current_detection"] = _detection_metrics(
                scored_samples,
                decision_threshold,
            )
        progress_callback(payload)

    predicted_class_names = [
        *ARRHYTHMIA_CLASSES,
        "Unclassified abnormal",
    ]
    class_confusion = {
        cls: {pred: 0 for pred in predicted_class_names}
        for cls in ARRHYTHMIA_CLASSES
    }
    per_class_support = defaultdict(int)
    score_rows = []
    scored_samples = []
    record_samples: dict[str, list[dict]] = defaultdict(list)
    abnormal_symbol_samples: dict[str, list[dict]] = defaultdict(list)
    processed = 0
    record_summaries = []
    win = settings.WINDOW_SAMPLES

    emit_progress("Loading selected model")
    for record_name in records:
        if processed >= req.max_beats:
            break

        emit_progress(f"Loading record {record_name}", include_metrics=bool(scored_samples))
        try:
            raw_signal, src_fs = _download_record(
                record_name,
                str(dataset["physionet_slug"]),
                required_lead=lead_name,
            )
            ann = _load_annotations(
                record_name,
                str(dataset["physionet_slug"]),
                str(dataset["annotation_extension"]),
            )
        except Exception as exc:
            record_summaries.append({
                "record": record_name,
                "status": "failed",
                "error": str(exc),
                "beats": 0,
            })
            emit_progress(f"Record {record_name} failed", include_metrics=bool(scored_samples))
            continue

        signal = preprocess_signal(raw_signal, src_fs)
        scale = settings.SAMPLING_RATE_HZ / src_fs
        peaks = (ann.sample * scale).astype(int)
        rhythm_at = _rhythm_lookup(peaks, getattr(ann, "aux_note", []))
        record_count = 0

        accepted = []
        for peak, symbol in zip(peaks, ann.symbol):
            rhythm = rhythm_at(int(peak))
            binary_truth = mitbih_symbol_is_anomaly(symbol, rhythm=rhythm)
            start = int(peak) - req.r_peak_before
            end = start + win
            if binary_truth is None or start < 0 or end > len(signal):
                continue
            accepted.append((int(peak), symbol, rhythm, bool(binary_truth)))
        accepted_peaks = np.asarray([item[0] for item in accepted], dtype=np.int64)
        intervals = (
            np.diff(accepted_peaks).astype(np.float32) / settings.SAMPLING_RATE_HZ
        )
        typical_interval = float(np.median(intervals)) if len(intervals) else 1.0

        for beat_index, (peak, symbol, rhythm, is_true_anomaly) in enumerate(accepted):
            if processed >= req.max_beats:
                break

            true_class = mitbih_symbol_to_class(symbol, rhythm=rhythm) or "OTHER"

            start = int(peak) - req.r_peak_before
            end = start + win
            if start < 0 or end > len(signal):
                continue

            window = signal[start:end].astype(np.float32)
            classification = None
            if engine.improved_system is not None:
                previous_rr = (
                    float(intervals[beat_index - 1])
                    if beat_index > 0
                    else typical_interval
                )
                next_rr = (
                    float(intervals[beat_index])
                    if beat_index < len(intervals)
                    else previous_rr
                )
                history = intervals[max(0, beat_index - 10):beat_index]
                if len(history) == 0:
                    history = np.asarray([typical_interval], dtype=np.float32)
                local_mean = float(np.mean(history))
                local_std = float(np.std(history))
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
                        intervals[max(0, beat_index - 20):beat_index],
                    )
                    if engine.improved_system.rr_feature_count == 15
                    else base_rr_features
                )
                classification = engine.classify_beat(
                    window, rr_features
                )
                score = float(classification["anomaly_probability"])
            else:
                score, _, _ = engine.score_window(window)
            scored_samples.append({
                "score": float(score),
                "true_anomaly": bool(is_true_anomaly),
            })
            sample_result = {
                "score": float(score),
                "true_anomaly": bool(is_true_anomaly),
            }
            record_samples[record_name].append(sample_result)
            if is_true_anomaly:
                abnormal_symbol_samples[str(symbol)].append(sample_result)

            if engine.classifier is not None and true_class in ARRHYTHMIA_CLASSES:
                classification = engine.classify_window(window)
            if (
                classification is not None
                and true_class in ARRHYTHMIA_CLASSES
            ):
                pred_class = classification["class"]
                if pred_class in class_confusion[true_class]:
                    class_confusion[true_class][pred_class] += 1
                else:
                    class_confusion[true_class]["Unclassified abnormal"] += 1

            per_class_support[true_class] += 1
            if len(score_rows) < 200:
                score_rows.append({
                    "record": record_name,
                    "sample": int(peak),
                    "symbol": symbol,
                    "true_class": true_class,
                    "true_anomaly": is_true_anomaly,
                    "score": float(score),
                    "predicted_anomaly": bool(score > decision_threshold),
                    "predicted_class": classification["class"] if classification else None,
                    "confidence": classification["confidence"] if classification else None,
                })

            processed += 1
            record_count += 1
            if processed <= 10 or processed % 25 == 0 or processed >= req.max_beats:
                emit_progress(
                    f"Scored {processed} beats; current record {record_name}",
                    include_metrics=True,
                )

        record_summaries.append({
            "record": record_name,
            "status": "ok",
            "beats": record_count,
        })
        emit_progress(
            f"Finished record {record_name}",
            include_metrics=bool(scored_samples),
        )

    sweep = _threshold_sweep(scored_samples, decision_threshold)
    selected_detection = sweep["best"] if req.optimize_threshold else sweep["default"]
    selected_threshold = selected_detection["threshold"]
    for row in score_rows:
        row["predicted_anomaly"] = bool(row["score"] > selected_threshold)

    per_record_detection = []
    for summary in record_summaries:
        name = str(summary["record"])
        samples = record_samples.get(name, [])
        if samples:
            per_record_detection.append({
                "record": name,
                **_detection_metrics(samples, selected_threshold),
            })

    abnormal_by_symbol = []
    for symbol, samples in sorted(
        abnormal_symbol_samples.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        detected = sum(
            float(sample["score"]) > selected_threshold for sample in samples
        )
        abnormal_by_symbol.append({
            "symbol": symbol,
            "support": len(samples),
            "detected": int(detected),
            "missed": int(len(samples) - detected),
            "recall": _safe_div(detected, len(samples)),
        })

    classifier_metrics = None
    if engine.classifier is not None or engine.improved_system is not None:
        per_class = {}
        correct = 0
        class_total = 0
        f1_values = []
        for cls in ARRHYTHMIA_CLASSES:
            tp = class_confusion[cls][cls]
            fp = sum(class_confusion[other][cls] for other in ARRHYTHMIA_CLASSES if other != cls)
            fn = sum(
                value
                for predicted_name, value in class_confusion[cls].items()
                if predicted_name != cls
            )
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

    result_data = {
        "dataset": {
            "id": dataset["id"],
            "name": dataset["name"],
            "physionet_slug": dataset["physionet_slug"],
            "lead_name": lead_name,
            "sampling_rate_hz": dataset["sampling_rate_hz"],
            "validation_kind": dataset["validation_kind"],
            "description": dataset["description"],
            "warning": dataset["warning"],
            "mode": req.mode,
        },
        "threshold": float(selected_threshold),
        "default_threshold": float(decision_threshold),
        "optimized_threshold": sweep["best"]["threshold"],
        "threshold_optimization": {
            "enabled": req.optimize_threshold,
            "objective": "balanced_accuracy",
            "interpretation": (
                "Exploratory only: this threshold was selected and measured "
                "on the same requested labels. Use grouped validation for an "
                "unbiased research estimate."
            ),
            "selected": sweep["best"] if req.optimize_threshold else sweep["default"],
            "default": sweep["default"],
            "best": sweep["best"],
            "candidates": sweep["candidates"],
        },
        "records": record_summaries,
        "n_beats": processed,
        "class_support": dict(per_class_support),
        "detection": selected_detection,
        "per_record_detection": per_record_detection,
        "abnormal_by_symbol": abnormal_by_symbol,
        "classifier": classifier_metrics,
        "examples": score_rows,
    }
    _save_evaluation_result(result_data)
    return OkResponse(ok=True, data=result_data)
