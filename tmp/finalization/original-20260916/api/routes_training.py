"""Training endpoints.

POST /api/training/start  -> queues a training run, returns run_id + model_id.
GET  /api/training/{run_id} -> gets current status.
POST /api/training/{run_id}/stop -> stops the run.
GET  /api/training -> lists runs.

Training runs in a background thread. Progress events are emitted through a
queue that the WebSocket service drains and forwards to subscribed clients.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import asdict
from typing import Dict

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy import text

from app.api.schemas import TrainRequest, OkResponse, ClassifierTrainRequest
from app.api.routes_auth import require_user
from app.core.config import settings
from app.db.session import get_db
from app.ml.training import TrainConfig, train, TrainResult
from app.ml.model import build_model
from app.ml.train_classifier import ClassifierTrainConfig, train_classifier, ClassifierTrainResult
from app.ml.classifier import ARRHYTHMIA_CLASSES, ECGClassifier, build_classifier

router = APIRouter(prefix="/api/training", tags=["training"])


# In-memory run registry (good enough for single-process FastAPI)
_RUNS: Dict[str, dict] = {}
_STOP_FLAGS: Dict[str, threading.Event] = {}


def _persist_progress(run_id: str, force: bool = False) -> None:
    """Persist in-memory progress so browser refreshes can resume the UI."""
    run = _RUNS.get(run_id)
    if not run:
        return
    now = time.time()
    if not force and now - run.get("last_persist_at", 0.0) < 2.0:
        return
    run["last_persist_at"] = now
    with get_db() as db:
        db.execute(text("""
            UPDATE TrainingRun
            SET logsJson = :logs, status = :status
            WHERE id = :id
        """), {
            "logs": json.dumps(run["progress"][-500:]),
            "status": run["status"],
            "id": run_id,
        })
        db.commit()


@router.get("/architectures", response_model=OkResponse)
def get_current_architectures(user=Depends(require_user)):
    """Return architecture summaries from the current Python model code."""
    auto_cfg = TrainConfig()
    autoencoder = build_model(
        input_length=settings.WINDOW_SAMPLES,
        skip_scale=auto_cfg.skip_scale,
    )
    classifier = ECGClassifier(
        encoder_state_dict=dict(autoencoder.state_dict()),
        input_length=settings.WINDOW_SAMPLES,
    )
    return OkResponse(ok=True, data={
        "autoencoder": autoencoder.architecture_summary(),
        "classifier": classifier.architecture_summary(),
    })


@router.post("/start", response_model=OkResponse)
def start_training(req: TrainRequest, user=Depends(require_user)):
    user_id, _ = user
    preview_cfg = TrainConfig()
    current_arch = build_model(
        input_length=settings.WINDOW_SAMPLES,
        skip_scale=preview_cfg.skip_scale,
    ).architecture_summary()
    with get_db() as db:
        # Create model version row first (status='training')
        model_id = secrets.token_hex(12)
        version = f"v{int(time.time())}"
        config_json = json.dumps(req.model_dump())
        db.execute(text("""
            INSERT INTO ModelVersion
                (id, name, version, description, architecture, parameters, latentDim,
                 compressionRatio, status, modelPath, threshold, thresholdK,
                 configJson, createdAt, updatedAt)
            VALUES
                (:id, :name, :version, :desc, :arch, :params, :latent,
                 :cr, 'training', NULL, NULL, :k, :cfg,
                 datetime('now'), datetime('now'))
        """), {
            "id": model_id, "name": req.model_name, "version": version,
            "desc": req.description,
            "arch": current_arch.get("name", "Autoencoder"),
            "params": int(current_arch.get("total_parameters", 0)),
            "latent": int(current_arch.get("latent_shape", [0, 0])[0]),
            "cr": float(current_arch.get("compression_ratio", 0.0)),
            "k": req.threshold_k, "cfg": config_json,
        })

        run_id = secrets.token_hex(12)
        db.execute(text("""
            INSERT INTO TrainingRun
                (id, modelId, userId, status, datasetName, configJson,
                 startedAt, completedAt, finalLoss, epochsRun, logsJson)
            VALUES
                (:id, :mid, :uid, 'queued', :ds, :cfg,
                 datetime('now'), NULL, NULL, NULL, '[]')
        """), {
            "id": run_id, "mid": model_id, "uid": user_id,
            "ds": req.dataset_name, "cfg": config_json,
        })
        db.commit()

    # Run training in background
    stop_flag = threading.Event()
    _STOP_FLAGS[run_id] = stop_flag
    _RUNS[run_id] = {
        "run_id": run_id,
        "user_id": user_id,
        "model_id": model_id,
        "status": "queued",
        "progress": [],
        "final_result": None,
        "error": None,
        "run_type": "autoencoder",
        "last_persist_at": 0.0,
    }

    def _bg():
        cfg = TrainConfig(
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            latent_channels=req.latent_channels,
            kernel_size=req.kernel_size,
            dropout=req.dropout,
            dataset_name=req.dataset_name,
            uploaded_dataset_id=req.uploaded_dataset_id or "",
            max_records=req.max_records,
            duration_per_record=req.duration_per_record,
            threshold_k=req.threshold_k,
            use_ecg_qc=req.use_ecg_qc,
            min_quality=req.min_quality,
        )

        def on_progress(ev: dict):
            _RUNS[run_id]["progress"].append(ev)
            _RUNS[run_id]["status"] = "running"
            if ev.get("type") != "batch":
                try:
                    _persist_progress(run_id, force=ev.get("type") in {"init", "qc", "epoch"})
                except Exception:
                    pass

        try:
            result: TrainResult = train(cfg, on_progress=on_progress,
                                        should_stop=lambda: stop_flag.is_set())
            _RUNS[run_id]["progress"].append({
                "type": "done",
                "status": "completed" if not stop_flag.is_set() else "stopped",
                "final_result": asdict(result),
            })
            _RUNS[run_id]["final_result"] = asdict(result)
            _RUNS[run_id]["status"] = "completed" if not stop_flag.is_set() else "stopped"
            _persist_progress(run_id, force=True)

            # Persist final state to DB
            with get_db() as db:
                db.execute(text("""
                    UPDATE ModelVersion
                    SET status = 'ready', modelPath = :mp, threshold = :thr,
                        thresholdK = :k, architecture = :arch, parameters = :params,
                        latentDim = :latent, compressionRatio = :cr,
                        updatedAt = datetime('now')
                    WHERE id = :id
                """), {
                    "mp": result.model_path, "thr": result.threshold,
                    "k": result.threshold_k,
                    "arch": result.architecture.get("name", "Dilated-U-Net-Autoencoder"),
                    "params": int(result.architecture.get("total_parameters", 0)),
                    "latent": int(result.architecture.get("latent_shape", [0, 0])[0]),
                    "cr": float(result.architecture.get("compression_ratio", 0.0)),
                    "id": model_id,
                })
                db.execute(text("""
                    UPDATE TrainingRun
                    SET status = :st, completedAt = datetime('now'),
                        finalLoss = :fl, epochsRun = :er,
                        logsJson = :logs
                    WHERE id = :id
                """), {
                    "st": _RUNS[run_id]["status"],
                    "fl": result.final_loss,
                    "er": result.epochs_run,
                    "logs": json.dumps(_RUNS[run_id]["progress"]),
                    "id": run_id,
                })

                # Insert per-epoch metrics
                for h in result.history:
                    for metric_name, value in (
                        ("train_loss", h["train_loss"]),
                        ("val_loss", h["val_loss"]),
                    ):
                        db.execute(text("""
                            INSERT INTO ModelMetric (id, modelId, metricName, metricValue, epoch, createdAt)
                            VALUES (:id, :mid, :mn, :mv, :ep, datetime('now'))
                        """), {
                            "id": secrets.token_hex(12),
                            "mid": model_id, "mn": metric_name,
                            "mv": value, "ep": h["epoch"],
                        })
                # Also store threshold + final summary metrics
                db.execute(text("""
                    INSERT INTO ModelMetric (id, modelId, metricName, metricValue, epoch, createdAt)
                    VALUES (:id, :mid, 'threshold', :v, NULL, datetime('now'))
                """), {"id": secrets.token_hex(12), "mid": model_id, "v": result.threshold})
                db.execute(text("""
                    INSERT INTO ModelMetric (id, modelId, metricName, metricValue, epoch, createdAt)
                    VALUES (:id, :mid, 'final_val_loss', :v, NULL, datetime('now'))
                """), {"id": secrets.token_hex(12), "mid": model_id, "v": result.val_loss})

                db.commit()

        except Exception as e:
            _RUNS[run_id]["status"] = "failed"
            _RUNS[run_id]["error"] = str(e)
            _RUNS[run_id]["progress"].append({"type": "error", "message": str(e)})
            _persist_progress(run_id, force=True)
            with get_db() as db:
                db.execute(text("""
                    UPDATE TrainingRun
                    SET status = 'failed', completedAt = datetime('now')
                    WHERE id = :id
                """), {"id": run_id})
                db.execute(text("""
                    UPDATE ModelVersion SET status = 'failed', updatedAt = datetime('now')
                    WHERE id = :id
                """), {"id": model_id})
                db.commit()

    threading.Thread(target=_bg, daemon=True).start()

    return OkResponse(ok=True, message="Training started", data={
        "run_id": run_id, "model_id": model_id,
    })


@router.get("/{run_id}", response_model=OkResponse)
def get_run_status(run_id: str, user=Depends(require_user)):
    user_id, _ = user
    run = _RUNS.get(run_id)
    if run and run.get("user_id") != user_id:
        raise HTTPException(403, "Not authorized to access this training run")
    if not run:
        # Try DB
        with get_db() as db:
            row = db.execute(text("""
                SELECT id, modelId, status, startedAt, completedAt, finalLoss, epochsRun, logsJson, configJson
                FROM TrainingRun WHERE id = :id AND userId = :uid
            """), {"id": run_id, "uid": user_id}).fetchone()
        if not row:
            raise HTTPException(404, "Run not found")
        config = json.loads(row[8] or "{}")
        logs = json.loads(row[7] or "[]")
        return OkResponse(ok=True, data={
            "run_id": row[0], "model_id": row[1], "status": row[2],
            "started_at": row[3], "completed_at": row[4],
            "final_loss": row[5], "epochs_run": row[6],
            "progress": logs,
            "logs": logs,
            "run_type": config.get("type", "autoencoder"),
            "config": config,
            "final_result": next((ev.get("final_result") for ev in reversed(logs)
                                  if ev.get("type") == "done" and ev.get("final_result")), None),
        })
    return OkResponse(ok=True, data={
        "run_id": run["run_id"], "model_id": run["model_id"],
        "status": run["status"],
        "progress": run["progress"][-100:],
        "final_result": run["final_result"],
        "error": run["error"],
        "run_type": run.get("run_type", "autoencoder"),
    })


@router.post("/{run_id}/stop", response_model=OkResponse)
def stop_run(run_id: str, user=Depends(require_user)):
    user_id, _ = user
    with get_db() as db:
        owner = db.execute(
            text("SELECT userId FROM TrainingRun WHERE id = :id"),
            {"id": run_id},
        ).fetchone()
    if not owner:
        raise HTTPException(404, "Run not found")
    if owner[0] != user_id:
        raise HTTPException(403, "Not authorized to stop this training run")
    flag = _STOP_FLAGS.get(run_id)
    if flag:
        flag.set()
        return OkResponse(ok=True, message="Stop requested")
    raise HTTPException(404, "Run not in active registry")


@router.get("", response_model=OkResponse)
def list_runs(user=Depends(require_user)):
    user_id, _ = user
    with get_db() as db:
        rows = db.execute(text("""
            SELECT id, modelId, status, datasetName, startedAt, completedAt, finalLoss, epochsRun, configJson
            FROM TrainingRun WHERE userId = :uid
            ORDER BY startedAt DESC
        """), {"uid": user_id}).fetchall()
    runs = []
    for r in rows:
        config = json.loads(r[8] or "{}")
        runs.append({
            "id": r[0], "model_id": r[1], "status": r[2], "dataset": r[3],
            "started_at": r[4], "completed_at": r[5],
            "final_loss": r[6], "epochs_run": r[7],
            "run_type": config.get("type", "autoencoder"),
            "config": config,
        })
    return OkResponse(ok=True, data={"runs": runs})


def consume_progress(run_id: str) -> list:
    """Used by the WebSocket service to drain progress events for a run."""
    run = _RUNS.get(run_id)
    if not run:
        return []
    # Return all events (caller tracks index)
    return run["progress"]


def get_run_status_pub(run_id: str) -> dict | None:
    return _RUNS.get(run_id)


# ============================================================================
# Classifier training (Model 2)
# ============================================================================

@router.post("/start-classifier", response_model=OkResponse)
def start_classifier_training(req: ClassifierTrainRequest, user=Depends(require_user)):
    """Start training the arrhythmia classifier (Model 2).

    Uses transfer learning: loads the encoder from a trained Model 1
    (autoencoder) and trains a classification head on MIT-BIH Arrhythmia
    Database with beat-level annotations.
    """
    user_id, _ = user

    # Get the encoder model path from the Model 1
    with get_db() as db:
        row = db.execute(text("""
            SELECT id, modelPath, status FROM ModelVersion WHERE id = :id
        """), {"id": req.encoder_model_id}).fetchone()
        if not row:
            raise HTTPException(404, "Encoder model not found")
        if row[2] != "ready":
            raise HTTPException(400, f"Encoder model is not ready (status: {row[2]})")
        encoder_path = row[1]
        if not encoder_path:
            raise HTTPException(400, "Encoder model has no saved .pt file")

    # Create a ModelVersion row for the classifier
    model_id = secrets.token_hex(12)
    version = f"clf-v{int(time.time())}"
    current_classifier_arch = build_classifier(
        encoder_path,
        input_length=settings.WINDOW_SAMPLES,
        hidden_dim=req.hidden_dim,
        dropout=req.dropout,
        freeze_encoder=req.freeze_encoder,
        use_raw_branch=True,
    ).architecture_summary()
    config_json = json.dumps({
        "type": "classifier",
        "encoder_model_id": req.encoder_model_id,
        "classifier_name": req.classifier_name,
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "learning_rate": req.learning_rate,
        "max_records": req.max_records,
        "hidden_dim": req.hidden_dim,
        "dropout": req.dropout,
        "freeze_encoder": req.freeze_encoder,
        "encoder_learning_rate": req.encoder_learning_rate,
        "focal_gamma": req.focal_gamma,
        "augment": req.augment,
    })

    with get_db() as db:
        db.execute(text("""
            INSERT INTO ModelVersion
                (id, name, version, description, architecture, parameters, latentDim,
                 compressionRatio, status, modelPath, threshold, thresholdK,
                 configJson, createdAt, updatedAt)
            VALUES
                (:id, :name, :version, :desc, :arch, :params, :latent,
                 0.0, 'training', NULL, NULL, 0, :cfg,
                 datetime('now'), datetime('now'))
        """), {
            "id": model_id, "name": req.classifier_name, "version": version,
            "desc": req.description or "Arrhythmia classifier (transfer learning)",
            "arch": current_classifier_arch.get("name", "ECG-Classifier"),
            "params": int(current_classifier_arch.get("total_parameters", 0)),
            "latent": int(current_classifier_arch.get("latent_channels", 0)),
            "cfg": config_json,
        })

        run_id = secrets.token_hex(12)
        db.execute(text("""
            INSERT INTO TrainingRun
                (id, modelId, userId, status, datasetName, configJson,
                 startedAt, completedAt, finalLoss, epochsRun, logsJson)
            VALUES
                (:id, :mid, :uid, 'queued', 'mit-bih-arrhythmia-annotated', :cfg,
                 datetime('now'), NULL, NULL, NULL, '[]')
        """), {
            "id": run_id, "mid": model_id, "uid": user_id,
            "cfg": config_json,
        })
        db.commit()

    # Run classifier training in background
    stop_flag = threading.Event()
    _STOP_FLAGS[run_id] = stop_flag
    _RUNS[run_id] = {
        "run_id": run_id,
        "user_id": user_id,
        "model_id": model_id,
        "status": "queued",
        "progress": [],
        "final_result": None,
        "error": None,
        "run_type": "classifier",
        "last_persist_at": 0.0,
    }

    def _bg():
        cfg = ClassifierTrainConfig(
            encoder_model_path=encoder_path,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            max_records=req.max_records,
            hidden_dim=req.hidden_dim,
            dropout=req.dropout,
            freeze_encoder=req.freeze_encoder,
            encoder_learning_rate=req.encoder_learning_rate,
            focal_gamma=req.focal_gamma,
            augment=req.augment,
        )

        def on_progress(ev: dict):
            _RUNS[run_id]["progress"].append(ev)
            _RUNS[run_id]["status"] = "running"
            if ev.get("type") != "batch":
                try:
                    _persist_progress(run_id, force=ev.get("type") in {"model_built", "init", "epoch", "evaluation"})
                except Exception:
                    pass

        try:
            result: ClassifierTrainResult = train_classifier(
                cfg, on_progress=on_progress,
                should_stop=lambda: stop_flag.is_set()
            )
            result_dict = asdict(result)
            _RUNS[run_id]["progress"].append({
                "type": "done",
                "status": "completed" if not stop_flag.is_set() else "stopped",
                "final_result": result_dict,
            })
            _RUNS[run_id]["final_result"] = result_dict
            _RUNS[run_id]["status"] = "completed" if not stop_flag.is_set() else "stopped"
            _persist_progress(run_id, force=True)

            with get_db() as db:
                db.execute(text("""
                    UPDATE ModelVersion
                    SET status = 'ready', modelPath = :mp,
                        architecture = :arch, parameters = :params,
                        latentDim = :latent, compressionRatio = :cr,
                        updatedAt = datetime('now')
                    WHERE id = :id
                """), {
                    "mp": result.model_path,
                    "arch": result.architecture.get("name", "ECG-Classifier"),
                    "params": int(result.architecture.get("total_parameters", 0)),
                    "latent": int(result.architecture.get("latent_channels", 0)),
                    "cr": 0.0,
                    "id": model_id,
                })

                db.execute(text("""
                    UPDATE TrainingRun
                    SET status = :st, completedAt = datetime('now'),
                        finalLoss = :fl, epochsRun = :er,
                        logsJson = :logs
                    WHERE id = :id
                """), {
                    "st": _RUNS[run_id]["status"],
                    "fl": result.final_loss,
                    "er": result.epochs_run,
                    "logs": json.dumps(_RUNS[run_id]["progress"]),
                    "id": run_id,
                })

                # Store metrics
                for h in result.history:
                    for metric_name, value in (("train_acc", h["train_acc"]),
                                                ("val_acc", h["val_acc"]),
                                                ("val_macro_f1", h.get("val_macro_f1", 0.0)),
                                                ("train_loss", h["train_loss"]),
                                                ("val_loss", h["val_loss"])):
                        db.execute(text("""
                            INSERT INTO ModelMetric (id, modelId, metricName, metricValue, epoch, createdAt)
                            VALUES (:id, :mid, :mn, :mv, :ep, datetime('now'))
                        """), {
                            "id": secrets.token_hex(12), "mid": model_id,
                            "mn": metric_name, "mv": value, "ep": h["epoch"],
                        })

                # Store test accuracy and confusion matrix as special metrics
                db.execute(text("""
                    INSERT INTO ModelMetric (id, modelId, metricName, metricValue, epoch, createdAt)
                    VALUES (:id, :mid, 'test_accuracy', :v, NULL, datetime('now'))
                """), {"id": secrets.token_hex(12), "mid": model_id, "v": result.test_accuracy})
                db.execute(text("""
                    INSERT INTO ModelMetric (id, modelId, metricName, metricValue, epoch, createdAt)
                    VALUES (:id, :mid, 'test_macro_f1', :v, NULL, datetime('now'))
                """), {"id": secrets.token_hex(12), "mid": model_id, "v": result.test_macro_f1})

                db.commit()

        except Exception as e:
            import traceback
            _RUNS[run_id]["status"] = "failed"
            _RUNS[run_id]["error"] = str(e)
            _RUNS[run_id]["progress"].append({"type": "error", "message": str(e)})
            _persist_progress(run_id, force=True)
            traceback.print_exc()
            with get_db() as db:
                db.execute(text("""
                    UPDATE TrainingRun
                    SET status = 'failed', completedAt = datetime('now')
                    WHERE id = :id
                """), {"id": run_id})
                db.execute(text("""
                    UPDATE ModelVersion SET status = 'failed', updatedAt = datetime('now')
                    WHERE id = :id
                """), {"id": model_id})
                db.commit()

    threading.Thread(target=_bg, daemon=True).start()

    return OkResponse(ok=True, message="Classifier training started", data={
        "run_id": run_id,
        "model_id": model_id,
        "class_names": ARRHYTHMIA_CLASSES,
    })
