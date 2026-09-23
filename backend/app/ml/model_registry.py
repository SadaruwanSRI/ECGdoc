"""Register shipped research artifacts in the application's model catalogue."""
from __future__ import annotations

import json
import hashlib
import joblib
from pathlib import Path

import torch
from sqlalchemy import text

from app.core.config import settings
from app.db.session import get_db


FINAL_MODEL_ID = "final-mlii-corrected-20260916"
FINAL_MODEL_VERSION = "final-mlii-local-window-20260916"
FINAL_MODEL_FILENAME = "corrected_mlii_temporal_holdout.joblib"
NSRDB_AUTOENCODER_ID = "nsrdb-primary-autoencoder"
NSRDB_AUTOENCODER_FILENAME = "model_nsrdb_primary_healthy.pt"
LEGACY_MODEL_IDS = (
    "5ce13c166ab4d50453df8504",
    "df7d4fab5867c5290b1482a3",
    "improved-hierarchical-v1",
    "full-subject-hierarchical-v2",
)


def ensure_final_model_registered() -> bool:
    """Register the final one-channel hierarchy and its temporal-test metrics."""
    model_path = settings.MODEL_DIR / FINAL_MODEL_FILENAME
    if not model_path.exists():
        print(f"[models] Final MLII model artifact not found: {model_path}")
        return False
    autoencoder_path = settings.MODEL_DIR / NSRDB_AUTOENCODER_FILENAME
    if not autoencoder_path.exists():
        print(f"[models] NSRDB autoencoder artifact not found: {autoencoder_path}")
        return False
    autoencoder_checkpoint = torch.load(
        autoencoder_path, map_location="cpu", weights_only=False
    )
    autoencoder_threshold = float(autoencoder_checkpoint["threshold"])
    autoencoder_config = dict(autoencoder_checkpoint.get("config", {}))

    result_path = settings.PROJECT_ROOT / "Final_Report_Research/experiments/corrected_temporal_holdout.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != result["artifacts"]["model_sha256"]:
        raise ValueError("Final model and evaluation provenance do not match")
    bundle = joblib.load(model_path)
    final_threshold = float(bundle["binary_threshold"])
    config = {
        "type": "classifier", "system_kind": "final_mlii_temporal_holdout",
        "lead_count": 1, "required_lead": "MLII", "requires_r_peak_alignment": True,
        "requires_rr_context": True, "clinical_use": False,
        "preprocessing_version": bundle["preprocessing_version"],
        "paired_autoencoder_id": NSRDB_AUTOENCODER_ID,
        "development_threshold": final_threshold,
        "split": result["protocol"]["classifier_training"],
        "evaluation_scope": result["protocol"]["generalization_claim"],
        "model_sha256": result["artifacts"]["model_sha256"],
    }
    h = result["headline"]
    metrics = {
        "test_binary_accuracy": h["binary_accuracy"],
        "test_binary_balanced_accuracy": h["binary_balanced_accuracy"],
        "test_subtype_accuracy": h["subtype_accuracy"],
        "test_subtype_macro_f1": h["subtype_macro_f1"],
        "test_whole_system_accuracy": h["whole_system_accuracy"],
        "test_whole_system_macro_f1": h["whole_system_macro_f1"],
        "subject_macro_accuracy": h["whole_system_subject_macro_accuracy"],
        "subject_clustered_ci_lower": h["whole_system_subject_clustered_ci_95"][0],
        "subject_clustered_ci_upper": h["whole_system_subject_clustered_ci_95"][1],
    }

    # Keep historical models and metrics: existing sessions can refer to them.
    with get_db() as db:
        autoencoder_exists = db.execute(
            text("SELECT id FROM ModelVersion WHERE id = :id"),
            {"id": NSRDB_AUTOENCODER_ID},
        ).fetchone()
        autoencoder_values = {
            "id": NSRDB_AUTOENCODER_ID,
            "name": "MIT-BIH Normal-Sinus Autoencoder",
            "version": "nsrdb-primary-20260802",
            "description": (
                "Frozen normal-only autoencoder developed from the complete "
                "first stored ECG channel of all 18 MIT-BIH Normal Sinus "
                "Rhythm records. The WFDB header calls this channel ECG1; its "
                "anatomical placement is not specified. It supplies four "
                "residual features to the final MLII hierarchy."
            ),
            "architecture": "Dilated-U-Net-Autoencoder",
            "path": str(Path(autoencoder_path).resolve()),
            "threshold": autoencoder_threshold,
            "config": json.dumps(autoencoder_config),
        }
        if autoencoder_exists:
            db.execute(
                text(
                    """
                    UPDATE ModelVersion
                    SET name=:name, version=:version, description=:description,
                        architecture=:architecture, modelPath=:path,
                        threshold=:threshold, configJson=:config,
                        status='ready', updatedAt=datetime('now')
                    WHERE id=:id
                    """
                ),
                autoencoder_values,
            )
        else:
            db.execute(
                text(
                    """
                    INSERT INTO ModelVersion
                        (id,name,version,description,architecture,parameters,
                         latentDim,compressionRatio,status,modelPath,threshold,
                         thresholdK,configJson,createdAt,updatedAt)
                    VALUES
                        (:id,:name,:version,:description,:architecture,0,0,1.0,
                         'ready',:path,:threshold,3.0,:config,
                         datetime('now'),datetime('now'))
                    """
                ),
                autoencoder_values,
            )
        exists = db.execute(
            text("SELECT id FROM ModelVersion WHERE id = :id"),
            {"id": FINAL_MODEL_ID},
        ).fetchone()
        if not exists:
            db.execute(
                text(
                    """
                    INSERT INTO ModelVersion
                        (id, name, version, description, architecture, parameters,
                         latentDim, compressionRatio, status, modelPath, threshold,
                         thresholdK, configJson, createdAt, updatedAt)
                    VALUES
                        (:id, :name, :version, :description, :architecture, 0,
                         0, 1.0, 'ready', :path, :threshold,
                         NULL, :config, datetime('now'), datetime('now'))
                    """
                ),
                {
                    "id": FINAL_MODEL_ID,
                    "name": "Final MLII ECG System (corrected September 2026)",
                    "version": FINAL_MODEL_VERSION,
                    "description": (
                        "Beat-aligned detector trained on the first 80% of all "
                        "46 MLII-containing MIT-BIH records and evaluated on "
                        "their final 20%. Both supervised stages use MLII only. The test "
                        "test segment was observed in earlier development; this is a retrospective rerun."
                    ),
                    "architecture": "MLII-ExtraTrees-Hierarchy",
                    "path": str(Path(model_path).resolve()),
                    "threshold": final_threshold,
                    "config": json.dumps(config),
                },
            )
        else:
            db.execute(
                text(
                    """
                    UPDATE ModelVersion
                    SET name = :name, version = :version,
                        description = :description, architecture = :architecture,
                        modelPath = :path, threshold = :threshold,
                        configJson = :config, updatedAt = datetime('now')
                    WHERE id = :id
                    """
                ),
                {
                    "id": FINAL_MODEL_ID,
                    "name": "Final MLII ECG System (corrected September 2026)",
                    "version": FINAL_MODEL_VERSION,
                    "description": (
                        "Beat-aligned one-lead system trained on the first 80% "
                        "of all 46 MLII-containing MIT-BIH records and evaluated "
                        "on the final 20%."
                    ),
                    "architecture": "MLII-ExtraTrees-Hierarchy",
                    "path": str(Path(model_path).resolve()),
                    "threshold": final_threshold,
                    "config": json.dumps(config),
                },
            )

        for metric_name, metric_value in metrics.items():
            metric_id = f"{FINAL_MODEL_ID}-{metric_name}"
            metric_exists = db.execute(
                text("SELECT id FROM ModelMetric WHERE id = :id"),
                {"id": metric_id},
            ).fetchone()
            if metric_exists:
                db.execute(
                    text(
                        "UPDATE ModelMetric SET metricValue = :value "
                        "WHERE id = :id"
                    ),
                    {"id": metric_id, "value": metric_value},
                )
            else:
                db.execute(
                    text(
                        """
                        INSERT INTO ModelMetric
                            (id, modelId, metricName, metricValue, epoch, createdAt)
                        VALUES
                            (:id, :model_id, :name, :value, NULL, datetime('now'))
                        """
                    ),
                    {
                        "id": metric_id,
                        "model_id": FINAL_MODEL_ID,
                        "name": metric_name,
                        "value": metric_value,
                    },
                )
        db.commit()
    return True
