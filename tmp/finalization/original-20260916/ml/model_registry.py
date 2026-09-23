"""Register shipped research artifacts in the application's model catalogue."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from sqlalchemy import text

from app.core.config import settings
from app.db.session import get_db


FINAL_MODEL_ID = "final-mlii-temporal-holdout"
FINAL_MODEL_VERSION = "final-mlii-nsrdb-20260802"
FINAL_MODEL_FILENAME = "final_mlii_temporal_holdout.joblib"
FINAL_THRESHOLD = 0.6208333333333333
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

    config = {
        "type": "classifier",
        "system_kind": "final_mlii_temporal_holdout",
        "binary_input": (
            "MLII morphology, spectrum, seven base RR features, "
            "eight causal RR-history features, and four frozen-autoencoder "
            "reconstruction-error features"
        ),
        "class_input": (
            "the same MLII morphology, spectrum, RR context, and "
            "autoencoder residual vector used by the binary detector"
        ),
        "lead_count": 1,
        "requires_r_peak_alignment": True,
        "requires_rr_context": True,
        "threshold_objective": "balanced accuracy on the 64--80% temporal validation region",
        "development_threshold": FINAL_THRESHOLD,
        "training_records": 46,
        "training_subjects": 45,
        "required_lead": "MLII",
        "split": "first 80% of each record for development; final 20% temporal test",
        "boundary_gap_beats": 16,
        "evaluation_scope": "unseen signal from known patients, not unseen patients",
        "internal_validation": "first 64% training and 64--80% temporal validation",
        "clinical_use": False,
    }
    metrics = {
        "test_binary_accuracy": 0.9847055460263008,
        "test_binary_balanced_accuracy": 0.9822144555868357,
        "test_subtype_accuracy": 0.9520341536916123,
        "test_subtype_macro_f1": 0.9325289702749451,
        "test_whole_system_accuracy": 0.9686471936995315,
        "test_whole_system_macro_f1": 0.9302682471166553,
        "subject_macro_accuracy": 0.9689592744813958,
        "subject_clustered_ci_lower": 0.9401985279471115,
        "subject_clustered_ci_upper": 0.9904584470560533,
    }

    with get_db() as db:
        for legacy_id in LEGACY_MODEL_IDS:
            db.execute(
                text("DELETE FROM ModelMetric WHERE modelId = :model_id"),
                {"model_id": legacy_id},
            )
            db.execute(
                text("DELETE FROM ModelVersion WHERE id = :model_id"),
                {"model_id": legacy_id},
            )
        db.execute(
            text(
                "DELETE FROM ModelMetric "
                "WHERE modelId NOT IN (SELECT id FROM ModelVersion)"
            )
        )
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
                    "name": "Final MLII Temporal-Holdout ECG System",
                    "version": FINAL_MODEL_VERSION,
                    "description": (
                        "Beat-aligned detector trained on the first 80% of all "
                        "46 MLII-containing MIT-BIH records and evaluated on "
                        "their final 20%. Both supervised stages use MLII only. The test "
                        "signal was unseen, but the patients were known."
                    ),
                    "architecture": "MLII-ExtraTrees-Hierarchy",
                    "path": str(Path(model_path).resolve()),
                    "threshold": FINAL_THRESHOLD,
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
                    "name": "Final MLII Temporal-Holdout ECG System",
                    "version": FINAL_MODEL_VERSION,
                    "description": (
                        "Beat-aligned one-lead system trained on the first 80% "
                        "of all 46 MLII-containing MIT-BIH records and evaluated "
                        "on the final 20%."
                    ),
                    "architecture": "MLII-ExtraTrees-Hierarchy",
                    "path": str(Path(model_path).resolve()),
                    "threshold": FINAL_THRESHOLD,
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
