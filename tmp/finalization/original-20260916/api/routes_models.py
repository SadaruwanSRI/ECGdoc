"""Model management endpoints — list versions, get metrics, register new versions."""
from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException, Depends, Query

from app.api.schemas import ModelVersionOut, MetricOut, OkResponse
from app.api.routes_auth import require_user
from app.db.session import get_db
from sqlalchemy import text

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=OkResponse)
def list_models(user=Depends(require_user)):
    """List all model versions (newest first)."""
    with get_db() as db:
        rows = db.execute(text("""
            SELECT id, name, version, description, architecture, parameters,
                   latentDim, compressionRatio, status, modelPath, threshold,
                   thresholdK, configJson, createdAt
            FROM ModelVersion
            ORDER BY createdAt DESC
        """)).fetchall()
    models = []
    for r in rows:
        models.append({
            "id": r[0], "name": r[1], "version": r[2], "description": r[3],
            "architecture": r[4], "parameters": r[5], "latent_dim": r[6],
            "compression_ratio": r[7], "status": r[8], "model_path": r[9],
            "threshold": r[10], "threshold_k": r[11], "config_json": r[12],
            "created_at": r[13],
        })
    return OkResponse(ok=True, data={"models": models})


# IMPORTANT: /compare must be defined BEFORE /{model_id}, otherwise FastAPI
# will match "/compare" as a model_id and return "Model not found".
@router.get("/compare", response_model=OkResponse)
def compare_models(ids: str = Query(..., description="Comma-separated model IDs"),
                   user=Depends(require_user)):
    """Compare multiple model versions side-by-side — used by dashboard."""
    id_list = [x.strip() for x in ids.split(",") if x.strip()]
    if not id_list:
        raise HTTPException(400, "No model IDs provided")
    placeholders = ",".join([f":id{i}" for i in range(len(id_list))])
    params = {f"id{i}": v for i, v in enumerate(id_list)}
    with get_db() as db:
        rows = db.execute(text(f"""
            SELECT id, name, version, architecture, parameters, status,
                   threshold, createdAt
            FROM ModelVersion WHERE id IN ({placeholders})
            ORDER BY createdAt
        """), params).fetchall()
        if not rows:
            raise HTTPException(404, "No matching models found")

        # Get all metrics for these models
        mrows = db.execute(text(f"""
            SELECT modelId, metricName, metricValue, epoch
            FROM ModelMetric WHERE modelId IN ({placeholders})
            ORDER BY modelId, epoch, metricName
        """), params).fetchall()

    # Build comparison structure
    comparison = []
    metric_names = set()
    for r in rows:
        m = {
            "id": r[0], "name": r[1], "version": r[2], "architecture": r[3],
            "parameters": r[4], "status": r[5], "threshold": r[6],
            "created_at": r[7], "metrics": {},
        }
        comparison.append(m)

    id_to_model = {m["id"]: m for m in comparison}
    for mr in mrows:
        m_id, name, val, epoch = mr[0], mr[1], mr[2], mr[3]
        if m_id in id_to_model:
            if name not in id_to_model[m_id]["metrics"]:
                id_to_model[m_id]["metrics"][name] = []
            id_to_model[m_id]["metrics"][name].append({"epoch": epoch, "value": val})
            metric_names.add(name)

    return OkResponse(ok=True, data={
        "models": comparison,
        "metric_names": sorted(metric_names),
    })


@router.get("/{model_id}", response_model=OkResponse)
def get_model(model_id: str, user=Depends(require_user)):
    with get_db() as db:
        row = db.execute(text("""
            SELECT id, name, version, description, architecture, parameters,
                   latentDim, compressionRatio, status, modelPath, threshold,
                   thresholdK, configJson, createdAt
            FROM ModelVersion WHERE id = :id
        """), {"id": model_id}).fetchone()
        if not row:
            raise HTTPException(404, "Model not found")
        metrics = db.execute(text("""
            SELECT metricName, metricValue, epoch FROM ModelMetric
            WHERE modelId = :id ORDER BY epoch, metricName
        """), {"id": model_id}).fetchall()
    return OkResponse(ok=True, data={
        "model": {
            "id": row[0], "name": row[1], "version": row[2], "description": row[3],
            "architecture": row[4], "parameters": row[5], "latent_dim": row[6],
            "compression_ratio": row[7], "status": row[8], "model_path": row[9],
            "threshold": row[10], "threshold_k": row[11], "config_json": row[12],
            "created_at": row[13],
        },
        "metrics": [
            {"metric_name": m[0], "metric_value": m[1], "epoch": m[2]} for m in metrics
        ],
    })



@router.delete("/{model_id}", response_model=OkResponse)
def delete_model(model_id: str, user=Depends(require_user)):
    """Permanently delete a model version and all its metrics."""
    import os
    with get_db() as db:
        # Get model path before deleting (so we can delete the .pt file too)
        row = db.execute(text("SELECT modelPath FROM ModelVersion WHERE id = :id"),
                          {"id": model_id}).fetchone()
        if not row:
            raise HTTPException(404, "Model not found")
        model_path = row[0]
        # Delete metrics, training runs, and the model itself
        db.execute(text("DELETE FROM ModelMetric WHERE modelId = :id"),
                    {"id": model_id})
        db.execute(text("DELETE FROM TrainingRun WHERE modelId = :id"),
                    {"id": model_id})
        # Unlink model from any sessions (set modelId to NULL)
        db.execute(text("UPDATE EcgSession SET modelId = NULL WHERE modelId = :id"),
                    {"id": model_id})
        db.execute(text("DELETE FROM ModelVersion WHERE id = :id"),
                    {"id": model_id})
        db.commit()
    # Delete the .pt file from disk
    if model_path and os.path.exists(model_path):
        try:
            os.remove(model_path)
        except Exception:
            pass  # file might be locked, ignore
    return OkResponse(ok=True, message="Model deleted")
