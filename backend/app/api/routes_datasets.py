"""Dataset enumeration + upload endpoints."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from app.api.schemas import OkResponse
from app.api.routes_auth import require_user
from app.core.config import settings
from app.ml.data import NSRDB_RECORDS, MITDB_RECORDS, list_uploaded_datasets

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=OkResponse)
def list_datasets(user=Depends(require_user)):
    """List available datasets (built-in + user-uploaded)."""
    uploaded = list_uploaded_datasets()
    uploaded_entries = [
        {
            "id": f"uploaded:{u['id']}",
            "name": f"Uploaded: {u['name']}",
            "description": f"{u['total_records']} records ({u['dat_files']} .dat, {u['csv_files']} .csv, {u['hea_files']} .hea)",
            "records": [],
        }
        for u in uploaded
    ]
    return OkResponse(ok=True, data={
        "training": [
            {
                "id": "synthetic",
                "name": "Synthetic Normal Sinus Rhythm",
                "description": "Procedurally generated normal ECG - always available, no download required.",
                "records": [],
            },
            {
                "id": "mit-bih-nsr",
                "name": "MIT-BIH Normal Sinus Rhythm DB",
                "description": "18 long-term ECG recordings from healthy subjects (PhysioNet). Will use local files if present in storage/datasets/nsrdb/, otherwise downloads.",
                "records": NSRDB_RECORDS,
            },
            *uploaded_entries,
        ],
        "testing": [
            {
                "id": "mit-bih-arrhythmia",
                "name": "MIT-BIH Arrhythmia Database",
                "description": "48 half-hour ECG recordings with annotated arrhythmias (PhysioNet). Will use local files if present in storage/datasets/mitdb/, otherwise downloads.",
                "records": MITDB_RECORDS,
            },
            {
                "id": "synthetic-arrhythmia",
                "name": "Synthetic Arrhythmia (PVCs)",
                "description": "Procedurally generated ECG with premature ventricular contractions every 3rd beat.",
                "records": [],
            },
            {
                "id": "arduino",
                "name": "Arduino Nano + AD8232",
                "description": "Live 128 Hz ECG acquisition from an Arduino Nano serial port.",
                "records": [],
            },
        ],
        "uploaded": uploaded,
    })


@router.get("/uploads", response_model=OkResponse)
def list_uploads(user=Depends(require_user)):
    """List all uploaded datasets with file counts."""
    return OkResponse(ok=True, data={"uploads": list_uploaded_datasets()})


@router.post("/upload", response_model=OkResponse)
async def upload_dataset(
    dataset_id: str = Form(..., description="Name for this dataset (e.g. 'my-mitbih')"),
    files: list[UploadFile] = File(..., description=".dat, .hea, and/or .csv files"),
    user=Depends(require_user),
):
    """Upload ECG dataset files (.dat + .hea pairs, or .csv files).

    Files are saved to storage/datasets/uploads/{dataset_id}/.
    After upload, select "uploaded:{dataset_id}" as the training dataset.

    Accepted file types:
    - .dat + .hea pairs (WFDB format - the native MIT-BIH format)
    - .csv files (single column of ECG values, or two columns: time, value)

    The dataset_id is used as the folder name and must be unique.
    """
    if not dataset_id or not dataset_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "dataset_id must contain only letters, numbers, hyphens, and underscores")

    upload_dir = settings.DATASET_DIR / "uploads" / dataset_id
    if upload_dir.exists():
        raise HTTPException(400, f"Dataset '{dataset_id}' already exists. Use a different name or delete it first.")

    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for f in files:
        if not f.filename:
            continue
        # Sanitize filename - only allow .dat, .hea, .csv extensions
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in (".dat", ".hea", ".csv"):
            continue
        # Sanitize the filename (no path traversal)
        safe_name = os.path.basename(f.filename)
        dest = upload_dir / safe_name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_files.append(safe_name)

    if not saved_files:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(400, "No valid files uploaded. Accepted: .dat, .hea, .csv")

    # Count files by type
    dat_count = len(list(upload_dir.glob("*.dat")))
    csv_count = len(list(upload_dir.glob("*.csv")))
    hea_count = len(list(upload_dir.glob("*.hea")))

    return OkResponse(ok=True, message=f"Uploaded {len(saved_files)} files", data={
        "id": dataset_id,
        "name": dataset_id,
        "path": str(upload_dir),
        "dat_files": dat_count,
        "csv_files": csv_count,
        "hea_files": hea_count,
        "total_records": dat_count + csv_count,
        "saved_files": saved_files,
    })


@router.delete("/uploads/{dataset_id}", response_model=OkResponse)
def delete_upload(dataset_id: str, user=Depends(require_user)):
    """Delete an uploaded dataset and all its files."""
    if not dataset_id or not dataset_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "Invalid dataset_id")

    upload_dir = settings.DATASET_DIR / "uploads" / dataset_id
    if not upload_dir.exists():
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")

    shutil.rmtree(upload_dir, ignore_errors=True)
    return OkResponse(ok=True, message=f"Dataset '{dataset_id}' deleted")


# ============================================================================
# Health information endpoints
# ============================================================================

@router.get("/health-info/{arrhythmia_type}", response_model=OkResponse)
def get_health_info_endpoint(arrhythmia_type: str, user=Depends(require_user)):
    """Get health information for a specific arrhythmia type."""
    from app.ml.health_info import get_health_info
    info = get_health_info(arrhythmia_type.upper())
    return OkResponse(ok=True, data=info)


@router.get("/health-info", response_model=OkResponse)
def list_health_info(user=Depends(require_user)):
    """List all arrhythmia types with their health information."""
    from app.ml.health_info import list_all_arrhythmia_types
    return OkResponse(ok=True, data={"types": list_all_arrhythmia_types()})
