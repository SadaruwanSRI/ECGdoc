"""Pydantic schemas for the API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------- Auth ----------

class UserCreate(BaseModel):
    email: str
    name: Optional[str] = None
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    name: Optional[str]
    role: str


# ---------- Models ----------

class TrainRequest(BaseModel):
    model_name: str = "ecg-ae-v1"
    description: Optional[str] = None
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    latent_channels: int = 1024
    kernel_size: int = 7
    dropout: float = 0.2
    dataset_name: str = "synthetic"  # "synthetic" | "mit-bih-nsr" | "uploaded"
    uploaded_dataset_id: Optional[str] = None  # folder name under storage/datasets/uploads/
    max_records: int = 4
    duration_per_record: float = 60.0
    threshold_k: float = 2.0
    # Quality control
    use_ecg_qc: bool = True       # use ecg_qc SQI-based quality classifier
    min_quality: int = 2          # 0-3; 2 = keep medium-high or better (recommended)


class ClassifierTrainRequest(BaseModel):
    """Request to train the arrhythmia classifier (Model 2)."""
    encoder_model_id: str          # ID of the Model 1 (autoencoder) to use as encoder
    classifier_name: str = "ecg-classifier-v1"
    description: Optional[str] = None
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_records: int = 48          # how many MIT-BIH Arrhythmia records to use (1-48)
    hidden_dim: int = 256
    dropout: float = 0.35
    freeze_encoder: bool = False
    encoder_learning_rate: float = 1e-4
    focal_gamma: float = 2.0
    augment: bool = True


class ModelVersionOut(BaseModel):
    id: str
    name: str
    version: str
    description: Optional[str]
    architecture: str
    parameters: int
    latent_dim: int
    compression_ratio: float
    status: str
    threshold: Optional[float]
    threshold_k: Optional[float]
    config_json: str
    created_at: str
    metrics: List[Dict[str, Any]] = []


class MetricOut(BaseModel):
    metric_name: str
    metric_value: float
    epoch: Optional[int]


# ---------- Sessions / Live ----------

class StartSessionRequest(BaseModel):
    model_id: str
    classifier_model_id: Optional[str] = None  # optional Model 2 (classifier)
    source_type: str  # "mit-bih-arrhythmia" | "arduino" | "synthetic-arrhythmia"
    source_detail: Optional[str] = None  # record name for mit-bih
    chunk_seconds: float = 4.0
    threshold_override: Optional[float] = None  # per-session sensitivity override


class ModelEvaluationRequest(BaseModel):
    model_id: str
    classifier_model_id: Optional[str] = None
    records: List[str] = Field(default_factory=lambda: ["100", "101", "102"])
    max_beats: int = 1000
    threshold_override: Optional[float] = None
    optimize_threshold: bool = False
    r_peak_before: int = 200


class AlertOut(BaseModel):
    id: str
    timestamp: str
    anomaly_score: float
    threshold: float
    severity: str
    message: str


class SessionOut(BaseModel):
    id: str
    user_id: str
    model_id: Optional[str]
    source_type: str
    source_detail: Optional[str]
    started_at: str
    ended_at: Optional[str]
    status: str
    total_beats: int
    anomaly_beats: int
    summary_json: str


# ---------- Reports ----------

class PdfReportRequest(BaseModel):
    session_id: str
    physician_name: Optional[str] = None
    notes: Optional[str] = None


# ---------- Generic ----------

class OkResponse(BaseModel):
    ok: bool
    message: Optional[str] = None
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    detail: str
