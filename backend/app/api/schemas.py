"""Pydantic schemas for the API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------- Auth ----------

class UserCreate(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    name: Optional[str] = Field(default=None, max_length=200)
    password: str = Field(min_length=8, max_length=1024)


class UserLogin(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


class UserOut(BaseModel):
    id: str
    email: str
    name: Optional[str]
    role: str


# ---------- Models ----------

class TrainRequest(BaseModel):
    model_name: str = "ecg-ae-v1"
    description: Optional[str] = None
    epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-3
    latent_channels: int = 1024
    kernel_size: int = 7
    dropout: float = 0.2
    dataset_name: str = "mitbih-nsrdb"
    uploaded_dataset_id: Optional[str] = None  # folder name under storage/datasets/uploads/
    max_records: int = 18
    duration_per_record: float = 0.0
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
    max_records: int = 46          # how many eligible MIT-BIH MLII records to use (1-46)
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
    metrics: List[Dict[str, Any]] = Field(default_factory=list)


class MetricOut(BaseModel):
    metric_name: str
    metric_value: float
    epoch: Optional[int]


# ---------- Sessions / Live ----------

class StartSessionRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    classifier_model_id: Optional[str] = Field(default=None, max_length=200)  # optional Model 2 (classifier)
    source_type: str = Field(min_length=1, max_length=50)
    source_detail: Optional[str] = None  # record name for mit-bih
    lead_name: str = Field(default="MLII", min_length=2, max_length=10)
    chunk_seconds: float = Field(default=4.0, ge=0.5, le=60.0)
    threshold_override: Optional[float] = Field(default=None, ge=0.0)


class ArduinoSerialTestRequest(BaseModel):
    """Configuration for a short, raw Arduino acquisition test."""
    port: str = Field(min_length=1, max_length=100)
    baud_rate: int = Field(default=115200, ge=1200, le=2_000_000)
    sample_count: int = Field(default=256, ge=16, le=2048)
    timeout_seconds: float = Field(default=8.0, ge=2.0, le=30.0)


class ModelEvaluationRequest(BaseModel):
    dataset_id: str = Field(default="mit-bih-arrhythmia", min_length=1, max_length=100)
    model_id: str = Field(min_length=1, max_length=200)
    classifier_model_id: Optional[str] = Field(default=None, max_length=200)
    records: List[str] = Field(default_factory=lambda: ["100", "101", "103"], max_length=100)
    max_beats: int = Field(default=1000, ge=1, le=1_000_000)
    lead_name: str = Field(default="MLII", min_length=1, max_length=20)
    mode: str = Field(default="offline", min_length=1, max_length=50)
    threshold_override: Optional[float] = Field(default=None, ge=0.0)
    optimize_threshold: bool = False
    r_peak_before: int = Field(default=200, ge=0, le=511)


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
