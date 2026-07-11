"""FastAPI application entry point.

Run with:
    cd /home/z/my-project/backend
    /home/z/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.session import health
from app.api.routes_auth import router as auth_router
from app.api.routes_models import router as models_router
from app.api.routes_training import router as training_router
from app.api.routes_sessions import router as sessions_router
from app.api.routes_reports import router as reports_router
from app.api.routes_datasets import router as datasets_router
from app.api.routes_evaluation import router as evaluation_router
from app.api.routes_arduino import router as arduino_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"[fastapi] Starting ECG Anomaly Detection backend v1.0.0")
    print(f"[fastapi] Database: {settings.SQLALCHEMY_URL}")
    print(f"[fastapi] Storage dir: {settings.STORAGE_DIR}")
    ok = health()
    print(f"[fastapi] DB health: {'OK' if ok else 'FAILED'}")
    yield
    # Shutdown
    print("[fastapi] Shutting down")


app = FastAPI(
    title="ECG Anomaly Detection API",
    description="Unsupervised deep-learning ECG anomaly detection backend. "
                "FastAPI + PyTorch + PostgreSQL-ready.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["*"],  # permissive for sandbox
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def healthz():
    return {"ok": True, "db": health(), "version": "1.0.0"}


# Register routers
app.include_router(auth_router)
app.include_router(models_router)
app.include_router(training_router)
app.include_router(sessions_router)
app.include_router(reports_router)
app.include_router(datasets_router)
app.include_router(evaluation_router)
app.include_router(arduino_router)


@app.get("/")
def root():
    return {
        "name": "ECG Anomaly Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
