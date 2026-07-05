"""Live session endpoints — start, stop, list, get alerts."""
from __future__ import annotations

import json
import secrets
import time
from typing import Dict, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import text

from app.api.schemas import StartSessionRequest, OkResponse, SessionOut
from app.api.routes_auth import require_user
from app.core.config import settings
from app.db.session import get_db
from app.ml.inference import get_engine
from app.ml.ecg_metrics import compute_ecg_metrics
from app.ml.data import (
    synthetic_arrhythmia_ecg,
    stream_mitbih_record,
    preprocess_signal,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


# In-memory active sessions (model_id -> session state)
# This is read by the WebSocket service for live streaming.
_LIVE_SESSIONS: Dict[str, dict] = {}


@router.post("/start", response_model=OkResponse)
def start_session(req: StartSessionRequest, user=Depends(require_user)):
    user_id, _ = user

    # Load autoencoder model info from DB
    with get_db() as db:
        m = db.execute(text("""
            SELECT id, modelPath, threshold, thresholdK FROM ModelVersion
            WHERE id = :id AND status = 'ready'
        """), {"id": req.model_id}).fetchone()
        if not m:
            raise HTTPException(404, "Ready model not found")

        # Load classifier model info (optional)
        classifier_path = None
        if req.classifier_model_id:
            c = db.execute(text("""
                SELECT id, modelPath FROM ModelVersion
                WHERE id = :id AND status = 'ready'
            """), {"id": req.classifier_model_id}).fetchone()
            if not c:
                raise HTTPException(404, "Ready classifier model not found")
            classifier_path = c[1]

        session_id = secrets.token_hex(12)
        db.execute(text("""
            INSERT INTO EcgSession
                (id, userId, modelId, sourceType, sourceDetail, startedAt,
                 endedAt, status, totalBeats, anomalyBeats, summaryJson)
            VALUES
                (:id, :uid, :mid, :st, :sd, datetime('now'),
                 NULL, 'running', 0, 0, '{}')
        """), {
            "id": session_id, "uid": user_id, "mid": req.model_id,
            "st": req.source_type, "sd": req.source_detail or "",
        })
        db.commit()

    # Spin up engine
    # Spin up engine (with optional classifier for two-step inference)
    threshold = req.threshold_override if req.threshold_override is not None else m[2]
    engine = get_engine(m[1], threshold=threshold, classifier_path=classifier_path)

    # Build the source iterator based on source_type
    if req.source_type == "mit-bih-arrhythmia":
        record = req.source_detail or "100"
        try:
            # Test load to fail fast
            from app.ml.data import _download_record
            _download_record(record, "mitdb")
        except Exception as e:
            raise HTTPException(400, f"Cannot load MIT-BIH record {record}: {e}")
        source_iter = stream_mitbih_record(record, chunk_seconds=req.chunk_seconds)
    elif req.source_type == "synthetic-arrhythmia":
        # Generate a 5-minute arrhythmia signal and chunk it
        sig = synthetic_arrhythmia_ecg(duration_s=300.0, seed=42)
        from app.ml.data import windowize
        # Chunk in 4s windows
        chunk_size = int(req.chunk_seconds * settings.SAMPLING_RATE_HZ)
        sig_pp = preprocess_signal(sig, settings.SAMPLING_RATE_HZ)
        source_iter = (sig_pp[i:i + chunk_size]
                       for i in range(0, len(sig_pp) - chunk_size + 1, chunk_size))
    elif req.source_type == "arduino":
        # Simulated Arduino stream — for demo (real hardware would push via serial)
        # We'll generate normal ECG with occasional PVCs
        from app.ml.data import synthetic_normal_ecg
        sig_n = synthetic_normal_ecg(duration_s=300.0, seed=7)
        sig_a = synthetic_arrhythmia_ecg(duration_s=300.0, seed=11)
        # Mix: 70% normal, 30% arrhythmia
        mix = sig_n.copy()
        n = len(mix)
        for i in range(0, n, 128 * 10):  # every 10s, swap to arrhythmia for 3s
            if (i // (128 * 10)) % 3 == 0:
                mix[i:i + 128 * 3] = sig_a[i:i + 128 * 3]
        mix = preprocess_signal(mix, settings.SAMPLING_RATE_HZ)
        chunk_size = int(req.chunk_seconds * settings.SAMPLING_RATE_HZ)
        source_iter = (mix[i:i + chunk_size]
                       for i in range(0, len(mix) - chunk_size + 1, chunk_size))
    else:
        raise HTTPException(400, f"Unknown source_type: {req.source_type}")

    _LIVE_SESSIONS[session_id] = {
        "session_id": session_id,
        "user_id": user_id,
        "model_id": req.model_id,
        "source_type": req.source_type,
        "source_detail": req.source_detail,
        "engine": engine,
        "source_iter": source_iter,
        "chunk_seconds": req.chunk_seconds,
        "started_at": time.time(),
        "total_beats": 0,
        "anomaly_beats": 0,
        "alerts": [],
        "last_t": 0,
        "active": True,
        # ECG metrics tracking
        "ecg_buffer": np.array([], dtype=np.float32),  # accumulated ECG for metrics
        "ecg_metrics": None,                            # last computed metrics
        "ecg_metrics_chunk_count": 0,                   # recompute every 5 chunks
    }

    return OkResponse(ok=True, message="Session started", data={
        "session_id": session_id,
        "model_info": engine.info(),
        "threshold_override": req.threshold_override,
    })


@router.post("/{session_id}/stop", response_model=OkResponse)
def stop_session(session_id: str, user=Depends(require_user)):
    sess = _LIVE_SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found or already stopped")
    sess["active"] = False

    # Persist final state
    with get_db() as db:
        db.execute(text("""
            UPDATE EcgSession
            SET status = 'completed', endedAt = datetime('now'),
                totalBeats = :tb, anomalyBeats = :ab,
                summaryJson = :sj
            WHERE id = :id
        """), {
            "tb": sess["total_beats"], "ab": sess["anomaly_beats"],
            "sj": json.dumps({
                "duration_s": time.time() - sess["started_at"],
                "alerts": len(sess["alerts"]),
            }),
            "id": session_id,
        })
        db.commit()
    return OkResponse(ok=True, message="Session stopped")


@router.get("", response_model=OkResponse)
def list_sessions(user=Depends(require_user)):
    user_id, _ = user
    with get_db() as db:
        rows = db.execute(text("""
            SELECT id, userId, modelId, sourceType, sourceDetail,
                   startedAt, endedAt, status, totalBeats, anomalyBeats, summaryJson
            FROM EcgSession WHERE userId = :uid
            ORDER BY startedAt DESC LIMIT 100
        """), {"uid": user_id}).fetchall()
    sessions = [{
        "id": r[0], "user_id": r[1], "model_id": r[2], "source_type": r[3],
        "source_detail": r[4], "started_at": r[5], "ended_at": r[6],
        "status": r[7], "total_beats": r[8], "anomaly_beats": r[9],
        "summary": json.loads(r[10] or "{}"),
    } for r in rows]
    return OkResponse(ok=True, data={"sessions": sessions})


@router.delete("/{session_id}", response_model=OkResponse)
def delete_session(session_id: str, user=Depends(require_user)):
    """Delete a session and all its related data (alerts, data points)."""
    user_id, _ = user
    with get_db() as db:
        # Verify the session belongs to this user
        row = db.execute(text("""
            SELECT userId FROM EcgSession WHERE id = :id
        """), {"id": session_id}).fetchone()
        if not row:
            raise HTTPException(404, "Session not found")
        if row[0] != user_id:
            raise HTTPException(403, "Not authorized to delete this session")

        # Also remove from live session registry if active
        if session_id in _LIVE_SESSIONS:
            _LIVE_SESSIONS[session_id]["active"] = False
            del _LIVE_SESSIONS[session_id]

        # Delete related alerts and data points (cascade)
        db.execute(text("DELETE FROM Alert WHERE sessionId = :id"), {"id": session_id})
        db.execute(text("DELETE FROM EcgDataPoint WHERE sessionId = :id"), {"id": session_id})
        db.execute(text("DELETE FROM EcgSession WHERE id = :id"), {"id": session_id})
        db.commit()
    return OkResponse(ok=True, message="Session deleted")


@router.get("/{session_id}", response_model=OkResponse)
def get_session(session_id: str, user=Depends(require_user)):
    with get_db() as db:
        row = db.execute(text("""
            SELECT id, userId, modelId, sourceType, sourceDetail,
                   startedAt, endedAt, status, totalBeats, anomalyBeats, summaryJson
            FROM EcgSession WHERE id = :id
        """), {"id": session_id}).fetchone()
        if not row:
            raise HTTPException(404, "Session not found")
        alerts = db.execute(text("""
            SELECT id, timestamp, anomalyScore, threshold, severity, message, contextJson
            FROM Alert WHERE sessionId = :id
            ORDER BY timestamp
        """), {"id": session_id}).fetchall()
    return OkResponse(ok=True, data={
        "session": {
            "id": row[0], "user_id": row[1], "model_id": row[2],
            "source_type": row[3], "source_detail": row[4],
            "started_at": row[5], "ended_at": row[6], "status": row[7],
            "total_beats": row[8], "anomaly_beats": row[9],
            "summary": json.loads(row[10] or "{}"),
        },
        "alerts": [{
            "id": a[0], "timestamp": a[1], "anomaly_score": a[2],
            "threshold": a[3], "severity": a[4], "message": a[5],
            "context": json.loads(a[6] or "{}"),
        } for a in alerts],
    })


@router.post("/{session_id}/next-chunk", response_model=OkResponse)
def next_chunk(session_id: str, user=Depends(require_user)):
    """Internal endpoint called by the WebSocket service to pull the next
    scored ECG chunk for live streaming. Returns one of:
      - {type: 'chunk', points: [...], alerts: [...], ...}
      - {type: 'end'}     — session exhausted
      - {type: 'error'}   — session error
    """
    from app.api.routes_sessions import next_live_chunk
    result = next_live_chunk(session_id)
    if result is None:
        return OkResponse(ok=True, data={"type": "end", "session_id": session_id})
    return OkResponse(ok=True, data=result)


@router.get("/{session_id}/datapoints", response_model=OkResponse)
def get_datapoints(session_id: str,
                   limit: int = 2000,
                   user=Depends(require_user)):
    """Retrieve stored ECG data points for a session (for report/replay)."""
    with get_db() as db:
        rows = db.execute(text("""
            SELECT t, value, prediction, anomalyScore, isAnomaly
            FROM EcgDataPoint WHERE sessionId = :id
            ORDER BY t LIMIT :lim
        """), {"id": session_id, "lim": limit}).fetchall()
    points = [{
        "t": r[0], "value": r[1], "prediction": r[2],
        "anomaly_score": r[3], "is_anomaly": bool(r[4]),
    } for r in rows]
    return OkResponse(ok=True, data={"points": points, "count": len(points)})


# ---------- Internal helpers used by the WebSocket service ----------

def get_live_session(session_id: str) -> Optional[dict]:
    return _LIVE_SESSIONS.get(session_id)


def next_live_chunk(session_id: str) -> Optional[dict]:
    """Pull the next chunk from the live session's source iterator, score it,
    persist data points + any alerts, and return the data + diagnostics.
    Returns None when the session is exhausted or stopped.
    """
    sess = _LIVE_SESSIONS.get(session_id)
    if not sess or not sess["active"]:
        return None

    try:
        chunk = next(sess["source_iter"])
    except StopIteration:
        sess["active"] = False
        return {"type": "end", "session_id": session_id}
    except Exception as e:
        sess["active"] = False
        return {"type": "error", "message": str(e)}

    # Score
    # Pass t_offset so the t values increase monotonically across chunks
    # (otherwise the chart overlaps on itself because every chunk restarts at t=0)
    engine = sess["engine"]
    points = engine.score_stream_chunk(chunk, t_offset=sess["last_t"])
    sess["total_beats"] += len(points)

    new_alerts = []
    for p in points:
        if p["is_anomaly"]:
            sess["anomaly_beats"] += 1
            # Emit an alert once per ~64 samples (avoid spamming)
            if p["t"] % 64 == 0:
                severity = "critical" if p["anomaly_score"] > engine.threshold * 1.5 else "warning"

                # Capture the 512-sample window centered on this anomaly
                # plus the model's reconstruction, so the UI and PDF report
                # can display the actual abnormal waveform.
                try:
                    local_t = p["t"] - sess["last_t"]
                    window, recon = engine.extract_window_at(local_t)
                    signal_downsampled = window[::2].tolist()
                    recon_downsampled = recon[::2].tolist()
                except Exception:
                    signal_downsampled = []
                    recon_downsampled = []

                # Step 2: Classify the arrhythmia type (if classifier available)
                classification = None
                if engine.classifier is not None:
                    try:
                        classification = engine.classify_window(window)
                    except Exception:
                        classification = None

                # Build alert message with classification info
                if classification:
                    msg = f"{classification['class_name']} detected (score={p['anomaly_score']:.4f}, confidence={classification['confidence']:.1%})"
                else:
                    msg = f"Abnormal ECG pattern detected (score={p['anomaly_score']:.4f}, threshold={engine.threshold:.4f})"

                alert = {
                    "timestamp": time.time(),
                    "anomaly_score": p["anomaly_score"],
                    "threshold": engine.threshold,
                    "severity": severity,
                    "message": msg,
                    "classification": classification,
                    "context": {
                        "t": p["t"],
                        "signal": signal_downsampled,
                        "reconstruction": recon_downsampled,
                        "classification": classification,
                        "fs": settings.SAMPLING_RATE_HZ // 2,
                    },
                }
                new_alerts.append(alert)
                sess["alerts"].append(alert)

                # Persist alert (with signal data in contextJson)
                with get_db() as db:
                    db.execute(text("""
                        INSERT INTO Alert
                            (id, sessionId, timestamp, anomalyScore, threshold,
                             severity, message, contextJson)
                        VALUES
                            (:id, :sid, datetime('now'), :as, :thr, :sev, :msg, :ctx)
                    """), {
                        "id": secrets.token_hex(12), "sid": session_id,
                        "as": p["anomaly_score"], "thr": engine.threshold,
                        "sev": severity, "msg": alert["message"],
                        "ctx": json.dumps(alert["context"]),
                    })
                    db.commit()

    # Persist data points (downsample by 4 to keep DB small — 32 samples/sec is plenty for replay)
    with get_db() as db:
        for p in points[::4]:
            db.execute(text("""
                INSERT INTO EcgDataPoint
                    (id, sessionId, t, value, prediction, anomalyScore, isAnomaly)
                VALUES
                    (:id, :sid, :t, :v, :pr, :as, :ia)
            """), {
                "id": secrets.token_hex(12), "sid": session_id,
                "t": p["t"],  # already includes t_offset from score_stream_chunk
                "v": p["value"],
                "pr": p["prediction"], "as": p["anomaly_score"],
                "ia": 1 if p["is_anomaly"] else 0,
            })
        db.commit()

    sess["last_t"] += len(points)

    # ---------- ECG vital signs (BPM, intervals, mean beat) ----------
    # Accumulate the raw ECG signal and recompute metrics every 5 chunks
    # (~5 seconds at 1 chunk/sec) to avoid excessive CPU usage.
    raw_values = np.array([p["value"] for p in points], dtype=np.float32)
    sess["ecg_buffer"] = np.concatenate([sess["ecg_buffer"], raw_values])
    # Keep only the last 30 seconds of signal (enough for stable BPM/HRV)
    max_buffer = settings.SAMPLING_RATE_HZ * 30
    if len(sess["ecg_buffer"]) > max_buffer:
        sess["ecg_buffer"] = sess["ecg_buffer"][-max_buffer:]

    sess["ecg_metrics_chunk_count"] += 1
    ecg_metrics = sess.get("ecg_metrics")  # default to previous value
    if sess["ecg_metrics_chunk_count"] % 5 == 0 or ecg_metrics is None:
        try:
            ecg_metrics = compute_ecg_metrics(
                sess["ecg_buffer"], fs=settings.SAMPLING_RATE_HZ
            )
            sess["ecg_metrics"] = ecg_metrics
        except Exception as e:
            print(f"[ecg_metrics] computation failed: {e}")
            # Keep previous metrics if computation failed

    return {
        "type": "chunk",
        "session_id": session_id,
        "points": points[::2],  # downsample 2x for websocket (still smooth)
        "alerts": new_alerts,
        "total_beats": sess["total_beats"],
        "anomaly_beats": sess["anomaly_beats"],
        "threshold": engine.threshold,
        "ecg_metrics": ecg_metrics,
    }
