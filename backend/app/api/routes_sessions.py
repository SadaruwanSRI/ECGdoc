"""Live session endpoints — start, stop, list, get alerts."""
from __future__ import annotations

import json
import secrets
import threading
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
from app.ml.live_hierarchical import BeatStreamAnalyzer
from app.ml.ecg_metrics import compute_ecg_metrics
from app.ml.data import (
    INCART_RECORDS,
    synthetic_arrhythmia_ecg,
    stream_incart_record,
    stream_mitbih_record,
    preprocess_signal,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


# In-memory active sessions (model_id -> session state)
# This is read by the WebSocket service for live streaming.
_LIVE_SESSIONS: Dict[str, dict] = {}


def _close_session_source(sess: dict) -> None:
    """Close a session source without racing an in-progress serial read."""
    source_lock = sess.get("source_lock")
    if source_lock is None:
        close_source = getattr(sess.get("source_iter"), "close", None)
        if callable(close_source):
            close_source()
        return
    with source_lock:
        close_source = getattr(sess.get("source_iter"), "close", None)
        if callable(close_source):
            close_source()


def _persist_terminal_session(sess: dict, status: str, error: str | None = None) -> None:
    """Persist natural stream completion or failure."""
    summary = {
        "duration_s": time.time() - sess["started_at"],
        "alerts": len(sess["alerts"]),
        "analysis_mode": sess["analysis_mode"],
        "count_unit": sess["count_unit"],
        "total_samples": sess["total_samples"],
        "decision_threshold": sess["engine"].decision_threshold,
        "classifier_model_id": sess.get("classifier_model_id"),
    }
    if error:
        summary["error"] = error
    with get_db() as db:
        db.execute(text("""
            UPDATE EcgSession
            SET status = :status, endedAt = datetime('now'),
                totalBeats = :total, anomalyBeats = :anomalies,
                summaryJson = :summary
            WHERE id = :id
        """), {
            "status": status,
            "total": sess["total_beats"],
            "anomalies": sess["anomaly_beats"],
            "summary": json.dumps(summary),
            "id": sess["session_id"],
        })
        db.commit()

def _require_session_owner(session_id: str, user_id: str) -> None:
    """Reject access to missing sessions or sessions owned by another user."""
    with get_db() as db:
        row = db.execute(
            text("SELECT userId FROM EcgSession WHERE id = :id"),
            {"id": session_id},
        ).fetchone()
    if not row:
        raise HTTPException(404, "Session not found")
    if row[0] != user_id:
        raise HTTPException(403, "Not authorized to access this session")

@router.post("/start", response_model=OkResponse)
def start_session(req: StartSessionRequest, user=Depends(require_user)):
    if req.lead_name.strip().upper() != "MLII":
        raise HTTPException(
            400,
            "The installed model accepts MLII only. Use the Lead-II-axis "
            "electrode placement and declare lead_name='MLII'.",
        )
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


    # Spin up engine with the optional final MLII hierarchy.
    threshold = req.threshold_override if req.threshold_override is not None else m[2]
    engine = get_engine(
        m[1],
        threshold=threshold,
        classifier_path=classifier_path,
        improved_threshold=req.threshold_override,
    )
    analysis_mode = (
        "beat-aligned-hierarchical"
        if engine.improved_system is not None
        else "sliding-reconstruction"
    )
    corrected_preparation = (
        getattr(engine.improved_system, "preprocessing_version", None) == "beat-local-bandpass-zscore-v1"
    )

    # Build the source iterator based on source_type
    if req.source_type == "mit-bih-arrhythmia":
        record = req.source_detail or "100"
        try:
            # Test load to fail fast
            from app.ml.data import _download_record
            _download_record(record, "mitdb", required_lead="MLII")
        except Exception as e:
            raise HTTPException(400, f"Cannot load MIT-BIH record {record}: {e}")
        source_iter = stream_mitbih_record(
            record,
            chunk_seconds=req.chunk_seconds,
            raw_resampled=corrected_preparation,
        )
        source_preprocessed = True
    elif req.source_type == "incartdb":
        record = req.source_detail or INCART_RECORDS[0]
        try:
            # INCART uses standard lead II, the closest external counterpart
            # to the MLII lead used to develop the installed final model.
            from app.ml.data import _download_record
            _download_record(record, "incartdb", required_lead="II")
        except Exception as e:
            raise HTTPException(400, f"Cannot load INCART record {record}: {e}")
        source_iter = stream_incart_record(
            record,
            chunk_seconds=req.chunk_seconds,
            raw_resampled=corrected_preparation,
        )
        source_preprocessed = True
    elif req.source_type == "synthetic-arrhythmia":
        # Generate a 5-minute arrhythmia signal and chunk it
        sig = synthetic_arrhythmia_ecg(duration_s=300.0, seed=42)
        from app.ml.data import windowize
        # Chunk in 4s windows
        chunk_size = int(req.chunk_seconds * settings.SAMPLING_RATE_HZ)
        # The synthetic generator already applies the project preprocessing.
        sig_pp = np.asarray(sig, dtype=np.float32)
        source_iter = (sig_pp[i:i + chunk_size]
                       for i in range(0, len(sig_pp) - chunk_size + 1, chunk_size))
        source_preprocessed = True
    elif req.source_type == "arduino":
        # Simulated Arduino stream — for demo (real hardware would push via serial)
        # We'll generate normal ECG with occasional PVCs
        from app.services.arduino_serial import ArduinoSerialStream
        port = (req.source_detail or "").strip()
        if not port:
            raise HTTPException(400, "Select an Arduino serial port")
        try:
            source_iter = ArduinoSerialStream(port=port, chunk_seconds=4.0)
            source_preprocessed = False
        except Exception as e:
            raise HTTPException(400, f"Cannot open Arduino on {port}: {e}") from e
    else:
        raise HTTPException(400, f"Unknown source_type: {req.source_type}")

    session_id = secrets.token_hex(12)
    with get_db() as db:
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

    _LIVE_SESSIONS[session_id] = {
        "session_id": session_id,
        "user_id": user_id,
        "model_id": req.model_id,
        "classifier_model_id": req.classifier_model_id,
        "source_type": req.source_type,
        "source_detail": req.source_detail,
        "engine": engine,
        "source_iter": source_iter,
        "source_lock": threading.RLock(),
        "processing_lock": threading.Lock(),
        "chunk_seconds": req.chunk_seconds,
        "source_preprocessed": source_preprocessed,
        "corrected_preparation": corrected_preparation,
        "analysis_mode": analysis_mode,
        "count_unit": "beats" if engine.improved_system is not None else "samples",
        "beat_analyzer": (
            BeatStreamAnalyzer(engine)
            if engine.improved_system is not None
            else None
        ),
        "started_at": time.time(),
        "total_beats": 0,
        "anomaly_beats": 0,
        "total_samples": 0,
        "alerts": [],
        # Defensive guard: a beat may be returned again when adjacent live
        # buffers overlap. Each R peak is allowed to create at most one alert.
        "alerted_peak_times": set(),
        "last_t": 0,
        "active": True,
        # ECG metrics tracking
        "ecg_buffer": np.array([], dtype=np.float32),  # accumulated ECG for metrics
        "ecg_metrics": None,                            # last computed metrics
        "ecg_metrics_chunk_count": 0,                   # recompute every 5 chunks
    }
    with get_db() as db:
        db.execute(text("""
            UPDATE EcgSession SET summaryJson = :summary WHERE id = :id
        """), {
            "id": session_id,
            "summary": json.dumps({
                "analysis_mode": analysis_mode,
                "count_unit": "beats" if engine.improved_system is not None else "samples",
                "decision_threshold": engine.decision_threshold,
                "classifier_model_id": req.classifier_model_id,
            }),
        })
        db.commit()

    return OkResponse(ok=True, message="Session started", data={
        "session_id": session_id,
        "model_info": engine.info(),
        "analysis_mode": analysis_mode,
        "count_unit": "beats" if engine.improved_system is not None else "samples",
        "threshold_override": req.threshold_override,
    })


@router.post("/{session_id}/stop", response_model=OkResponse)
def stop_session(session_id: str, user=Depends(require_user)):
    user_id, _ = user
    _require_session_owner(session_id, user_id)
    sess = _LIVE_SESSIONS.get(session_id)
    if not sess or not sess["active"]:
        raise HTTPException(404, "Session not found or already stopped")
    sess["active"] = False
    _close_session_source(sess)

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
                "analysis_mode": sess["analysis_mode"],
                "count_unit": sess["count_unit"],
                "total_samples": sess["total_samples"],
                "decision_threshold": sess["engine"].decision_threshold,
                "classifier_model_id": sess.get("classifier_model_id"),
                "live_inference_note": (
                    "R-peak-aligned inference with RR context; results are delayed "
                    "until the next RR interval and post-peak waveform are available."
                    if sess["analysis_mode"] == "beat-aligned-hierarchical"
                    else "Legacy sliding reconstruction inference."
                ),
            }),
            "id": session_id,
        })
        db.commit()
    _LIVE_SESSIONS.pop(session_id, None)
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
            _close_session_source(_LIVE_SESSIONS[session_id])
            del _LIVE_SESSIONS[session_id]

        # Delete related alerts and data points (cascade)
        db.execute(text("DELETE FROM Alert WHERE sessionId = :id"), {"id": session_id})
        db.execute(text("DELETE FROM EcgDataPoint WHERE sessionId = :id"), {"id": session_id})
        db.execute(text("DELETE FROM EcgSession WHERE id = :id"), {"id": session_id})
        db.commit()
    return OkResponse(ok=True, message="Session deleted")


@router.get("/{session_id}", response_model=OkResponse)
def get_session(session_id: str, user=Depends(require_user)):
    user_id, _ = user
    _require_session_owner(session_id, user_id)
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
    user_id, _ = user
    _require_session_owner(session_id, user_id)
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
    user_id, _ = user
    _require_session_owner(session_id, user_id)
    limit = max(1, min(limit, 10_000))
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
    """Serialize chunk processing so overlapping pollers cannot race state."""
    sess = _LIVE_SESSIONS.get(session_id)
    if not sess or not sess["active"]:
        return None
    processing_lock = sess["processing_lock"]
    if not processing_lock.acquire(blocking=False):
        return {"type": "pending", "session_id": session_id}
    try:
        result = _next_live_chunk(session_id)
        if result and result.get("type") in {"end", "error"}:
            _LIVE_SESSIONS.pop(session_id, None)
        return result
    finally:
        processing_lock.release()


def _next_live_chunk(session_id: str) -> Optional[dict]:
    """Pull the next chunk from the live session's source iterator, score it,
    persist data points + any alerts, and return the data + diagnostics.
    Returns None when the session is exhausted or stopped.
    """
    sess = _LIVE_SESSIONS.get(session_id)
    if not sess or not sess["active"]:
        return None

    # A four-second hardware read is longer than the WebSocket polling period.
    # Reject overlapping pulls rather than allowing concurrent PySerial reads.
    source_lock = sess["source_lock"]
    if not source_lock.acquire(blocking=False):
        return {"type": "pending", "session_id": session_id}
    try:
        try:
            source_item = next(sess["source_iter"])
        except StopIteration:
            sess["active"] = False
            close_source = getattr(sess.get("source_iter"), "close", None)
            if callable(close_source):
                close_source()
            _persist_terminal_session(sess, "completed")
            return {"type": "end", "session_id": session_id}
        except Exception as exc:
            sess["active"] = False
            close_source = getattr(sess.get("source_iter"), "close", None)
            if callable(close_source):
                close_source()
            _persist_terminal_session(sess, "failed", str(exc))
            return {"type": "error", "message": str(exc)}
    finally:
        source_lock.release()

    # The final system accepts one ECG lead only.
    chunk = source_item

    # Prepare the display signal and run the active inference mode.
    engine = sess["engine"]
    if sess["analysis_mode"] == "beat-aligned-hierarchical":
        display_chunk = (
            np.asarray(chunk, dtype=np.float32).reshape(-1)
            if sess["source_preprocessed"]
            else preprocess_signal(chunk, settings.SAMPLING_RATE_HZ)
        )
        points = [
            {
                "t": sess["last_t"] + index,
                "value": float(value),
                "prediction": 0.0,
                "anomaly_score": 0.0,
                "is_anomaly": False,
            }
            for index, value in enumerate(display_chunk)
        ]
        beat_input = np.asarray(chunk, dtype=np.float32).reshape(-1) if sess.get("corrected_preparation") else display_chunk
        beat_results = sess["beat_analyzer"].append(beat_input)
        sess["total_beats"] += len(beat_results)
        sess["anomaly_beats"] += sum(
            1 for event in beat_results if event["is_anomaly"]
        )
    else:
        points = engine.score_stream_chunk(
            chunk,
            t_offset=sess["last_t"],
            already_preprocessed=sess["source_preprocessed"],
        )
        beat_results = []
        # The legacy mode has no beat detector, so its historical sample counts
        # are preserved and explicitly labelled as samples in the API.
        sess["total_beats"] += len(points)
    sess["total_samples"] += len(points)

    new_alerts = []
    if sess["analysis_mode"] == "beat-aligned-hierarchical":
        for event in beat_results:
            if not event["is_anomaly"]:
                continue
            beat_t = int(event["t"])
            if beat_t in sess["alerted_peak_times"]:
                continue
            sess["alerted_peak_times"].add(beat_t)
            probability = float(event["anomaly_probability"])
            decision_threshold = float(event["threshold"])
            severity = (
                "critical"
                if probability >= max(0.75, decision_threshold + 0.25)
                else "warning"
            )
            classification = {
                "class": event["class"],
                "class_name": event["class_name"],
                "confidence": event["confidence"],
                "probabilities": event["probabilities"],
                "mode": event["classification_mode"],
            }
            if event["class"] == "Unclassified abnormal":
                message = (
                    "Unclassified abnormal beat detected "
                    f"(probability={probability:.1%})"
                )
            else:
                message = (
                    f"{event['class_name']} detected "
                    f"(abnormal probability={probability:.1%}, "
                    f"class confidence={event['confidence']:.1%})"
                )
            alert = {
                "timestamp": time.time(),
                "anomaly_score": probability,
                "threshold": decision_threshold,
                "severity": severity,
                "message": message,
                "classification": classification,
                "context": {
                    "beat_id": f"{session_id}:{beat_t}",
                    "t": beat_t,
                    "signal": event["signal"][::2].tolist(),
                    "reconstruction": [],
                    "classification": classification,
                    "fs": settings.SAMPLING_RATE_HZ // 2,
                    "r_peak_index": 200 // 2,
                    "score_type": "abnormal_probability",
                    "analysis_mode": sess["analysis_mode"],
                    "rr_seconds": event["rr_seconds"],
                    "rr_features": event["rr_features"],
                    "classification_mode": event["classification_mode"],
                },
            }
            new_alerts.append(alert)
            sess["alerts"].append(alert)
            with get_db() as db:
                db.execute(text("""
                    INSERT INTO Alert
                        (id, sessionId, timestamp, anomalyScore, threshold,
                         severity, message, contextJson)
                    VALUES
                        (:id, :sid, datetime('now'), :as, :thr, :sev, :msg, :ctx)
                """), {
                    "id": secrets.token_hex(12), "sid": session_id,
                    "as": probability, "thr": decision_threshold,
                    "sev": severity, "msg": message,
                    "ctx": json.dumps(alert["context"]),
                })
                db.commit()

    for p in points if sess["analysis_mode"] != "beat-aligned-hierarchical" else []:
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
                "pr": (
                    None
                    if sess["analysis_mode"] == "beat-aligned-hierarchical"
                    else p["prediction"]
                ),
                "as": p["anomaly_score"],
                "ia": 1 if p["is_anomaly"] else 0,
            })
        if beat_results:
            for event in beat_results:
                half_width = int(0.08 * settings.SAMPLING_RATE_HZ)
                db.execute(text("""
                    UPDATE EcgDataPoint
                    SET anomalyScore = :score, isAnomaly = :is_anomaly
                    WHERE sessionId = :sid AND t BETWEEN :start_t AND :end_t
                """), {
                    "score": event["anomaly_probability"],
                    "is_anomaly": 1 if event["is_anomaly"] else 0,
                    "sid": session_id,
                    "start_t": event["t"] - half_width,
                    "end_t": event["t"] + half_width,
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
        "threshold": engine.decision_threshold,
        "analysis_mode": sess["analysis_mode"],
        "count_unit": sess["count_unit"],
        "total_samples": sess["total_samples"],
        "beat_results": [
            {
                key: value
                for key, value in event.items()
                if key != "signal"
            }
            for event in beat_results
        ],
        "ecg_metrics": ecg_metrics,
        "source_status": getattr(sess.get("source_iter"), "last_status", None),
    }
