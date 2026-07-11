"""Raw Arduino/AD8232 hardware test endpoints (no ML processing)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.routes_auth import require_user
from app.api.schemas import ArduinoSerialTestRequest, OkResponse
from app.services.arduino_serial import acquire_test_samples, list_serial_ports

router = APIRouter(prefix="/api/arduino", tags=["arduino"])


@router.get("/ports", response_model=OkResponse)
def get_ports(user=Depends(require_user)):
    try:
        ports = list_serial_ports()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return OkResponse(ok=True, data={"ports": ports, "count": len(ports)})


@router.post("/test", response_model=OkResponse)
def test_serial(req: ArduinoSerialTestRequest, user=Depends(require_user)):
    try:
        result = acquire_test_samples(
            port=req.port,
            baud_rate=req.baud_rate,
            sample_count=req.sample_count,
            timeout_seconds=req.timeout_seconds,
        )
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # PySerial raises platform-specific SerialException subclasses, so keep
        # their actionable message while preventing an internal-server error.
        raise HTTPException(status_code=400, detail=f"Cannot read serial port {req.port}: {exc}") from exc

    return OkResponse(ok=True, message="Raw Arduino test completed", data=result)
