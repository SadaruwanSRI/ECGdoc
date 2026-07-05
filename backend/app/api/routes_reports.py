"""Report endpoints — generate and download PDF reports."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse

from app.api.schemas import PdfReportRequest, OkResponse
from app.api.routes_auth import require_user
from app.services.reports import generate_session_report

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("/generate", response_model=OkResponse)
def generate_report(req: PdfReportRequest, user=Depends(require_user)):
    try:
        path = generate_session_report(
            session_id=req.session_id,
            physician_name=req.physician_name,
            notes=req.notes,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {e}")

    return OkResponse(ok=True, message="Report generated", data={
        "report_path": str(path),
        "filename": Path(path).name,
    })


@router.get("/download/{filename}")
def download_report(filename: str):
    """Download a generated PDF by filename."""
    # Prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    from app.core.config import settings
    path = settings.REPORT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=filename,
    )
