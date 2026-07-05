"""Auth endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Header

from app.api.schemas import UserCreate, UserLogin, UserOut, OkResponse
from app.core.auth import hash_password, verify_password, make_token, verify_token
from app.db.session import get_db
from sqlalchemy import text

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=OkResponse)
def register(payload: UserCreate):
    with get_db() as db:
        existing = db.execute(
            text("SELECT id FROM User WHERE email = :email"),
            {"email": payload.email},
        ).fetchone()
        if existing:
            raise HTTPException(400, "Email already registered")
        import secrets as _s
        user_id = _s.token_hex(12)
        db.execute(
            text("""
                INSERT INTO User (id, email, name, role, passwordHash, createdAt, updatedAt)
                VALUES (:id, :email, :name, 'clinician', :pwh, datetime('now'), datetime('now'))
            """),
            {"id": user_id, "email": payload.email, "name": payload.name,
             "pwh": hash_password(payload.password)},
        )
        db.commit()
    token = make_token(user_id, payload.email, "clinician")
    return OkResponse(ok=True, message="Registered", data={"token": token, "user_id": user_id})


@router.post("/login", response_model=OkResponse)
def login(payload: UserLogin):
    with get_db() as db:
        row = db.execute(
            text("SELECT id, email, name, role, passwordHash FROM User WHERE email = :email"),
            {"email": payload.email},
        ).fetchone()
    if not row or not verify_password(payload.password, row[4]):
        raise HTTPException(401, "Invalid credentials")
    token = make_token(row[0], row[1], row[3])
    return OkResponse(ok=True, message="Logged in", data={
        "token": token,
        "user": {"id": row[0], "email": row[1], "name": row[2], "role": row[3]},
    })


def require_user(authorization: str | None = Header(None)) -> tuple[str, str]:
    """Dependency: returns (user_id, email). Raises 401 if unauthenticated."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization[7:]
    claims = verify_token(token)
    if not claims:
        raise HTTPException(401, "Invalid or expired token")
    return claims[0], claims[1]
