"""Tiny auth helper — bcrypt-style hashes via hashlib (no external deps).

This is intentionally lightweight for a research project; swap for passlib[bcrypt]
in production. Sessions are signed tokens (HMAC-SHA256 via hashlib).

The signing secret is persisted to a file so tokens survive backend restarts.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import os
import secrets
from pathlib import Path
from typing import Optional, Tuple

from app.core.config import settings


def _load_or_create_secret() -> str:
    """Load the signing secret from a file, or create it on first run.

    This ensures tokens remain valid across backend restarts (the previous
    implementation regenerated the secret on every restart, invalidating
    all existing tokens).
    """
    secret_file = settings.STORAGE_DIR / ".auth_secret"
    if secret_file.exists():
        return secret_file.read_text().strip()
    # Create a new secret on first run
    import secrets as _s
    secret = _s.token_hex(32)
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    secret_file.write_text(secret)
    # Set restrictive permissions (best effort on Windows)
    try:
        os.chmod(secret_file, 0o600)
    except Exception:
        pass
    return secret


_SECRET = _load_or_create_secret()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, hex_hash = stored.split("$", 1)
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return hmac.compare_digest(h.hex(), hex_hash)
    except Exception:
        return False


def make_token(user_id: str, email: str, role: str = "clinician",
               ttl_seconds: int = 86400) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + ttl_seconds,
    }
    body = json.dumps(payload, sort_keys=True).encode()
    sig = hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return f"{body.hex()}.{sig}"


def verify_token(token: str) -> Optional[Tuple[str, str, str]]:
    """Returns (user_id, email, role) if valid, else None."""
    try:
        body_hex, sig = token.split(".", 1)
        body = bytes.fromhex(body_hex)
        expected = hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(body)
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload["sub"], payload["email"], payload.get("role", "clinician")
    except Exception:
        return None
