"""SQLAlchemy database engine + session.

We share the same SQLite file as the Prisma frontend. The schema is owned by
Prisma (run `bun run db:push` from the project root to create tables); FastAPI
uses raw SQL / SQLAlchemy core to read/write the same tables.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings


# SQLAlchemy needs a filesystem path for SQLite, not a URL with sqlite:///
engine = create_engine(
    settings.SQLALCHEMY_URL,
    connect_args={"check_same_thread": False} if settings.SQLALCHEMY_URL.startswith("sqlite") else {},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def health() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
