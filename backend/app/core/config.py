"""Application configuration."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    # config.py lives at backend/app/core/config.py
    # BASE_DIR  = backend/app/  (where storage/, logs/ live)
    # PROJECT_ROOT = parent of backend/ (where package.json, prisma/, .env live)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent  # backend/app/
    PROJECT_ROOT: Path = BASE_DIR.parent.parent              # project root
    PRISMA_DIR: Path = PROJECT_ROOT / "prisma"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    MODEL_DIR: Path = STORAGE_DIR / "models"
    REPORT_DIR: Path = STORAGE_DIR / "reports"
    DATASET_DIR: Path = STORAGE_DIR / "datasets"
    LOG_DIR: Path = BASE_DIR / "logs"

    # Database — same SQLite file used by Prisma frontend (shared schema).
    # Accepts either Prisma format (file:./dev.db or file:/abs/path) or SQLAlchemy URL.
    # To switch to PostgreSQL in production: set DATABASE_URL=postgresql://...
    # and update prisma/schema.prisma provider to "postgresql".
    DATABASE_URL: str = "file:./dev.db"

    @property
    def SQLALCHEMY_URL(self) -> str:
        """Return SQLAlchemy-style URL (sqlite:////abs/path or postgresql://...).

        IMPORTANT: Prisma resolves `file:./dev.db` relative to the schema.prisma
        file location (i.e. the prisma/ directory). We do the same here so both
        Prisma and FastAPI use the same SQLite file.
        """
        raw = self.DATABASE_URL
        if raw.startswith("postgresql://") or raw.startswith("postgres://"):
            return raw
        if raw.startswith("file:"):
            path_part = raw[5:]
            if not path_part.startswith("/"):
                # Relative path — resolve the same way Prisma does:
                # relative to the prisma/ directory (where schema.prisma lives).
                path_part = str((self.PRISMA_DIR / path_part).resolve())
            return f"sqlite:///{path_part}"
        # Already a SQLAlchemy URL or bare path
        if raw.startswith("sqlite:"):
            return raw
        return f"sqlite:///{raw}"

    @property
    def DB_FILE_PATH(self) -> str:
        """Filesystem path to the SQLite file (for non-SQLAlchemy uses)."""
        url = self.SQLALCHEMY_URL
        if url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "", 1)
        return ""

    # ML defaults — mirror the slide specs
    SAMPLING_RATE_HZ: int = 128
    WINDOW_SECONDS: float = 4.0
    WINDOW_SAMPLES: int = 512  # 128 Hz * 4 s
    BANDPASS_LOW: float = 0.5
    BANDPASS_HIGH: float = 50.0
    DEFAULT_THRESHOLD_K: float = 2.0
    DEFAULT_LEARNING_RATE: float = 1e-3
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_EPOCHS: int = 20  # quick default for sandbox; use 50+ for real training

    # CORS — allow the Next.js frontend (port 3000) and Caddy gateway (port 81)
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:81",
        "http://127.0.0.1:81",
    ]

    # Service port (must match what Caddy forwards via XTransformPort=8000)
    PORT: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"

    def ensure_dirs(self) -> None:
        for d in (self.STORAGE_DIR, self.MODEL_DIR, self.REPORT_DIR,
                  self.DATASET_DIR, self.LOG_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
