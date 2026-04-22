from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings

_settings = get_settings()

_engine_kwargs: dict = {"pool_pre_ping": True, "future": True}
if _settings.database_url.startswith("sqlite"):
    # SQLite (dev/test only) — share a single in-memory connection across threads
    _engine_kwargs.update(
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    _engine_kwargs.update(pool_size=10, max_overflow=20)

engine = create_engine(_settings.database_url, **_engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
