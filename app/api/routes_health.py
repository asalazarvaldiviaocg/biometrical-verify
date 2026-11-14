from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from alembic.config import Config
from alembic.script import ScriptDirectory

from app import __version__
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import engine
from app.services.redis_client import get_redis
from app.services.storage import _client as _s3_client

router = APIRouter(tags=["health"])
log = get_logger(__name__)


@router.get("/healthz")
async def healthz() -> dict:
    """Liveness probe — process is up. Does NOT check dependencies."""
    return {"status": "ok", "version": __version__}


def _probe_db() -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def _probe_schema() -> str:
    """Return the database's current alembic revision, or raise.

    A database that answers SELECT 1 but is missing its tables is the worst
    state this service can be in: every other probe below passes, every route
    that writes 500s, and readiness reports ready. That is not hypothetical -
    the engine booted without `alembic upgrade head` answered /healthz and
    /readyz with ready while every challenge insert failed on a missing
    issued_nonces table, and the API in front of it answered 502 for
    something its own health check called healthy.
    """
    with engine.connect() as conn:
        row = conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).first()
    current = row[0] if row else None
    if not current:
        raise RuntimeError("no migrations applied: alembic_version is empty")
    return current


def _head_revisions() -> list[str]:
    cfg = Config(str(_ALEMBIC_INI))
    return ScriptDirectory.from_config(cfg).get_heads()


def _probe_redis() -> None:
    get_redis().ping()


def _probe_s3(bucket: str) -> None:
    _s3_client().head_bucket(Bucket=bucket)


def _probe_model_present(model_path: str) -> bool:
    return os.path.exists(model_path) and os.path.getsize(model_path) > 1024


@router.get("/readyz")
async def readyz(response: Response) -> dict:
    """Readiness probe — service can actually handle requests.

    Probes:
      - Postgres connectivity (SELECT 1).
      - Redis connectivity (PING).
      - Object store reachability (HEAD bucket).
      - Anti-spoof model presence (when running in production).

    All blocking probes run via asyncio.to_thread so the event loop stays
    responsive even if a dependency is hung. Returns 503 with per-component
    status when any check fails so a load balancer pulls the pod out of
    rotation. The previous handler always returned 200 — Kubernetes happily
    routed traffic to broken instances.
    """
    s = get_settings()
    checks: dict[str, dict] = {}
    overall_ok = True

    # DB
    try:
        await asyncio.to_thread(_probe_db)
        checks["db"] = {"ok": True}
    except Exception as e:
        overall_ok = False
        checks["db"] = {"ok": False, "error": str(e)[:120]}

    # Schema: the database is reachable AND migrated to head. Connectivity
    # alone is the probe that had nothing to say about an empty schema.
    try:
        revision = await asyncio.to_thread(_probe_schema)
        heads = await asyncio.to_thread(_head_revisions)
        at_head = revision in heads
        checks["schema"] = {"ok": at_head, "revision": revision, "head": ",".join(heads)}
        if not at_head:
            overall_ok = False
    except Exception as e:
        overall_ok = False
        checks["schema"] = {"ok": False, "error": str(e)[:120]}

    # Redis
    try:
        await asyncio.to_thread(_probe_redis)
        checks["redis"] = {"ok": True}
    except Exception as e:
        overall_ok = False
        checks["redis"] = {"ok": False, "error": str(e)[:120]}

    # Object store
    try:
        await asyncio.to_thread(_probe_s3, s.s3_bucket)
        checks["object_store"] = {"ok": True, "bucket": s.s3_bucket}
    except Exception as e:
        overall_ok = False
        checks["object_store"] = {"ok": False, "error": str(e)[:120]}

    # Anti-spoof model file — INFORMATIONAL, and deliberately not fatal.
    #
    # Nothing in the pipeline loads this file: the liveness path is the
    # proprietary multi-signal engine in services/biometrical_liveness_engine.py
    # (OpenCV + NumPy, no pre-trained weights), and grepping the runtime for
    # onnxruntime finds exactly one hit - this probe. Treating it as fatal in
    # production readied-out a service whose liveness never needed it, which
    # is how the KYC box spent weeks reporting ok:false for a file no code
    # path opens.
    #
    # It stays reported so a future engine that DOES load a model cannot lose
    # the check silently: a note distinguishes "absent and unused" from
    # "absent while something depends on it".
    model_path = s.anti_spoof_model_path
    model_present = await asyncio.to_thread(_probe_model_present, model_path)
    checks["anti_spoof_model"] = {
        "ok": model_present,
        "path": model_path,
        "note": "loaded_by_pipeline=false; liveness runs the multi-signal engine",
    }

    if not overall_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if overall_ok else "degraded",
        "version": __version__,
        "checks": checks,
    }
