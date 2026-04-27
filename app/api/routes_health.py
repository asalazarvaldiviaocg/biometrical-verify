from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Response, status
from sqlalchemy import text

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

    # Anti-spoof model — in production its absence falls back to a sharpness
    # heuristic that approves printed photos. Treat as fatal there.
    model_path = s.anti_spoof_model_path
    model_present = await asyncio.to_thread(_probe_model_present, model_path)
    checks["anti_spoof_model"] = {"ok": model_present, "path": model_path}
    if s.is_prod and not model_present:
        overall_ok = False

    if not overall_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if overall_ok else "degraded",
        "version": __version__,
        "checks": checks,
    }
