from __future__ import annotations

from fastapi import APIRouter

from app import __version__

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/readyz")
async def readyz() -> dict:
    # Lightweight check — extend with DB/Redis pings if needed
    return {"status": "ready", "version": __version__}
