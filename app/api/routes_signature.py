"""Signature comparison REST endpoint.

POST /api/v1/signature/compare
    Body (multipart/form-data):
        id_back_image: file       — INE/IFE back-side photo (JPEG/PNG/WebP)
        canvas_signature: file    — canvas signature PNG (transparent OK)
    Returns: SignatureCompareResult JSON.

Auth: same Bearer-token / rate-limit machinery used by the other verify
routes — see app.api.deps.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.api.deps import CurrentUser, rate_limit
from app.core.logging import get_logger
from app.services.mime_sniff import sniff
from app.services.signature_engine import (
    SIGNATURE_THRESHOLD_DEFAULT,
    compare_signatures,
)

router = APIRouter(prefix="/api/v1/signature", tags=["signature"])
log = get_logger(__name__)

ALLOWED_ID_MIME = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_SIG_MIME = {"image/png"}  # canvas always emits PNG; reject jpeg-of-canvas
MAX_BYTES = 5 * 1024 * 1024  # 5 MB per side


class SignatureCompareResult(BaseModel):
    similarity:             int
    ssim:                   float
    match_pass:             bool
    threshold:              int
    id_signature_found:     bool
    canvas_signature_found: bool
    model:                  str
    reason:                 str | None = None


async def _read_capped(f: UploadFile, cap: int) -> bytes:
    data = await f.read(cap + 1)
    if len(data) > cap:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")
    return data


def _enforce_mime(payload: bytes, header: str | None, allowed: set[str], label: str) -> None:
    base = (header or "").split(";", 1)[0].strip().lower()
    if base not in allowed:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{label} declared content-type {header!r} not allowed",
        )
    detected = sniff(payload)
    if detected not in allowed:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{label} bytes do not match a permitted format (detected={detected!r})",
        )


@router.post(
    "/compare",
    response_model=SignatureCompareResult,
    status_code=status.HTTP_200_OK,
)
async def compare(
    id_back_image:    UploadFile = File(...),
    canvas_signature: UploadFile = File(...),
    user:             CurrentUser = Depends(rate_limit(bucket="signature_compare")),
) -> SignatureCompareResult:
    id_bytes  = await _read_capped(id_back_image,    MAX_BYTES)
    sig_bytes = await _read_capped(canvas_signature, MAX_BYTES)
    _enforce_mime(id_bytes,  id_back_image.content_type,    ALLOWED_ID_MIME,  "id_back_image")
    _enforce_mime(sig_bytes, canvas_signature.content_type, ALLOWED_SIG_MIME, "canvas_signature")

    try:
        # SSIM + OpenCV are CPU-bound (~10–50ms). Running them inline in an
        # async handler would block the event loop and serialize concurrent
        # signature comparisons. Defer to the FastAPI thread pool so other
        # requests progress in parallel.
        result = await run_in_threadpool(
            compare_signatures, id_bytes, sig_bytes, SIGNATURE_THRESHOLD_DEFAULT,
        )
    except ValueError as exc:
        # decode failures from the engine surface here
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    log.info(
        "signature_compare",
        user_id=user.id,
        similarity=result.similarity,
        match_pass=result.match_pass,
        reason=result.reason,
    )
    return SignatureCompareResult(**result.to_dict())
