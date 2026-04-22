from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, current_user, rate_limit
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_db
from app.models.verification import AuditEvent, Verification
from app.schemas.verify import (
    ChallengeResponse,
    VerifyAccepted,
    VerifyResult,
)
from app.services.storage import store_encrypted_blob

router = APIRouter(prefix="/api/v1/verify", tags=["verify"])
log = get_logger(__name__)
_settings = get_settings()

ALLOWED_IMG = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VID = {"video/mp4", "video/webm"}


# ---------- challenge ----------

@router.get("/challenge", response_model=ChallengeResponse)
async def issue_challenge(_: CurrentUser = Depends(current_user)) -> ChallengeResponse:
    """Anti-replay nonce + human-readable instruction for the client to render."""
    nonce = secrets.token_urlsafe(24)
    expires = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    return ChallengeResponse(
        challenge="blink_twice",
        instruction="Mira a la cámara y parpadea dos veces lentamente.",
        nonce=nonce,
        expires_at=expires,
    )


# ---------- submit ----------

async def _read_capped(f: UploadFile, cap: int) -> bytes:
    data = await f.read(cap + 1)
    if len(data) > cap:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")
    return data


@router.post(
    "/submit",
    response_model=VerifyAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_verification(
    contract_id: str = Form(..., min_length=1, max_length=64),
    challenge: str = Form("blink_twice"),
    nonce: str = Form(..., min_length=1, max_length=128),
    id_image: UploadFile = File(...),
    selfie_video: UploadFile = File(...),
    user: CurrentUser = Depends(rate_limit()),
    db: Session = Depends(get_db),
) -> VerifyAccepted:
    if id_image.content_type not in ALLOWED_IMG:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "ID must be JPEG/PNG/WEBP"
        )
    if selfie_video.content_type not in ALLOWED_VID:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Video must be MP4/WEBM"
        )

    id_bytes = await _read_capped(id_image, _settings.max_id_bytes)
    vid_bytes = await _read_capped(selfie_video, _settings.max_video_bytes)

    sha_id = hashlib.sha256(id_bytes).hexdigest()
    sha_vid = hashlib.sha256(vid_bytes).hexdigest()

    job_id = str(uuid.uuid4())
    id_ref = store_encrypted_blob(user.id, job_id, "id", id_bytes)
    vid_ref = store_encrypted_blob(user.id, job_id, "video", vid_bytes)

    record = Verification(
        id=job_id,
        user_id=user.id,
        contract_id=contract_id,
        status="queued",
        challenge=challenge,
        nonce=nonce,
        sha_id=sha_id,
        sha_video=sha_vid,
        id_blob_ref=id_ref.to_dict(),
        video_blob_ref=vid_ref.to_dict(),
    )
    db.add(record)
    db.add(AuditEvent(verification_id=job_id, event="submitted",
                      detail={"sha_id": sha_id, "sha_video": sha_vid}))
    db.commit()

    # Enqueue async — import lazily so the API stays bootable without Celery during tests
    from app.workers.tasks import verify_identity_task

    verify_identity_task.delay(job_id=job_id)
    log.info("verification_submitted", job_id=job_id, user_id=user.id, contract_id=contract_id)

    return VerifyAccepted(
        job_id=job_id, status="queued", sha256_id=sha_id, sha256_video=sha_vid
    )


# ---------- status ----------

@router.get("/{job_id}", response_model=VerifyResult)
async def get_verification(
    job_id: str,
    user: CurrentUser = Depends(current_user),
    db: Session = Depends(get_db),
) -> VerifyResult:
    rec = db.get(Verification, job_id)
    if rec is None or rec.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Verification not found")

    receipt = rec.receipt or {}
    return VerifyResult(
        job_id=rec.id,
        status=rec.status,
        decision=rec.decision,
        similarity=rec.similarity,
        distance=rec.distance,
        threshold=rec.threshold,
        liveness_score=rec.liveness_score,
        challenge_passed=rec.challenge_passed,
        deepfake_suspicious=rec.deepfake_suspicious,
        model=rec.model,
        receipt_hash=receipt.get("msg_sha256"),
        receipt_signature=receipt.get("signature"),
        reason=rec.reason,
        created_at=rec.created_at.isoformat() if rec.created_at else None,
        finished_at=rec.finished_at.isoformat() if rec.finished_at else None,
    )
