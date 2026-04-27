from __future__ import annotations

import hashlib
import re
import uuid

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
from app.services import nonce_store
from app.services.mime_sniff import sniff
from app.services.storage import delete_blob, store_encrypted_blob

router = APIRouter(prefix="/api/v1/verify", tags=["verify"])
log = get_logger(__name__)
_settings = get_settings()

ALLOWED_IMG = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VID = {"video/mp4", "video/webm"}

# Permitted challenge values. Match the type alias in schemas; explicit set
# here so we can reject early before any DB work.
ALLOWED_CHALLENGES = {"blink_twice", "blink_once", "turn_head", "none"}

# Defence-in-depth alongside CurrentUser: contract IDs must be safe to embed in
# logs and S3 metadata.
_SAFE_CONTRACT_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_SAFE_NONCE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


# ---------- challenge ----------

@router.get("/challenge", response_model=ChallengeResponse)
def issue_challenge(
    user: CurrentUser = Depends(rate_limit(bucket="challenge")),
    db: Session = Depends(get_db),
) -> ChallengeResponse:
    """Persisted anti-replay nonce + human-readable instruction. Bound to the
    requesting user so /submit can prove it's the same identity."""
    challenge = "blink_twice"
    issued = nonce_store.issue(user.id, challenge, db)
    return ChallengeResponse(
        challenge=challenge,
        instruction="Mira a la cámara y parpadea dos veces lentamente.",
        nonce=issued.nonce,
        expires_at=issued.expires_at.isoformat(),
    )


# ---------- submit ----------

async def _read_capped(f: UploadFile, cap: int) -> bytes:
    data = await f.read(cap + 1)
    if len(data) > cap:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")
    return data


def _base_mime(value: str | None) -> str | None:
    """Strip parameters like ';codecs=vp9' from a Content-Type so allow-list
    membership compares cleanly. Browsers commonly send composite media-types
    like `video/webm;codecs=vp9` — the codec is metadata, not part of the
    media type registration."""
    if not value:
        return None
    return value.split(";", 1)[0].strip().lower()


def _check_mime(payload: bytes, header: str | None, allowed: set[str], label: str) -> None:
    header_base = _base_mime(header)
    if header_base not in allowed:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{label} declared content-type {header!r} not allowed",
        )
    detected = sniff(payload)
    if detected not in allowed:
        # The header lied (or the bytes are corrupt). Refuse rather than feed
        # arbitrary bytes to OpenCV / FFmpeg downstream.
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{label} bytes do not match a permitted format (detected={detected!r})",
        )
    if detected != header_base:
        # Header and content disagree but both happen to be allowed types —
        # likely a benign mislabel from the browser. Log and proceed.
        log.info(
            "mime_mismatch_allowed",
            label=label, declared=header, detected=detected,
        )


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
    user: CurrentUser = Depends(rate_limit(bucket="submit")),
    db: Session = Depends(get_db),
) -> VerifyAccepted:
    if challenge not in ALLOWED_CHALLENGES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Unknown challenge")
    if not _SAFE_CONTRACT_ID.match(contract_id):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid contract_id")
    if not _SAFE_NONCE.match(nonce):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid nonce format")

    # Atomic anti-replay: nonce must exist, belong to this user, match the
    # challenge declared, not be expired, and not have been consumed.
    ok, reason = nonce_store.consume(nonce, user.id, challenge, db)
    if not ok:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid nonce ({reason})")

    id_bytes = await _read_capped(id_image, _settings.max_id_bytes)
    vid_bytes = await _read_capped(selfie_video, _settings.max_video_bytes)
    _check_mime(id_bytes, id_image.content_type, ALLOWED_IMG, "id_image")
    _check_mime(vid_bytes, selfie_video.content_type, ALLOWED_VID, "selfie_video")

    sha_id = hashlib.sha256(id_bytes).hexdigest()
    sha_vid = hashlib.sha256(vid_bytes).hexdigest()

    job_id = str(uuid.uuid4())

    # Encrypt + upload to S3 first, then commit the DB row that references
    # them. If anything between the first upload and the DB commit fails we
    # MUST delete the just-uploaded blobs — otherwise the bucket accumulates
    # orphaned ciphertext+meta pairs nothing in the system knows about, and
    # they're undecryptable so they can't even be cleaned up by the user.
    id_ref = store_encrypted_blob(user.id, job_id, "id", id_bytes, db)
    try:
        vid_ref = store_encrypted_blob(user.id, job_id, "video", vid_bytes, db)
    except Exception:
        # Video upload failed after the ID already landed in S3 — drop it.
        try:
            delete_blob(id_ref)
        except Exception as cleanup_err:
            log.error("orphan_cleanup_failed", phase="video_upload", err=str(cleanup_err))
        raise

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
    try:
        db.commit()
    except Exception:
        db.rollback()
        # DB rejected the row → S3 blobs are now orphaned. Delete them so
        # bucket usage stays consistent with the Verification table.
        for ref in (id_ref, vid_ref):
            try:
                delete_blob(ref)
            except Exception as cleanup_err:
                log.error("orphan_cleanup_failed", phase="db_commit", err=str(cleanup_err))
        raise

    # Enqueue async — import lazily so the API stays bootable without Celery during tests
    from app.workers.tasks import verify_identity_task

    verify_identity_task.delay(job_id=job_id)
    log.info("verification_submitted", job_id=job_id, contract_id=contract_id)

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
