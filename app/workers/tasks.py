"""End-to-end verification pipeline run by Celery workers.

Steps:
  1. Load encrypted ID + video from object store, decrypt in memory.
  2. Inspect ID quality (blur, glare, has-face).
  3. Pick sharpest selfie frame.
  4. Run liveness (passive + challenge).
  5. Run deepfake heuristics.
  6. Extract embeddings + compare.
  7. Decide APPROVE / REVIEW / REJECT.
  8. Sign a receipt and persist outcome.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
from celery.exceptions import SoftTimeLimitExceeded

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import sign_receipt
from app.db.session import SessionLocal
from app.models.verification import AuditEvent, Verification
from app.services import deepfake, face_engine, id_parser, liveness, nonce_store
from app.services.crypto_vault import UserKekRevoked, purge_revoked_user_keks
from app.services.storage import BlobRef, load_encrypted_blob
from app.workers.celery_app import celery

log = get_logger(__name__)
_settings = get_settings()


def _decode(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def _decide(
    similarity: float,
    distance: float,
    match_verified: bool,
    is_live: bool,
    df_suspicious: bool,
) -> tuple[str, str]:
    if df_suspicious:
        return "REJECT", "Deepfake heuristics triggered"
    if not is_live:
        return "REJECT", "Liveness check failed"
    # APPROVE requires BOTH the model-author threshold (distance ≤ 0.68 for
    # ArcFace on LFW) AND our configured similarity floor. Previously the
    # decision used only `similarity ≥ 0.40`, which approves matches that
    # the model considers ambiguous and quietly degraded the 99.83% LFW
    # accuracy claimed in the README.
    if match_verified and similarity >= _settings.approve_similarity_min:
        return "APPROVE", "Match above approve threshold"
    if similarity >= _settings.review_similarity_min:
        return "REVIEW", "Borderline similarity — manual review"
    return "REJECT", "Similarity below review threshold"


@celery.task(
    name="verify.identity",
    bind=True,
    max_retries=2,
    acks_late=True,
    # Cold-start budget: DeepFace (~30 s warm-load on first call) plus
    # MediaPipe FaceMesh (~5-10 s) plus AES-GCM unwrap, PIL decode,
    # frame-extract, and BLE pipeline. Earlier limit of 30 s timed out
    # EVERY first request after a quiet container window, kicking the
    # task into max_retries=2 retry storms. 90 s soft / 120 s hard
    # tolerates cold-start while still bounding pathological frames.
    soft_time_limit=90,
    time_limit=120,
)
def verify_identity_task(self, job_id: str) -> dict:
    db = SessionLocal()
    record: Verification | None = None
    vpath: str | None = None
    try:
        record = db.get(Verification, job_id)
        if record is None:
            log.error("verification_missing", job_id=job_id)
            return {"status": "error", "reason": "record missing"}

        record.status = "running"
        db.commit()

        id_ref = BlobRef(**record.id_blob_ref)
        vid_ref = BlobRef(**record.video_blob_ref)
        try:
            id_bytes = load_encrypted_blob(id_ref, db)
            vid_bytes = load_encrypted_blob(vid_ref, db)
        except UserKekRevoked as e:
            return _finalise(
                db, record, decision="REJECT",
                reason=f"User KEK revoked — blob unrecoverable ({e})",
                similarity=None, distance=None, threshold=None,
                liveness_score=None, challenge_passed=None,
                deepfake_suspicious=None, model=None,
            )

        # ID quality
        id_q = id_parser.assess(id_bytes)
        db.add(AuditEvent(
            verification_id=job_id, event="id_quality", detail=id_q.to_public()
        ))
        if not id_q.acceptable:
            return _finalise(
                db, record, decision="REJECT",
                reason=f"ID quality unacceptable ({id_q.notes})",
                similarity=None, distance=None, threshold=None,
                liveness_score=None, challenge_passed=None,
                deepfake_suspicious=None, model=None,
            )

        # Persist video to a temp file for OpenCV / MediaPipe
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            tf.write(vid_bytes)
            vpath = tf.name

        # Best frame for embedding + passive liveness
        frame_bytes = face_engine.best_frame_from_video(vpath)
        frame_bgr = _decode(frame_bytes)

        # Liveness + deepfake
        challenge = record.challenge if record.challenge in {
            "blink_twice", "blink_once", "turn_head", "none",
        } else "blink_twice"
        live = liveness.run_liveness(vpath, frame_bgr, challenge=challenge)  # type: ignore[arg-type]
        df = deepfake.analyse(frame_bgr, vpath)
        db.add(AuditEvent(verification_id=job_id, event="liveness", detail=live.to_public()))
        db.add(AuditEvent(verification_id=job_id, event="deepfake", detail=df.to_public()))

        # Match
        match = face_engine.compare(id_bytes, frame_bytes)
        db.add(AuditEvent(verification_id=job_id, event="match", detail=match.to_public()))

        decision, reason = _decide(
            match.similarity, match.distance, match.verified,
            live.is_live, df.suspicious,
        )

        return _finalise(
            db, record, decision=decision, reason=reason,
            similarity=match.similarity, distance=match.distance,
            threshold=match.threshold, liveness_score=live.score,
            challenge_passed=live.challenge_passed,
            deepfake_suspicious=df.suspicious, model=match.model,
        )

    except SoftTimeLimitExceeded:
        # Worker was about to be killed; mark as error and let the celery
        # retry decorator schedule another attempt.
        log.error("verify_pipeline_timeout", job_id=job_id)
        if record is not None:
            record.status = "error"
            record.reason = "pipeline timed out"
            db.add(AuditEvent(
                verification_id=job_id, event="error", detail={"error": "soft_time_limit"}
            ))
            db.commit()
        return {"status": "error", "reason": "timeout"}

    except Exception as e:
        log.exception("verify_pipeline_failed", job_id=job_id, error=str(e))
        if record is not None:
            record.status = "error"
            record.reason = str(e)[:250]
            db.add(AuditEvent(
                verification_id=job_id, event="error", detail={"error": str(e)[:500]}
            ))
            db.commit()
        try:
            raise self.retry(exc=e, countdown=10)
        except Exception:
            return {"status": "error", "reason": str(e)}

    finally:
        if vpath and os.path.exists(vpath):
            try:
                os.unlink(vpath)
            except OSError:
                pass
        db.close()


def _finalise(
    db, record: Verification, *,
    decision: str, reason: str,
    similarity: float | None, distance: float | None, threshold: float | None,
    liveness_score: float | None, challenge_passed: bool | None,
    deepfake_suspicious: bool | None, model: str | None,
) -> dict:
    # Sign the FULL evidence chain so a verifier reading the receipt offline
    # can independently reconstruct *why* the decision was made:
    #   - challenge_passed proves the active-liveness reply matched the
    #     server-issued nonce/instruction
    #   - deepfake_suspicious proves the GAN/replay heuristics actually ran
    #     and what they returned
    #   - distance + threshold lets a verifier re-derive the match outcome
    #     against any future ArcFace threshold revision
    receipt = sign_receipt({
        "job_id": record.id,
        "user_id": record.user_id,
        "contract_id": record.contract_id,
        "nonce": record.nonce,                     # bound into the signed body
        "challenge": record.challenge,
        "sha_id": record.sha_id,
        "sha_video": record.sha_video,
        "decision": decision,
        "similarity": similarity,
        "distance": distance,
        "threshold": threshold,
        "model": model,
        "liveness_score": liveness_score,
        "challenge_passed": challenge_passed,
        "deepfake_suspicious": deepfake_suspicious,
    })
    record.status = "done"
    record.decision = decision
    record.reason = reason[:250]
    record.similarity = similarity
    record.distance = distance
    record.threshold = threshold
    record.liveness_score = liveness_score
    record.challenge_passed = challenge_passed
    record.deepfake_suspicious = deepfake_suspicious
    record.model = model
    record.receipt = receipt
    record.finished_at = datetime.now(UTC)

    db.add(AuditEvent(
        verification_id=record.id, event="finalised",
        detail={"decision": decision, "reason": reason},
    ))
    db.commit()
    log.info(
        "verification_done", job_id=record.id, decision=decision,
        similarity=similarity, liveness=liveness_score,
    )
    return {"status": "done", "decision": decision, "similarity": similarity}


# ── Periodic sweepers (scheduled by Celery beat) ──────────────────────────

# A `queued` row that's been sitting more than this many seconds without ever
# transitioning to `running` was almost certainly orphaned (broker drop,
# worker crash before pickup, etc.) and is safe to re-enqueue.
_STUCK_QUEUED_AFTER_S = 5 * 60


@celery.task(name="verify.requeue_stuck", acks_late=True)
def requeue_stuck_verifications() -> dict:
    """Re-emit `verify_identity_task` for verifications stuck in `queued`
    state past the cutoff. Without this sweeper, a Celery outage at /submit
    time leaves rows in queued forever — the client polls and never sees
    a result.

    Idempotent: a row that gets re-enqueued more than once still ends up
    finalised exactly once because `verify_identity_task` writes
    `status='running'` first thing.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=_STUCK_QUEUED_AFTER_S)
    db = SessionLocal()
    try:
        rows = (
            db.query(Verification)
            .filter(Verification.status == "queued", Verification.created_at < cutoff)
            .limit(100)
            .all()
        )
        for r in rows:
            verify_identity_task.delay(job_id=r.id)
            db.add(AuditEvent(
                verification_id=r.id,
                event="requeued",
                detail={"reason": "stuck_in_queued", "cutoff_seconds": _STUCK_QUEUED_AFTER_S},
            ))
        if rows:
            db.commit()
            log.warning("requeued_stuck_verifications", count=len(rows))
        return {"requeued": len(rows)}
    finally:
        db.close()


@celery.task(name="verify.purge_expired_nonces", acks_late=True)
def purge_expired_nonces_task() -> dict:
    """Delete issued_nonces rows older than 1 hour past their expiry. Keeps
    the table from growing without bound."""
    db = SessionLocal()
    try:
        n = nonce_store.purge_expired(db)
        return {"purged": n}
    finally:
        db.close()


@celery.task(name="verify.purge_revoked_keks", acks_late=True)
def purge_revoked_keks_task() -> dict:
    """Hard-delete user_keks rows that have been soft-revoked. Crypto-shred
    is already effective from the moment revoked_at is set; this just frees
    the row from the table."""
    db = SessionLocal()
    try:
        n = purge_revoked_user_keks(db)
        return {"purged": n}
    finally:
        db.close()
