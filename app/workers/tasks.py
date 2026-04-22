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
from datetime import UTC, datetime

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import sign_receipt
from app.db.session import SessionLocal
from app.models.verification import AuditEvent, Verification
from app.services import deepfake, face_engine, id_parser, liveness
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


def _decide(similarity: float, is_live: bool, df_suspicious: bool) -> tuple[str, str]:
    if df_suspicious:
        return "REJECT", "Deepfake heuristics triggered"
    if not is_live:
        return "REJECT", "Liveness check failed"
    if similarity >= _settings.approve_similarity_min:
        return "APPROVE", "Match above approve threshold"
    if similarity >= _settings.review_similarity_min:
        return "REVIEW", "Borderline similarity — manual review"
    return "REJECT", "Similarity below review threshold"


@celery.task(name="verify.identity", bind=True, max_retries=2, acks_late=True)
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
        id_bytes = load_encrypted_blob(id_ref)
        vid_bytes = load_encrypted_blob(vid_ref)

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
        live = liveness.run_liveness(vpath, frame_bgr, challenge=record.challenge)  # type: ignore[arg-type]
        df = deepfake.analyse(frame_bgr, vpath)
        db.add(AuditEvent(verification_id=job_id, event="liveness", detail=live.to_public()))
        db.add(AuditEvent(verification_id=job_id, event="deepfake", detail=df.to_public()))

        # Match
        match = face_engine.compare(id_bytes, frame_bytes)
        db.add(AuditEvent(verification_id=job_id, event="match", detail=match.to_public()))

        decision, reason = _decide(match.similarity, live.is_live, df.suspicious)

        return _finalise(
            db, record, decision=decision, reason=reason,
            similarity=match.similarity, distance=match.distance,
            threshold=match.threshold, liveness_score=live.score,
            challenge_passed=live.challenge_passed,
            deepfake_suspicious=df.suspicious, model=match.model,
        )

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
    receipt = sign_receipt({
        "job_id": record.id,
        "user_id": record.user_id,
        "contract_id": record.contract_id,
        "sha_id": record.sha_id,
        "sha_video": record.sha_video,
        "decision": decision,
        "similarity": similarity,
        "model": model,
        "liveness_score": liveness_score,
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
