"""Admin endpoints — currently exposes the GDPR Art. 17 / LFPDPPP Art. 23
crypto-shred operation.

Authentication: requires a JWT carrying `role: admin`. The default user
token does NOT grant access; mint admin tokens out-of-band only for the
operations team.

Rate-limited (10/min/admin) so a stolen admin token can't sweep-erase the
tenant in a single burst — every privileged call is also written to
admin_audit_events for compliance traceability.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from redis.exceptions import RedisError
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.security import decode_token
from app.db.session import get_db
from app.models.verification import AdminAuditEvent
from app.services.crypto_vault import erase_user
from app.services.redis_client import get_redis

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
log = get_logger(__name__)

_ADMIN_RATE_PER_MIN = 10


def _require_admin(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_token(token)
    except Exception as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from e
    if payload.get("role") != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return str(payload["sub"])


def _admin_rate_check(actor: str) -> None:
    """Per-admin sliding-window cap. Redis-backed; fail-open on Redis blip
    so an outage doesn't lock out compliance personnel. Cap is intentionally
    LOW (10/min) — admin operations are never bulk."""
    minute_window = int(time.time() // 60)
    key = f"rl:admin:{actor}:{minute_window}".encode()
    try:
        r = get_redis()
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, 70)
        count, _ = pipe.execute()
        if int(count) > _ADMIN_RATE_PER_MIN:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"Admin rate limit exceeded ({_ADMIN_RATE_PER_MIN}/min)",
            )
    except RedisError:
        # Fail-open: better to risk a few extra admin calls than to wedge the
        # compliance team during a Redis blip. The operation is still audited.
        log.warning("admin_rate_limit_redis_unavailable", actor=actor)


@router.post("/users/{user_id}/erase", status_code=status.HTTP_202_ACCEPTED)
def erase_user_endpoint(
    user_id: str,
    request: Request,
    actor: str = Depends(_require_admin),
    db: Session = Depends(get_db),
) -> dict:
    """Crypto-shred a user. Marks their KEK revoked; subsequent decryption
    attempts on any of their blobs raise UserKekRevoked. A periodic sweeper
    hard-deletes the row after a retention window; the data is already
    cryptographically inaccessible from this point onward.

    Every call (intent + outcome) lands in admin_audit_events with the
    actor's JWT subject + the target user_id + the source IP — required by
    LFPDPPP Art. 22 traceability for privileged data operations.
    """
    _admin_rate_check(actor)

    # Audit BEFORE the operation so even a crash mid-erase leaves a trail.
    src_ip = (request.client.host if request.client else None) or "unknown"
    db.add(AdminAuditEvent(
        actor=actor,
        action="user.erase",
        target=user_id,
        detail={"src_ip": src_ip},
    ))
    db.commit()

    revoked = erase_user(user_id, db)

    # Append the outcome as a second event so the trail captures both intent
    # and result. Two-row pattern is idempotent and survives partial failures.
    db.add(AdminAuditEvent(
        actor=actor,
        action="user.erase.result",
        target=user_id,
        detail={"revoked": revoked, "src_ip": src_ip},
    ))
    db.commit()

    log.warning(
        "admin_erase", actor=actor, target=user_id, revoked=revoked, src_ip=src_ip,
    )
    return {"user_id": user_id, "revoked": revoked, "actor": actor}
