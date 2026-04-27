"""Persistent server-issued challenge nonces.

Backed by Postgres (table `issued_nonces`) so /submit can prove that:
  1. The nonce was actually issued by /challenge.
  2. It hasn't expired.
  3. It hasn't been consumed before.
  4. It was issued to the same user that's now submitting.

Without this layer the receipt's `nonce` is attacker-controlled — a captured
video could be re-submitted indefinitely with arbitrary nonce strings.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.nonce import IssuedNonce

log = get_logger(__name__)

DEFAULT_TTL_MIN = 5


def issue(
    user_id: str,
    challenge: str,
    db: Session,
    ttl_minutes: int = DEFAULT_TTL_MIN,
) -> IssuedNonce:
    import secrets

    record = IssuedNonce(
        nonce=secrets.token_urlsafe(24)[:64],
        user_id=user_id,
        challenge=challenge,
        expires_at=datetime.now(UTC) + timedelta(minutes=ttl_minutes),
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError:
        # Astronomically unlikely token collision — retry once with a fresh value.
        db.rollback()
        record.nonce = secrets.token_urlsafe(24)[:64]
        db.add(record)
        db.commit()
    return record


def _as_aware(dt: datetime | None) -> datetime | None:
    """SQLite drops the tzinfo on round-trip; treat any naive value as UTC.
    Postgres returns aware datetimes natively so this is a no-op there.
    """
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def consume(nonce: str, user_id: str, challenge: str, db: Session) -> tuple[bool, str | None]:
    """Atomically validate + mark a nonce as used. Returns (ok, reason)."""
    row = db.get(IssuedNonce, nonce)
    if row is None:
        return False, "unknown_nonce"
    if row.user_id != user_id:
        return False, "wrong_owner"
    if row.challenge != challenge:
        return False, "wrong_challenge"
    if row.consumed_at is not None:
        return False, "already_consumed"
    expires_at = _as_aware(row.expires_at)
    if expires_at is None or expires_at <= datetime.now(UTC):
        return False, "expired"
    row.consumed_at = datetime.now(UTC)
    db.commit()
    return True, None


def purge_expired(db: Session) -> int:
    """Run from a periodic Celery beat to keep the table small."""
    from sqlalchemy import delete

    now = datetime.now(UTC)
    res = db.execute(
        delete(IssuedNonce).where(IssuedNonce.expires_at < now - timedelta(hours=1))
    )
    db.commit()
    n = res.rowcount or 0
    if n:
        log.info("nonces_purged", count=n)
    return n
