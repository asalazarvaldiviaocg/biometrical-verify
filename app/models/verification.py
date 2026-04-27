from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

# Polymorphic types: real JSONB / native UUID on Postgres, plain JSON / string
# on SQLite so unit tests can `create_all()` without a Postgres engine. The
# storage layout is identical at the application level.
_JSON_VARIANT = JSON().with_variant(JSONB(), "postgresql")
_UUID_VARIANT = String(36).with_variant(UUID(as_uuid=False), "postgresql")


def _uuid() -> str:
    return str(uuid.uuid4())


class Verification(Base):
    __tablename__ = "verifications"

    id: Mapped[str] = mapped_column(_UUID_VARIANT, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    contract_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued", nullable=False)
    decision: Mapped[str | None] = mapped_column(String(16), nullable=True)

    challenge: Mapped[str] = mapped_column(String(32), default="blink_twice")
    nonce: Mapped[str | None] = mapped_column(String(64), nullable=True)

    sha_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sha_video: Mapped[str | None] = mapped_column(String(64), nullable=True)

    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    liveness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    challenge_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    deepfake_suspicious: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    model: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True)

    id_blob_ref: Mapped[dict | None] = mapped_column(_JSON_VARIANT, nullable=True)
    video_blob_ref: Mapped[dict | None] = mapped_column(_JSON_VARIANT, nullable=True)
    receipt: Mapped[dict | None] = mapped_column(_JSON_VARIANT, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    audits: Mapped[list[AuditEvent]] = relationship(
        back_populates="verification", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_verifications_user_status", "user_id", "status"),
        Index("ix_verifications_status_created", "status", "created_at"),
    )


class AuditEvent(Base):
    """Append-only audit log. Operators read this; raw biometrics never live here."""

    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(_UUID_VARIANT, primary_key=True, default=_uuid)
    verification_id: Mapped[str] = mapped_column(
        _UUID_VARIANT, ForeignKey("verifications.id", ondelete="CASCADE")
    )
    event: Mapped[str] = mapped_column(String(64), nullable=False)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    verification: Mapped[Verification] = relationship(back_populates="audits")


class AdminAuditEvent(Base):
    """Admin-action audit trail (separate from per-verification AuditEvent).

    Captures privileged operations (crypto-shred erasure, key rotation, etc.)
    with the actor JWT subject and the target so compliance reviewers can
    reconstruct who did what without trawling app logs.
    """

    __tablename__ = "admin_audit_events"

    id: Mapped[str] = mapped_column(_UUID_VARIANT, primary_key=True, default=_uuid)
    actor: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    action: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    target: Mapped[str | None] = mapped_column(String(128), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
