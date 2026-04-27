"""Per-user Key Encryption Key for crypto-shred GDPR Art. 17 / LFPDPPP Art. 23.

Each user gets a 32-byte AES-256 KEK on first verification. That KEK is
wrapped under the global master KEK (LocalKMS or AWSKMS) and stored here.
Per-blob DEKs are wrapped under the per-user KEK, not the global one.

Erasure path:
    1. DELETE FROM user_keks WHERE user_id = $1
    2. Every wrapped DEK referencing that user becomes mathematically
       unrecoverable in milliseconds — no need to scrub object storage.

Cost: one extra `SELECT user_keks WHERE user_id` per encrypt/decrypt call.
With pool_pre_ping + idx on PK that's <1 ms typical.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, LargeBinary, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class UserKek(Base):
    __tablename__ = "user_keks"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # wrapped_kek = AES-256-GCM(global_kek, user_kek_plaintext, nonce || aad)
    # Layout in bytes: [12-byte nonce][ciphertext + GCM tag]
    wrapped_kek: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # Identifies the global KEK that wrapped this user-KEK so we can rotate
    # the master without losing context (each row remembers which master
    # encrypted it).
    kms_key_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Marker for soft-revocation (set when erase_user is called). The row is
    # then DELETEd by a follow-up sweeper, but immediate effect is achieved
    # by treating any non-null value here as "no KEK" in code.
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
