"""Envelope encryption with per-user KEK for crypto-shred GDPR Art. 17.

Three layers of keys:

    1. Master KEK     — held in AWS KMS (or LocalKMS in dev). Wraps the user-KEKs
                        at rest.
    2. User KEK       — 32 random bytes per user, wrapped under the master KEK
                        and stored in `user_keks`. Deleting this row makes every
                        wrapped DEK unrecoverable for that user.
    3. Per-blob DEK   — 32 random bytes per blob, used for AES-256-GCM, wrapped
                        under the user KEK and stored alongside the ciphertext.

Erasure path is `erase_user(user_id, db)`: marks `revoked_at`, then a sweeper
hard-deletes the row. From the moment `revoked_at` is set, `unwrap_user_kek`
returns None and every blob ref linked to the user becomes unrecoverable —
operationally that is GDPR-compliant erasure even before disk is overwritten.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.user_kek import UserKek

log = get_logger(__name__)


class UserKekRevoked(RuntimeError):
    """Raised when a blob is read for a user whose KEK has been erased."""


@dataclass
class SealedBlob:
    ciphertext: bytes
    nonce: bytes
    wrapped_dek: bytes
    aad: bytes
    user_id: str
    kms_key_id: str

    def to_record(self) -> dict:
        return {
            "ciphertext_b64": base64.b64encode(self.ciphertext).decode(),
            "nonce_b64": base64.b64encode(self.nonce).decode(),
            "wrapped_dek_b64": base64.b64encode(self.wrapped_dek).decode(),
            "aad_b64": base64.b64encode(self.aad).decode(),
            "user_id": self.user_id,
            "kms_key_id": self.kms_key_id,
        }

    @classmethod
    def from_record(cls, rec: dict) -> SealedBlob:
        return cls(
            ciphertext=base64.b64decode(rec["ciphertext_b64"]),
            nonce=base64.b64decode(rec["nonce_b64"]),
            wrapped_dek=base64.b64decode(rec["wrapped_dek_b64"]),
            aad=base64.b64decode(rec["aad_b64"]),
            user_id=rec.get("user_id", ""),
            kms_key_id=rec["kms_key_id"],
        )


class KMSBackend(Protocol):
    key_id: str

    def wrap(self, dek: bytes) -> bytes: ...
    def unwrap(self, wrapped: bytes) -> bytes: ...


class LocalKMS:
    """Dev-only master KEK from env-provided MASTER_KEY (base64, 32 bytes)."""

    def __init__(self) -> None:
        s = get_settings()
        if not s.master_key:
            raise RuntimeError("MASTER_KEY missing for local KMS backend")
        self._kek = base64.b64decode(s.master_key)
        if len(self._kek) != 32:
            raise RuntimeError("MASTER_KEY must decode to 32 bytes")
        self.key_id = "local/master"

    def wrap(self, dek: bytes) -> bytes:
        nonce = os.urandom(12)
        ct = AESGCM(self._kek).encrypt(nonce, dek, b"kek-wrap")
        return nonce + ct

    def unwrap(self, wrapped: bytes) -> bytes:
        nonce, ct = wrapped[:12], wrapped[12:]
        return AESGCM(self._kek).decrypt(nonce, ct, b"kek-wrap")


class AWSKMS:
    def __init__(self) -> None:
        import boto3

        s = get_settings()
        if not s.aws_kms_key_id:
            raise RuntimeError("AWS_KMS_KEY_ID required for aws KMS backend")
        self._client = boto3.client("kms", region_name=s.s3_region)
        self.key_id = s.aws_kms_key_id

    def wrap(self, dek: bytes) -> bytes:
        r = self._client.encrypt(KeyId=self.key_id, Plaintext=dek)
        return r["CiphertextBlob"]

    def unwrap(self, wrapped: bytes) -> bytes:
        r = self._client.decrypt(KeyId=self.key_id, CiphertextBlob=wrapped)
        return r["Plaintext"]


_kms: KMSBackend | None = None


def get_kms() -> KMSBackend:
    global _kms
    if _kms is not None:
        return _kms
    s = get_settings()
    _kms = AWSKMS() if s.kms_backend == "aws" else LocalKMS()
    log.info("kms_backend_loaded", backend=s.kms_backend, key_id=_kms.key_id)
    return _kms


# ── Per-user KEK lifecycle ────────────────────────────────────────────────

def _new_user_kek_plaintext() -> bytes:
    return AESGCM.generate_key(bit_length=256)


def get_or_create_user_kek(user_id: str, db: Session) -> bytes:
    """Returns the plaintext per-user KEK, creating one on first use."""
    row = db.get(UserKek, user_id)
    if row is not None and row.revoked_at is None:
        return get_kms().unwrap(row.wrapped_kek)
    if row is not None and row.revoked_at is not None:
        raise UserKekRevoked(f"User KEK revoked for user_id={user_id}")
    plaintext = _new_user_kek_plaintext()
    kms = get_kms()
    wrapped = kms.wrap(plaintext)
    db.add(UserKek(user_id=user_id, wrapped_kek=wrapped, kms_key_id=kms.key_id))
    db.commit()
    log.info("user_kek_created", user_id=user_id, kms_key_id=kms.key_id)
    return plaintext


def unwrap_user_kek(user_id: str, db: Session) -> bytes | None:
    """Returns the plaintext KEK for an existing user, or None if revoked/missing."""
    row = db.get(UserKek, user_id)
    if row is None or row.revoked_at is not None:
        return None
    return get_kms().unwrap(row.wrapped_kek)


def erase_user(user_id: str, db: Session) -> bool:
    """Crypto-shred entry point. Marks the KEK revoked; sweeper hard-deletes."""
    row = db.get(UserKek, user_id)
    if row is None:
        return False
    if row.revoked_at is None:
        row.revoked_at = datetime.now(UTC)
        db.commit()
        log.warning("user_kek_revoked", user_id=user_id)
    return True


def purge_revoked_user_keks(db: Session) -> int:
    """Sweeper — hard-delete revoked rows. Run from a periodic Celery beat."""
    rows = db.query(UserKek).filter(UserKek.revoked_at.is_not(None)).all()
    for r in rows:
        db.delete(r)
    db.commit()
    if rows:
        log.info("user_kek_purged", count=len(rows))
    return len(rows)


# ── Blob seal / open ──────────────────────────────────────────────────────

def seal(plaintext: bytes, aad: bytes, *, user_id: str, db: Session) -> SealedBlob:
    """Encrypt plaintext with a fresh DEK; wrap the DEK under the per-user KEK."""
    user_kek = get_or_create_user_kek(user_id, db)
    dek = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    ct = AESGCM(dek).encrypt(nonce, plaintext, aad)
    wrap_nonce = os.urandom(12)
    wrapped = wrap_nonce + AESGCM(user_kek).encrypt(wrap_nonce, dek, b"dek-wrap")
    del dek
    del user_kek
    return SealedBlob(
        ciphertext=ct,
        nonce=nonce,
        wrapped_dek=wrapped,
        aad=aad,
        user_id=user_id,
        kms_key_id=get_kms().key_id,
    )


def open_blob(sealed: SealedBlob, db: Session) -> bytes:
    user_kek = unwrap_user_kek(sealed.user_id, db)
    if user_kek is None:
        raise UserKekRevoked(
            f"Cannot decrypt blob: user KEK missing/revoked for user_id={sealed.user_id}"
        )
    wrap_nonce, wrap_ct = sealed.wrapped_dek[:12], sealed.wrapped_dek[12:]
    dek = AESGCM(user_kek).decrypt(wrap_nonce, wrap_ct, b"dek-wrap")
    try:
        return AESGCM(dek).decrypt(sealed.nonce, sealed.ciphertext, sealed.aad)
    finally:
        del dek
        del user_kek
