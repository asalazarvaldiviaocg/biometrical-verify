"""Envelope encryption for biometric blobs.

In production: KEK lives in AWS KMS / HashiCorp Vault. Per-blob DEK is
generated, used for AES-256-GCM encryption in memory, then wrapped with
the KEK and stored alongside the ciphertext. Crypto-shred a user by
deleting their KEK — every wrapped DEK becomes unrecoverable.

In dev: a 32-byte MASTER_KEY in .env acts as a single KEK. NEVER use this
backend in production.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class SealedBlob:
    ciphertext: bytes
    nonce: bytes
    wrapped_dek: bytes
    aad: bytes
    kms_key_id: str

    def to_record(self) -> dict:
        return {
            "ciphertext_b64": base64.b64encode(self.ciphertext).decode(),
            "nonce_b64": base64.b64encode(self.nonce).decode(),
            "wrapped_dek_b64": base64.b64encode(self.wrapped_dek).decode(),
            "aad_b64": base64.b64encode(self.aad).decode(),
            "kms_key_id": self.kms_key_id,
        }

    @classmethod
    def from_record(cls, rec: dict) -> SealedBlob:
        return cls(
            ciphertext=base64.b64decode(rec["ciphertext_b64"]),
            nonce=base64.b64decode(rec["nonce_b64"]),
            wrapped_dek=base64.b64decode(rec["wrapped_dek_b64"]),
            aad=base64.b64decode(rec["aad_b64"]),
            kms_key_id=rec["kms_key_id"],
        )


class KMSBackend(Protocol):
    key_id: str

    def wrap(self, dek: bytes) -> bytes: ...
    def unwrap(self, wrapped: bytes) -> bytes: ...


class LocalKMS:
    """Dev-only KEK from env-provided MASTER_KEY (base64, 32 bytes)."""

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
        ct = AESGCM(self._kek).encrypt(nonce, dek, b"dek-wrap")
        return nonce + ct

    def unwrap(self, wrapped: bytes) -> bytes:
        nonce, ct = wrapped[:12], wrapped[12:]
        return AESGCM(self._kek).decrypt(nonce, ct, b"dek-wrap")


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


def seal(plaintext: bytes, aad: bytes) -> SealedBlob:
    """Encrypt plaintext with a fresh DEK; wrap the DEK under the KEK."""
    kms = get_kms()
    dek = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    ct = AESGCM(dek).encrypt(nonce, plaintext, aad)
    wrapped = kms.wrap(dek)
    # Best-effort wipe — Python doesn't truly zeroize, but minimise references
    del dek
    return SealedBlob(
        ciphertext=ct,
        nonce=nonce,
        wrapped_dek=wrapped,
        aad=aad,
        kms_key_id=kms.key_id,
    )


def open_blob(sealed: SealedBlob) -> bytes:
    kms = get_kms()
    dek = kms.unwrap(sealed.wrapped_dek)
    try:
        return AESGCM(dek).decrypt(sealed.nonce, sealed.ciphertext, sealed.aad)
    finally:
        del dek
