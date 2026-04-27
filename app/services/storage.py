"""S3 / MinIO storage of encrypted biometric blobs.

Storage layout: s3://{bucket}/{user_id}/{job_id}/{kind}.bin (ciphertext)
and a sibling .meta.json carrying the nonce, wrapped DEK, AAD, and the
owning user_id (so open_blob can fetch the matching per-user KEK).

Both `user_id` and `job_id` are validated against a strict regex before any
key construction. JWT claims arrive from external issuers; without
sanitisation a malicious `sub` like `../tenantB/jobX` could traverse into
another tenant's prefix.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass

import boto3
from botocore.client import Config
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.crypto_vault import SealedBlob, open_blob, seal

log = get_logger(__name__)
_settings = get_settings()

# Restrictive allow-list. Both UUIDs (job_id) and email-or-uuid-shaped
# subjects fit comfortably under 64 chars of [A-Za-z0-9_-]. Anything else is
# refused to keep the S3 prefix tree free of traversal sequences.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_KIND_ALLOWED = {"id", "video"}


@dataclass
class BlobRef:
    bucket: str
    ciphertext_key: str
    meta_key: str

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "ciphertext_key": self.ciphertext_key,
            "meta_key": self.meta_key,
        }


def _client():
    return boto3.client(
        "s3",
        endpoint_url=_settings.s3_endpoint,
        aws_access_key_id=_settings.s3_access_key,
        aws_secret_access_key=_settings.s3_secret_key,
        region_name=_settings.s3_region,
        use_ssl=_settings.s3_use_ssl,
        config=Config(signature_version="s3v4"),
    )


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.match(value):
        raise ValueError(f"Invalid {label}: must match {_SAFE_ID.pattern}")
    return value


def _key(user_id: str, job_id: str, kind: str) -> tuple[str, str]:
    user_id = _safe_id(user_id, "user_id")
    job_id = _safe_id(job_id, "job_id")
    if kind not in _KIND_ALLOWED:
        raise ValueError(f"Invalid blob kind: {kind!r}")
    base = f"{user_id}/{job_id}/{kind}"
    return f"{base}.bin", f"{base}.meta.json"


def store_encrypted_blob(
    user_id: str,
    job_id: str,
    kind: str,
    plaintext: bytes,
    db: Session,
) -> BlobRef:
    user_id = _safe_id(user_id, "user_id")
    job_id = _safe_id(job_id, "job_id")
    aad = f"{user_id}|{job_id}|{kind}".encode()
    sealed = seal(plaintext, aad, user_id=user_id, db=db)
    ct_key, meta_key = _key(user_id, job_id, kind)

    s3 = _client()
    s3.put_object(
        Bucket=_settings.s3_bucket,
        Key=ct_key,
        Body=sealed.ciphertext,
        ContentType="application/octet-stream",
        Metadata={"kind": kind, "user_id": user_id, "job_id": job_id},
    )
    meta = {
        "nonce_b64": base64.b64encode(sealed.nonce).decode(),
        "wrapped_dek_b64": base64.b64encode(sealed.wrapped_dek).decode(),
        "aad_b64": base64.b64encode(sealed.aad).decode(),
        "user_id": sealed.user_id,
        "kms_key_id": sealed.kms_key_id,
    }
    s3.put_object(
        Bucket=_settings.s3_bucket,
        Key=meta_key,
        Body=json.dumps(meta).encode(),
        ContentType="application/json",
    )
    log.info("blob_stored", user_id=user_id, job_id=job_id, kind=kind, key=ct_key)
    return BlobRef(bucket=_settings.s3_bucket, ciphertext_key=ct_key, meta_key=meta_key)


def load_encrypted_blob(ref: BlobRef, db: Session) -> bytes:
    s3 = _client()
    ct = s3.get_object(Bucket=ref.bucket, Key=ref.ciphertext_key)["Body"].read()
    meta_raw = s3.get_object(Bucket=ref.bucket, Key=ref.meta_key)["Body"].read()
    meta = json.loads(meta_raw)
    sealed = SealedBlob(
        ciphertext=ct,
        nonce=base64.b64decode(meta["nonce_b64"]),
        wrapped_dek=base64.b64decode(meta["wrapped_dek_b64"]),
        aad=base64.b64decode(meta["aad_b64"]),
        user_id=meta.get("user_id", ""),
        kms_key_id=meta["kms_key_id"],
    )
    return open_blob(sealed, db)


def delete_blob(ref: BlobRef) -> None:
    s3 = _client()
    s3.delete_objects(
        Bucket=ref.bucket,
        Delete={"Objects": [{"Key": ref.ciphertext_key}, {"Key": ref.meta_key}]},
    )
    log.info("blob_deleted", key=ref.ciphertext_key)
