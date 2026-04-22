"""S3 / MinIO storage of encrypted biometric blobs.

Storage layout: s3://{bucket}/{user_id}/{job_id}/{kind}.bin (ciphertext)
and a sibling .meta.json carrying the nonce, wrapped DEK, and AAD.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import boto3
from botocore.client import Config

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.crypto_vault import SealedBlob, open_blob, seal

log = get_logger(__name__)
_settings = get_settings()


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


def _key(user_id: str, job_id: str, kind: str) -> tuple[str, str]:
    base = f"{user_id}/{job_id}/{kind}"
    return f"{base}.bin", f"{base}.meta.json"


def store_encrypted_blob(user_id: str, job_id: str, kind: str, plaintext: bytes) -> BlobRef:
    aad = f"{user_id}|{job_id}|{kind}".encode()
    sealed = seal(plaintext, aad)
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
        "nonce_b64": __import__("base64").b64encode(sealed.nonce).decode(),
        "wrapped_dek_b64": __import__("base64").b64encode(sealed.wrapped_dek).decode(),
        "aad_b64": __import__("base64").b64encode(sealed.aad).decode(),
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


def load_encrypted_blob(ref: BlobRef) -> bytes:
    import base64

    s3 = _client()
    ct = s3.get_object(Bucket=ref.bucket, Key=ref.ciphertext_key)["Body"].read()
    meta_raw = s3.get_object(Bucket=ref.bucket, Key=ref.meta_key)["Body"].read()
    meta = json.loads(meta_raw)
    sealed = SealedBlob(
        ciphertext=ct,
        nonce=base64.b64decode(meta["nonce_b64"]),
        wrapped_dek=base64.b64decode(meta["wrapped_dek_b64"]),
        aad=base64.b64decode(meta["aad_b64"]),
        kms_key_id=meta["kms_key_id"],
    )
    return open_blob(sealed)


def delete_blob(ref: BlobRef) -> None:
    s3 = _client()
    s3.delete_objects(
        Bucket=ref.bucket,
        Delete={"Objects": [{"Key": ref.ciphertext_key}, {"Key": ref.meta_key}]},
    )
    log.info("blob_deleted", key=ref.ciphertext_key)
