from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from app.core.config import get_settings

# ---------- JWT ----------

def create_access_token(subject: str, claims: dict[str, Any] | None = None) -> str:
    s = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=s.jwt_expire_minutes)).timestamp()),
        **(claims or {}),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_alg)


def decode_token(token: str) -> dict[str, Any]:
    s = get_settings()
    return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_alg])


# ---------- Receipt signing (Ed25519) ----------

_signing_key: Ed25519PrivateKey | None = None


def _load_or_create_signing_key() -> Ed25519PrivateKey:
    global _signing_key
    if _signing_key is not None:
        return _signing_key
    s = get_settings()
    if s.receipt_signing_key:
        raw = base64.b64decode(s.receipt_signing_key)
        if len(raw) != 32:
            raise ValueError("RECEIPT_SIGNING_KEY must decode to 32 bytes")
        _signing_key = Ed25519PrivateKey.from_private_bytes(raw)
    else:
        _signing_key = Ed25519PrivateKey.generate()
    return _signing_key


def public_key_b64() -> str:
    pub: Ed25519PublicKey = _load_or_create_signing_key().public_key()
    return base64.b64encode(pub.public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()


def private_key_b64() -> str:
    pk = _load_or_create_signing_key()
    return base64.b64encode(
        pk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    ).decode()


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sign_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    body = {**payload, "signed_at": datetime.now(UTC).isoformat()}
    msg = canonical_json(body)
    sig = _load_or_create_signing_key().sign(msg)
    return {
        "payload": body,
        "signature": base64.b64encode(sig).decode(),
        "alg": "Ed25519",
        "msg_sha256": hashlib.sha256(msg).hexdigest(),
        "public_key": public_key_b64(),
    }


def verify_receipt(receipt: dict[str, Any]) -> bool:
    try:
        body = receipt["payload"]
        sig = base64.b64decode(receipt["signature"])
        pub_raw = base64.b64decode(receipt["public_key"])
        pub = Ed25519PublicKey.from_public_bytes(pub_raw)
        pub.verify(sig, canonical_json(body))
        return True
    except Exception:
        return False
