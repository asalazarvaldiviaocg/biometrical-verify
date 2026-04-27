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
        "iss": s.jwt_issuer,
        "aud": s.jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=s.jwt_expire_minutes)).timestamp()),
        **(claims or {}),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_alg)


def decode_token(token: str) -> dict[str, Any]:
    s = get_settings()
    return jwt.decode(
        token,
        s.jwt_secret,
        algorithms=[s.jwt_alg],
        audience=s.jwt_audience,
        issuer=s.jwt_issuer,
        options={"require": ["sub", "exp", "iat", "iss", "aud"]},
    )


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
        # Production validation in Settings.model_validator already refuses to
        # boot without a fixed key. This branch only runs in dev/test.
        if s.is_prod:
            raise RuntimeError(
                "RECEIPT_SIGNING_KEY is required in production "
                "(autogenerating on each restart would invalidate every prior receipt)"
            )
        _signing_key = Ed25519PrivateKey.generate()
    return _signing_key


def public_key_b64() -> str:
    pub: Ed25519PublicKey = _load_or_create_signing_key().public_key()
    return base64.b64encode(pub.public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()


def public_key_id() -> str:
    """Stable short identifier for the active receipt-signing pubkey.
    Verifiers pin this against the value served by /api/v1/keys.
    """
    raw = _load_or_create_signing_key().public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )
    return hashlib.sha256(raw).hexdigest()[:16]


def private_key_b64() -> str:
    pk = _load_or_create_signing_key()
    return base64.b64encode(
        pk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    ).decode()


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sign_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    body = {
        **payload,
        "signed_at": datetime.now(UTC).isoformat(),
        # Embed a stable key identifier instead of the public key bytes. A
        # receipt that carries its own pubkey "verifies against itself" — a
        # forger swaps the pubkey for their own and the math passes. Verifiers
        # must look the pubkey up by key_id at /api/v1/keys against a pinned
        # root of trust.
        "key_id": public_key_id(),
    }
    msg = canonical_json(body)
    sig = _load_or_create_signing_key().sign(msg)
    return {
        "payload": body,
        "signature": base64.b64encode(sig).decode(),
        "alg": "Ed25519",
        "msg_sha256": hashlib.sha256(msg).hexdigest(),
        "key_id": body["key_id"],
    }


def verify_receipt(receipt: dict[str, Any], *, trusted_pubkey_b64: str) -> bool:
    """Verify a receipt against an EXTERNALLY supplied trusted public key.

    Callers must obtain the trusted key out-of-band (e.g. /api/v1/keys
    pinned at deploy time, or shipped with the verifier). NEVER trust a
    pubkey embedded in the receipt itself.
    """
    try:
        body = receipt["payload"]
        sig = base64.b64decode(receipt["signature"])
        pub_raw = base64.b64decode(trusted_pubkey_b64)
        pub = Ed25519PublicKey.from_public_bytes(pub_raw)
        pub.verify(sig, canonical_json(body))
        return True
    except Exception:
        return False
