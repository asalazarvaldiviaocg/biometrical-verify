from __future__ import annotations

import pytest

from app.core.security import public_key_b64, sign_receipt, verify_receipt
from app.services.crypto_vault import (
    UserKekRevoked,
    erase_user,
    open_blob,
    seal,
)


def test_seal_open_roundtrip(db_session):
    pt = b"sensitive-biometric-bytes" * 100
    aad = b"user-1|job-1|id"
    sealed = seal(pt, aad, user_id="user_1", db=db_session)
    assert sealed.ciphertext != pt
    out = open_blob(sealed, db_session)
    assert out == pt


def test_seal_tamper_detection(db_session):
    sealed = seal(b"hello", b"aad", user_id="user_2", db=db_session)
    sealed.ciphertext = sealed.ciphertext[:-1] + bytes([sealed.ciphertext[-1] ^ 1])
    from cryptography.exceptions import InvalidTag

    with pytest.raises(InvalidTag):
        open_blob(sealed, db_session)


def test_crypto_shred_user(db_session):
    """Erasing a user makes their existing blobs irrecoverable."""
    sealed = seal(b"payload", b"aad", user_id="user_3", db=db_session)
    # Sanity check: opens before erase.
    assert open_blob(sealed, db_session) == b"payload"

    erased = erase_user("user_3", db_session)
    assert erased is True

    # After erasure the wrapped DEK is no longer recoverable.
    with pytest.raises(UserKekRevoked):
        open_blob(sealed, db_session)


def test_receipt_sign_and_verify():
    receipt = sign_receipt({"job_id": "j1", "decision": "APPROVE", "similarity": 0.81})
    assert verify_receipt(receipt, trusted_pubkey_b64=public_key_b64()) is True

    receipt["payload"]["decision"] = "REJECT"
    assert verify_receipt(receipt, trusted_pubkey_b64=public_key_b64()) is False


def test_receipt_rejects_wrong_pubkey():
    """A receipt cannot validate against an attacker-supplied pubkey."""
    import base64

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    receipt = sign_receipt({"job_id": "j2", "decision": "APPROVE"})
    attacker = Ed25519PrivateKey.generate()
    attacker_pub = base64.b64encode(
        attacker.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode()
    assert verify_receipt(receipt, trusted_pubkey_b64=attacker_pub) is False
