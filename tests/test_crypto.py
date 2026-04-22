from __future__ import annotations

from app.core.security import sign_receipt, verify_receipt
from app.services.crypto_vault import open_blob, seal


def test_seal_open_roundtrip():
    pt = b"sensitive-biometric-bytes" * 100
    aad = b"user-1|job-1|id"
    sealed = seal(pt, aad)
    assert sealed.ciphertext != pt
    out = open_blob(sealed)
    assert out == pt


def test_seal_tamper_detection():
    sealed = seal(b"hello", b"aad")
    sealed.ciphertext = sealed.ciphertext[:-1] + bytes([sealed.ciphertext[-1] ^ 1])
    import pytest
    from cryptography.exceptions import InvalidTag

    with pytest.raises(InvalidTag):
        open_blob(sealed)


def test_receipt_sign_and_verify():
    receipt = sign_receipt({"job_id": "j1", "decision": "APPROVE", "similarity": 0.81})
    assert verify_receipt(receipt) is True

    receipt["payload"]["decision"] = "REJECT"
    assert verify_receipt(receipt) is False
