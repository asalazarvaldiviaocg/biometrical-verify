"""Public-key directory for receipt verification.

Verifiers MUST pin the active key out-of-band (deploy-time fingerprint, ops
runbook, etc.) and use this endpoint to look up the matching pubkey bytes.
Trusting a pubkey embedded inside the receipt itself is meaningless — a
forger swaps the pubkey for their own and the math passes.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.security import public_key_b64, public_key_id

router = APIRouter(prefix="/api/v1/keys", tags=["keys"])


@router.get("")
def list_keys() -> dict:
    return {
        "active": {
            "key_id": public_key_id(),
            "alg": "Ed25519",
            "public_key_b64": public_key_b64(),
        },
        # Old key-IDs would be listed here during a rotation window so old
        # receipts remain verifiable. With a single static signing key today
        # the list contains only the active entry.
        "previous": [],
    }
