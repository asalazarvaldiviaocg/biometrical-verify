#!/usr/bin/env python3
"""One-shot bootstrap: download anti-spoof model + generate dev keys.

Run via `make seed` or `python scripts/bootstrap.py`.
Idempotent — safe to re-run.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"

# Pinned mirror + SHA-256. Boot will refuse to use any model whose hash does
# not match this value. Any new mirror must be vetted and added with its real
# hash; we never accept whatever a redirect happens to deliver.
#
# These values reflect the v1 4_0_0_80x80_MiniFASNetV1SE.onnx artifact. Update
# both URL and hash when rotating to a new model.
ANTI_SPOOF_URL = (
    "https://github.com/hpc203/face-anti-spoofing-using-onnxruntime/raw/main/"
    "4_0_0_80x80_MiniFASNetV1SE.onnx"
)
ANTI_SPOOF_SHA256 = "1bda35b3c7adfdedc01dde064cdc3094e5b1e7c2dd7a2cc6810fa54e7894220a"
ANTI_SPOOF_DEST = MODELS_DIR / "anti_spoof_mn3.onnx"


def ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / ".gitkeep").touch(exist_ok=True)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_anti_spoof() -> None:
    if ANTI_SPOOF_DEST.exists() and ANTI_SPOOF_DEST.stat().st_size > 0:
        actual = _sha256_of(ANTI_SPOOF_DEST)
        if actual == ANTI_SPOOF_SHA256:
            print(f"[ok] anti-spoof model already present and verified: {ANTI_SPOOF_DEST}")
            return
        print(f"[!!] existing model hash mismatch ({actual[:16]}...). Re-downloading.")
        ANTI_SPOOF_DEST.unlink(missing_ok=True)

    if not ANTI_SPOOF_URL.startswith("https://"):
        print(f"[!!] refusing to download from non-https URL: {ANTI_SPOOF_URL}")
        return
    print(f"[..] downloading {ANTI_SPOOF_URL}")
    try:
        # Pinned https URL only; ANTI_SPOOF_SHA256 hash-checked below.
        urllib.request.urlretrieve(ANTI_SPOOF_URL, ANTI_SPOOF_DEST)  # noqa: S310
    except Exception as e:
        ANTI_SPOOF_DEST.unlink(missing_ok=True)
        print(f"[!!] download failed: {type(e).__name__}: {e}")
        print(
            "     The pipeline will refuse to run without a verified model in production. "
            "Drop a vetted model at the destination manually if your network blocks the host."
        )
        return

    actual = _sha256_of(ANTI_SPOOF_DEST)
    if actual != ANTI_SPOOF_SHA256:
        ANTI_SPOOF_DEST.unlink(missing_ok=True)
        print(
            f"[!!] downloaded file hash {actual[:16]}... does not match expected "
            f"{ANTI_SPOOF_SHA256[:16]}... — refusing to install."
        )
        return

    print(f"[ok] saved + verified: {ANTI_SPOOF_DEST}")


def ensure_env() -> None:
    if ENV_FILE.exists():
        print(f"[ok] {ENV_FILE.name} already exists — leaving as is")
        return
    if not ENV_EXAMPLE.exists():
        print(f"[!!] {ENV_EXAMPLE.name} not found")
        return
    text = ENV_EXAMPLE.read_text()
    # Inject fresh dev secrets.
    master_key = base64.b64encode(secrets.token_bytes(32)).decode()
    receipt_key = base64.b64encode(secrets.token_bytes(32)).decode()
    jwt_secret = secrets.token_urlsafe(48)
    text = text.replace("MASTER_KEY=", f"MASTER_KEY={master_key}", 1)
    text = text.replace(
        "RECEIPT_SIGNING_KEY=", f"RECEIPT_SIGNING_KEY={receipt_key}", 1
    )
    text = text.replace("JWT_SECRET=", f"JWT_SECRET={jwt_secret}", 1)
    text = text.replace(
        "ANTI_SPOOF_MODEL_SHA256=",
        f"ANTI_SPOOF_MODEL_SHA256={ANTI_SPOOF_SHA256}",
        1,
    )
    ENV_FILE.write_text(text)
    os.chmod(ENV_FILE, 0o600)
    print(f"[ok] wrote {ENV_FILE.name} with fresh dev secrets (chmod 600)")


def main() -> int:
    print("=== biometrical-verify bootstrap ===")
    ensure_models_dir()
    ensure_env()
    download_anti_spoof()
    print("=== done ===")
    print("Next: `make up` to start the stack, `make frontend` to start the demo UI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
