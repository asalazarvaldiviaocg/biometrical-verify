#!/usr/bin/env python3
"""One-shot bootstrap: download anti-spoof model + generate dev keys.

Run via `make seed` or `python scripts/bootstrap.py`.
Idempotent — safe to re-run.
"""

from __future__ import annotations

import base64
import os
import secrets
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"

# Multiple mirrors — first one that responds wins. The system gracefully falls
# back to a sharpness-based heuristic if no model is present (dev only — replace
# in production). Drop a model in `models/anti_spoof_mn3.onnx` to enable the CNN.
ANTI_SPOOF_MIRRORS = [
    "https://huggingface.co/datasets/biometrical-org/anti-spoof-mn3/resolve/main/anti_spoof_mn3.onnx",
    "https://github.com/hpc203/face-anti-spoofing-using-onnxruntime/raw/main/4_0_0_80x80_MiniFASNetV1SE.onnx",
]
ANTI_SPOOF_DEST = MODELS_DIR / "anti_spoof_mn3.onnx"


def ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / ".gitkeep").touch(exist_ok=True)


def download_anti_spoof() -> None:
    if ANTI_SPOOF_DEST.exists() and ANTI_SPOOF_DEST.stat().st_size > 0:
        print(f"[ok] anti-spoof model already present: {ANTI_SPOOF_DEST}")
        return
    for url in ANTI_SPOOF_MIRRORS:
        print(f"[..] trying {url}")
        try:
            urllib.request.urlretrieve(url, ANTI_SPOOF_DEST)
            if ANTI_SPOOF_DEST.stat().st_size > 1024:
                print(f"[ok] saved to {ANTI_SPOOF_DEST}")
                return
            ANTI_SPOOF_DEST.unlink(missing_ok=True)
        except Exception as e:
            print(f"     skip ({type(e).__name__}: {str(e)[:80]})")
    print("[!!] no mirror reachable — liveness will use sharpness-heuristic fallback.")
    print(f"     For production, drop your own anti-spoof ONNX at: {ANTI_SPOOF_DEST}")
    print("     Recommended: Silent-Face-Anti-Spoofing MiniFASNet exported to ONNX.")


def ensure_env() -> None:
    if ENV_FILE.exists():
        print(f"[ok] {ENV_FILE.name} already exists")
        return
    if not ENV_EXAMPLE.exists():
        print(f"[!!] {ENV_EXAMPLE.name} not found")
        return
    text = ENV_EXAMPLE.read_text()
    # Inject fresh dev secrets
    master_key = base64.b64encode(secrets.token_bytes(32)).decode()
    receipt_key = base64.b64encode(secrets.token_bytes(32)).decode()
    jwt_secret = secrets.token_urlsafe(48)
    text = text.replace(
        "MASTER_KEY=ZGV2LW9ubHktbWFzdGVyLWtleS1jaGFuZ2UtbWUtMzItYnl0ZXM=",
        f"MASTER_KEY={master_key}",
    )
    text = text.replace("RECEIPT_SIGNING_KEY=", f"RECEIPT_SIGNING_KEY={receipt_key}")
    text = text.replace("JWT_SECRET=dev-jwt-secret-change-me", f"JWT_SECRET={jwt_secret}")
    ENV_FILE.write_text(text)
    print(f"[ok] wrote {ENV_FILE.name} with fresh dev secrets")


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
