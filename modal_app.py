"""
biometrical-verify — Modal serverless deployment.

Standalone face-match + liveness service used by biometrical-contract for
identity verification during the signing flow. This is a SEPARATE Modal app
from the user's PREV.IA project (`previa-rppg`); they share a Modal workspace
but no code, secrets, or runtime resources.

Tech stack (all OSS):
- DeepFace + ArcFace (face descriptors, MIT license)
- ONNX runtime + tensorflow-cpu (inference)
- OpenCV (image decode)
- boto3 (S3 fetch)

Endpoint
--------
POST /verify-face
Headers:
  Content-Type: application/json
  X-Verify-Auth: <shared secret>
Body:
  {"id_image_key": "...", "selfie_image_key": "..."}
Response (200):
  {
    "similarity": 73,           # 0–100
    "distance": 0.27,           # raw cosine distance (ArcFace)
    "match_pass": true,         # similarity >= threshold
    "threshold": 35,
    "model": "ArcFace"
  }
Response (4xx/5xx) on auth or processing failure with `reason` in body.

Deploy
------
  modal deploy modal_app.py

Secrets required (already created by setup):
  - biometrical-verify-auth → SHARED_SECRET
  - biometrical-verify-aws  → AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                              AWS_DEFAULT_REGION, BUCKET
"""

from __future__ import annotations

import os

import modal
from fastapi import Header, HTTPException

APP_NAME = "biometrical-verify"

app = modal.App(APP_NAME)

# ── Container image ─────────────────────────────────────────────────────────
# Heavy ML deps; first build is ~10 minutes, then cached. Linux apt deps are
# OpenCV's runtime libraries.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        "fastapi[standard]>=0.110",
        "pydantic>=2.6",
        # Pin TF below 2.20 (no Linux wheels for newer); pin mediapipe below
        # 0.10.15 (removed mediapipe.solutions API). DeepFace>=0.0.93 wraps
        # ArcFace, RetinaFace, MTCNN, etc.
        "tensorflow-cpu>=2.15,<2.20",
        "tf-keras",
        "deepface>=0.0.93",
        "onnxruntime>=1.17",
        "opencv-python-headless>=4.9",
        "numpy>=1.26",
        "scikit-image>=0.22",
        "scipy>=1.11",
        "boto3>=1.34",
    )
    # Mount the OSS package so Modal functions can import the canonical
    # signature_engine implementation. Single source of truth for the
    # comparison algorithm — Modal and the OSS FastAPI app run identical
    # code instead of two parallel copies that drift over time.
    .add_local_dir("./app", remote_path="/root/app")
)

auth_secret = modal.Secret.from_name("biometrical-verify-auth")
aws_secret = modal.Secret.from_name("biometrical-verify-aws")


# ── Match threshold ─────────────────────────────────────────────────────────
# ArcFace cosine distance:
#   - same person, recent photo: 0.20–0.40 → similarity 60–80%
#   - same person, aged INE 10y: 0.40–0.65 → similarity 35–60%
#   - different people:           0.70–1.00 → similarity 0–30%
# 35% lets aged IDs pass while still rejecting unrelated faces.
MATCH_THRESHOLD = 35


# Hardening for the Modal HTTP endpoints: every S3 key the caller sends
# is validated against this regex before we hit S3, and HEAD'd against a
# size cap before we read the object body into RAM. Without these guards
# a caller holding the shared secret could (a) request arbitrary objects
# from the shared bucket (cross-tenant data exfil) and (b) point at a
# huge object to OOM-kill the worker (4 GiB container, 100 MB JPEG is
# enough). The regex matches the prefixes biometrical-contract actually
# writes — sessions/<contractId>/... and contracts/<companyId>/... — and
# rejects path traversal / absolute paths.
import re as _re
_S3_KEY_RE = _re.compile(
    r"^(sessions|contracts|kyc|signatures|evidence)/"
    r"[A-Za-z0-9._-]{1,128}/"
    r"[A-Za-z0-9._/-]{1,256}$"
)
_MAX_IMAGE_BYTES = 8 * 1024 * 1024   # 8 MB per ID / selfie still
_MAX_SIG_B64_LEN = (8 * 1024 * 1024 * 4) // 3  # ≈ 8 MB decoded


def _validate_key(label: str, key: str) -> None:
    if not _S3_KEY_RE.match(key):
        raise HTTPException(status_code=422, detail=f"invalid_{label}_key")
    if ".." in key:  # belt-and-suspenders against traversal
        raise HTTPException(status_code=422, detail=f"invalid_{label}_key")


def _check_size(s3_client, bucket: str, label: str, key: str, cap: int) -> None:
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"s3_head_failed_{label}: {exc}") from exc
    size = int(head.get("ContentLength") or 0)
    if size > cap:
        raise HTTPException(
            status_code=413,
            detail=f"{label}_too_large ({size} bytes; max {cap})",
        )


@app.function(
    image=image,
    secrets=[auth_secret, aws_secret],
    cpu=2,
    memory=4096,
    timeout=120,
    min_containers=0,
)
@modal.fastapi_endpoint(method="POST", docs=False)
def verify_face(payload: dict, x_verify_auth: str = Header(default="")):
    expected = os.environ.get("SHARED_SECRET", "")
    if not expected or x_verify_auth != expected:
        raise HTTPException(status_code=401, detail="invalid auth")

    id_key = payload.get("id_image_key") or ""
    selfie_key = payload.get("selfie_image_key") or ""
    if not id_key or not selfie_key:
        raise HTTPException(status_code=422, detail="missing id_image_key / selfie_image_key")

    # Hardening: validate the S3 key shape BEFORE we open a boto3 client,
    # so a malicious caller can't point us at an unrelated tenant's data
    # or at S3 internals (`.well-known`, `..` traversal, etc.).
    _validate_key("id_image", id_key)
    _validate_key("selfie_image", selfie_key)

    bucket = os.environ.get("BUCKET", "")
    if not bucket:
        raise HTTPException(status_code=500, detail="bucket not configured")

    # Lazy imports — the cold-start cost lives here, not at module import,
    # so the auth check above runs immediately on warm invocations.
    import boto3
    import cv2
    import numpy as np
    from deepface import DeepFace

    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-1"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    def fetch(key: str, label: str) -> bytes:
        # head_object first to enforce a size cap before pulling the body.
        # Without this a multi-hundred-MB object would OOM-kill the 4 GiB
        # worker — possible because callers control the S3 key.
        _check_size(s3, bucket, label, key, _MAX_IMAGE_BYTES)
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"s3_fetch_failed: {exc}") from exc

    def decode(b: bytes) -> np.ndarray:
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=422, detail="image_decode_failed")
        return img

    id_img = decode(fetch(id_key, "id_image"))
    selfie_img = decode(fetch(selfie_key, "selfie_image"))

    def descriptor(img: np.ndarray, label: str) -> np.ndarray:
        # Detector cascade: retinaface (strongest) → mtcnn (catches small or
        # rotated faces retinaface misses) → opencv (Haar cascade — fast and
        # very forgiving on lower-quality scans). If all three give up we
        # accept the verdict and surface no_face_<label> so the front end
        # can show a face-detection-specific message instead of a fake "0%
        # similarity" comparison.
        #
        # Why three backends: a real-world failure that was rejecting paying
        # users — passport bio-page photos are physically smaller and often
        # have security overlays/holograms that confuse RetinaFace's anchor
        # boxes. MTCNN works on the same passport scan. INE photos are
        # similarly tricky when laminated under glare. OpenCV's Haar cascade
        # is the universal cheapest fallback (1990s tech, still robust on
        # well-lit frontal faces). DeepFace ships all three — no extra deps.
        last_err: str | None = None
        for backend in ("retinaface", "mtcnn", "opencv"):
            try:
                reps = DeepFace.represent(
                    img_path=img,
                    model_name="ArcFace",
                    detector_backend=backend,
                    enforce_detection=True,
                    align=True,
                    normalization="ArcFace",
                )
            except Exception as e:
                last_err = f"{backend}: {e}"
                continue
            if not reps:
                last_err = f"{backend}: empty_reps"
                continue
            # Pick the largest face detected (handles multi-face frames safely).
            rep = max(reps, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])
            vec = np.asarray(rep["embedding"], dtype=np.float32)
            n = float(np.linalg.norm(vec))
            if n == 0:
                last_err = f"{backend}: degenerate_descriptor"
                continue
            return vec / n
        # All three detectors failed — genuinely no face here, or the image
        # is too damaged. Log the trail of failures so we can tune later.
        print(f"[verify_face] no_face_{label} after cascade: {last_err}")
        raise HTTPException(status_code=422, detail=f"no_face_{label}")

    id_vec = descriptor(id_img, "id")
    selfie_vec = descriptor(selfie_img, "selfie")

    distance = float(1.0 - float(np.dot(id_vec, selfie_vec)))
    similarity = max(0, min(100, round((1.0 - distance) * 100)))

    return {
        "similarity": similarity,
        "distance": distance,
        "match_pass": similarity >= MATCH_THRESHOLD,
        "threshold": MATCH_THRESHOLD,
        "model": "ArcFace",
    }


# ── Signature comparison ────────────────────────────────────────────────────
# Compares the canvas signature drawn by the signer against the printed
# signature on the back of their INE/IFE. HARD GATE on the contract side —
# below el threshold, biometrical-contract devuelve 422 'below_threshold' y
# bloquea el flujo después de 5 intentos. Calibrado con SSIM + Hu Moments
# (modelo ssim+humoments+otsu-v2).
#
# v1 ship-now permissive (37): legitimate signers with poor lighting / fast
# strokes / passport bio-page glare were getting flagged as no-match. Half
# of the previous 75 threshold reduces false-rejects while the other layers
# (face match, video consent, identity continuity, name OCR) absorb the
# residual fraud risk. Re-tune up once we have a labelled production corpus.
SIGNATURE_THRESHOLD = 37


@app.function(
    image=image,
    secrets=[auth_secret, aws_secret],
    cpu=2,
    memory=2048,
    timeout=60,
    min_containers=0,
)
@modal.fastapi_endpoint(method="POST", docs=False)
def verify_signature(payload: dict, x_verify_auth: str = Header(default="")):
    """Compare a canvas signature (base64 PNG) against the signature region
    of an INE/IFE back-side image stored in S3.

    Body:
      {
        "id_back_image_key": "sessions/<id>/id-back-...",
        "signature_b64": "data:image/png;base64,..."  | "<base64-only>"
      }

    Response (200):
      {
        "similarity": 64,                  # 0–100, higher = more similar
        "ssim": 0.42,                      # raw SSIM in [-1, 1]
        "match_pass": true,                # similarity >= threshold
        "threshold": 60,
        "id_signature_found": true,        # false if no signature blob detected
        "model": "ssim+otsu-v1"
      }
    """
    expected = os.environ.get("SHARED_SECRET", "")
    if not expected or x_verify_auth != expected:
        raise HTTPException(status_code=401, detail="invalid auth")

    id_back_key = payload.get("id_back_image_key") or ""
    sig_b64 = payload.get("signature_b64") or ""
    # `id_type` selects the signature-region cropper. Defaults to INE so
    # legacy callers that only ever sent INE keep working without changes.
    id_type = (payload.get("id_type") or "INE").upper()
    if id_type not in ("INE", "PASAPORTE"):
        raise HTTPException(status_code=422, detail="invalid id_type (must be INE or PASAPORTE)")
    if not id_back_key or not sig_b64:
        raise HTTPException(status_code=422, detail="missing id_back_image_key / signature_b64")

    # Same hardening as verify_face: validate the S3 key shape and cap
    # the base64 body length before we open the S3 client. Without these
    # guards a caller can request arbitrary cross-tenant S3 objects or
    # OOM the worker with a huge base64 payload.
    _validate_key("id_back_image", id_back_key)
    if len(sig_b64) > _MAX_SIG_B64_LEN:
        raise HTTPException(
            status_code=413,
            detail=f"signature_b64 too large ({len(sig_b64)} chars; max {_MAX_SIG_B64_LEN})",
        )

    bucket = os.environ.get("BUCKET", "")
    if not bucket:
        raise HTTPException(status_code=500, detail="bucket not configured")

    import boto3

    # Single source of truth: the canonical OSS signature engine. Modal and
    # the standalone FastAPI app at app/main.py both call the same function,
    # so the comparison algorithm cannot drift between deploy targets.
    from app.services.signature_engine import compare_signatures_b64

    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-1"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    # head_object size cap — same rationale as verify_face's fetch().
    _check_size(s3, bucket, "id_back_image", id_back_key, _MAX_IMAGE_BYTES)

    try:
        obj = s3.get_object(Bucket=bucket, Key=id_back_key)
        id_back_bytes = obj["Body"].read()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"s3_fetch_failed: {exc}") from exc

    try:
        result = compare_signatures_b64(
            id_back_bytes, sig_b64,
            threshold=SIGNATURE_THRESHOLD,
            id_type=id_type,
        )
    except ValueError as exc:
        # Engine raises ValueError on decode failures; surface as 422 so the
        # Node client can map it to a recognisable reason.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result.to_dict()


@app.local_entrypoint()
def smoke():
    """`modal run modal_app.py` — basic local check that the app can be parsed."""
    print(f"Modal app '{APP_NAME}' loaded. Deploy with: modal deploy modal_app.py")
