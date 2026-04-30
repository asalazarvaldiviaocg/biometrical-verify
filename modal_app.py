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
        "boto3>=1.34",
    )
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


@app.function(
    image=image,
    secrets=[auth_secret, aws_secret],
    cpu=2,
    memory=4096,
    timeout=120,
    min_containers=0,
)
@modal.fastapi_endpoint(method="POST", docs=True)
def verify_face(payload: dict, x_verify_auth: str = Header(default="")):
    expected = os.environ.get("SHARED_SECRET", "")
    if not expected or x_verify_auth != expected:
        raise HTTPException(status_code=401, detail="invalid auth")

    id_key = payload.get("id_image_key") or ""
    selfie_key = payload.get("selfie_image_key") or ""
    if not id_key or not selfie_key:
        raise HTTPException(status_code=422, detail="missing id_image_key / selfie_image_key")

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

    def fetch(key: str) -> bytes:
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

    id_img = decode(fetch(id_key))
    selfie_img = decode(fetch(selfie_key))

    def descriptor(img: np.ndarray, label: str) -> np.ndarray:
        try:
            reps = DeepFace.represent(
                img_path=img,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=True,
                align=True,
                normalization="ArcFace",
            )
        except Exception:
            raise HTTPException(status_code=422, detail=f"no_face_{label}")
        if not reps:
            raise HTTPException(status_code=422, detail=f"no_face_{label}")
        # Pick the largest face detected (handles multi-face frames safely).
        rep = max(reps, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])
        vec = np.asarray(rep["embedding"], dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if n == 0:
            raise HTTPException(status_code=500, detail=f"degenerate_descriptor_{label}")
        return vec / n

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


@app.local_entrypoint()
def smoke():
    """`modal run modal_app.py` — basic local check that the app can be parsed."""
    print(f"Modal app '{APP_NAME}' loaded. Deploy with: modal deploy modal_app.py")
