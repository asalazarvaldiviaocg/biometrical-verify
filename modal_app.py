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

import hmac
import os
import re

import modal
from fastapi import Header, HTTPException

APP_NAME = "biometrical-verify-v3"

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


# Hardening for the Modal HTTP endpoints. Two lines of defense:
#
#   1. Shape validation — reject path traversal, absolute paths, weird
#      control characters. The KEY shape is intentionally permissive
#      (any `[A-Za-z0-9._/-]` of reasonable length) because real
#      production keys come from many code paths in biometrical-contract:
#        - signing/id/<contractId>-<timestamp>     (presigned PUT)
#        - signing/selfie/<contractId>-<timestamp> (presigned PUT)
#        - sessions/<contractId>/selfie-<token>.jpg
#        - sessions/<contractId>/id-<token>.jpg
#        - contracts/<companyId>/...
#        - kyc/<contractId>/...
#      Pinning a closed prefix list bricked the verify route (422 in
#      ~80 ms) because it didn't include `signing/` — the most common
#      prefix. Cross-tenant exfil concern is addressed structurally by
#      the cuid-generated contractIds (~1e21 keyspace, unguessable) and
#      by the shared-secret gate, NOT by the prefix list.
#
#   2. Size cap (HEAD before GET, see _check_size) — protects the
#      4 GiB worker from OOM via a multi-hundred-MB malicious upload.
#      This is the load-bearing protection against the "callers control
#      the S3 key" threat. Permissive shape + size cap > narrow shape
#      + no size cap, because the size cap is what actually matters.
# `\Z` (not `$`) so a trailing newline can't slip a control char past the
# class — `$` matches before a final `\n`, `\Z` anchors the true end.
_S3_KEY_RE = re.compile(r"^[A-Za-z0-9._/-]{1,512}\Z")
_MAX_IMAGE_BYTES = 8 * 1024 * 1024   # 8 MB per ID / selfie still
_MAX_SIG_B64_LEN = (8 * 1024 * 1024 * 4) // 3  # ≈ 8 MB decoded
_MAX_VIDEO_BYTES = 30 * 1024 * 1024  # 30 MB per liveness video (6s @ ~1080p)
# Decompression-bomb guard: cap the DECODED pixel count. The byte caps above
# bound the compressed payload, but a highly-compressible PNG (uniform color)
# can expand to a multi-gigapixel array inside cv2.imdecode and OOM the worker.
# A real ID/selfie is well under this; 50 MP ≈ an 8000×6250 image.
_MAX_DECODED_PIXELS = 50 * 1024 * 1024


def _validate_key(label: str, key: str) -> None:
    if not _S3_KEY_RE.match(key):
        raise HTTPException(status_code=422, detail=f"invalid_{label}_key")
    # Reject relative-path traversal even when each path segment passes
    # the character class (a string like "a/../b" would otherwise slip
    # through). Also reject `//` collapses, leading `/`, leading `.`
    # (hidden files), and trailing `/` (directory refs).
    if (
        ".." in key or "//" in key
        or key.startswith("/") or key.startswith(".") or key.endswith("/")
    ):
        raise HTTPException(status_code=422, detail=f"invalid_{label}_key")


def _auth_ok(x_verify_auth: str) -> bool:
    """Constant-time shared-secret check. `hmac.compare_digest` avoids the
    byte-by-byte short-circuit of `==`, which is timing-observable on these
    public endpoints."""
    expected = os.environ.get("SHARED_SECRET", "")
    if not expected:
        return False
    return hmac.compare_digest(x_verify_auth, expected)


def _fetch_capped(s3_client, bucket: str, label: str, key: str, cap: int) -> bytes:
    """GET the object and enforce the size cap from the SAME response's
    ContentLength before reading the body — one round trip, no TOCTOU gap
    between a separate HEAD and the GET, and no cap-bypass when a backend
    omits ContentLength on HEAD. Raw botocore errors are logged, not returned,
    so bucket/key/existence metadata never leaks in the HTTP body."""
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        print(f"[s3_fetch_failed] {label} {key}: {exc}")
        raise HTTPException(status_code=502, detail=f"s3_fetch_failed_{label}") from exc
    declared = obj.get("ContentLength")
    if declared is not None and int(declared) > cap:
        raise HTTPException(
            status_code=413,
            detail=f"{label}_too_large ({int(declared)} bytes; max {cap})",
        )
    # Read at most cap+1 bytes so a missing/lying ContentLength still can't
    # stream an unbounded body into the worker.
    body = obj["Body"].read(cap + 1)
    if len(body) > cap:
        raise HTTPException(status_code=413, detail=f"{label}_too_large (>{cap} bytes)")
    return body


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
    if not _auth_ok(x_verify_auth):
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

    def decode(b: bytes) -> np.ndarray:
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=422, detail="image_decode_failed")
        if img.shape[0] * img.shape[1] > _MAX_DECODED_PIXELS:
            raise HTTPException(status_code=413, detail="image_too_large_decoded")
        return img

    id_img = decode(_fetch_capped(s3, bucket, "id_image", id_key, _MAX_IMAGE_BYTES))
    selfie_img = decode(_fetch_capped(s3, bucket, "selfie_image", selfie_key, _MAX_IMAGE_BYTES))

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
# bloquea el flujo después de 5 intentos. Motor multi-feature v5 (SSIM + Hu
# Moments + HOG + stroke density + aspect ratio + Hausdorff; ver
# signature_engine.MODEL_NAME).
#
# v1 ship-now permissive (37): legitimate signers with poor lighting / fast
# strokes / passport bio-page glare were getting flagged as no-match. Half
# of the previous 75 threshold reduces false-rejects while the other layers
# (face match, video consent, identity continuity, name OCR) absorb the
# residual fraud risk.
#
# v3 (40): real production calibration. v2's bump to 50 was based on a
# theoretical distribution; live firmas legítimas (finger-on-canvas vs
# printed-on-passport) actually peak around 45-48 because the texture
# gap between the two capture media physically caps SSIM and HOG. With
# the threshold at 50, a real signer trying to clear the gate against
# their own passport would fail every time. We rely on the hard floors
# in signature_engine.py (SSIM_HARD_FLOOR / HOG_HARD_FLOOR) to block
# doodles regardless of the additive total — that's where fraud
# protection lives now, not the threshold. The threshold's only job is
# to set the *minimum total quality* a signer's strokes must reach;
# 40 is the empirical floor for real captures of real signatures.
SIGNATURE_THRESHOLD = 40


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
        "match_pass": true,                # similarity >= threshold AND floors
        "threshold": 40,
        "id_signature_found": true,        # false if no signature blob detected
        "model": "ssim+hu+hog+stroke+aspect+hausdorff-v5"
      }
    """
    if not _auth_ok(x_verify_auth):
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

    # Single GET with an inline size cap — same rationale as verify_face.
    id_back_bytes = _fetch_capped(s3, bucket, "id_back_image", id_back_key, _MAX_IMAGE_BYTES)

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


# ── Server-side liveness / anti-spoofing ────────────────────────────────────
# Tier-3 hardening: the client-side selfie capture (face-api.js + EAR blink
# detection) is the first line of defense, but a determined attacker can spoof
# the JS layer (e.g. by replaying a recorded video to a virtual camera). This
# endpoint runs the canonical OSS biometrical_liveness_engine on a sample of
# decoded video frames, server-side, where the client cannot tamper with the
# inputs. The 5 signals (HSV variance, LBP entropy, FFT mid-band ratio, skin
# density, gradient variance) target the four common spoof modalities:
#
#   - Printed photo:  low LBP entropy + skewed FFT toward over-smooth
#   - Phone screen:   moiré pattern in FFT high-band + reduced gradient variance
#   - 3D silicone mask: skin density mismatch + HSV uniformity
#   - Replay attack:  texture flatness + frame-to-frame entropy collapse
#
# Verdict logic combines a per-frame is_live flag with an aggregate floor —
# both the mean score AND ≥60 % of analyzed frames must clear the bar. This
# kills the case where the attacker happens to nail one frame but the rest
# look like screen replay.

# Analyze 6 frames evenly spaced across the video. More = slower (each
# DeepFace.extract_faces call is ~150-300 ms cold). 6 is the empirical
# sweet spot for detecting frame-level inconsistencies in deepfakes
# without running into Modal's 120 s function timeout for very long
# uploads.
_LIVENESS_FRAME_SAMPLES = 6
# Per-frame floor for the BLE anti-spoof score. NOTE: this is a deliberate,
# stricter-than-engine-default value. The engine's own LIVENESS_PASS_THRESHOLD
# is 0.35 ("ship-now permissive") with 0.55-0.65 documented as the
# post-calibration target; we apply the stricter 0.55 here as a hard gate.
# Trade-off: stronger anti-spoof, but real users whose fused BLE score lands
# in [0.35, 0.55) are rejected. Re-tune (up or down) against a labelled
# corpus of real selfies + attacks before treating this number as final.
_LIVENESS_PASS_THRESHOLD = 0.55
# Aggregate floor — 60 % of analyzed frames must pass. Looser than per-frame
# alone (a single bad frame from a real signer doesn't fail the recording)
# but tight enough that a still-photo replay can't pass (texture is constant
# so all frames pass-or-fail together).
_LIVENESS_PASS_RATIO_MIN = 0.60
# Minimum number of face-bearing frames we must actually analyze before the
# pass_ratio is meaningful. Without this, a video where only 1 of 6 sampled
# frames has a detectable face collapses to pass_ratio over n=1 — a single
# lucky printed-photo flash frame scores is_live=True. Fewer than this ⇒ we
# can't render an anti-spoof verdict, so we surface no_face_in_video (which
# the contract treats as a non-fatal skip, same as a Modal outage).
_LIVENESS_MIN_ANALYZED = 3
# Minimum real frame count for a usable liveness clip (~0.4 s at 30 fps).
_LIVENESS_MIN_TOTAL_FRAMES = 12
# Hard cap on the linear-walk counting pass so a pathological/huge upload
# can't spin the CPU unbounded (30 MB video cap already bounds this loosely).
_LIVENESS_MAX_WALK_FRAMES = 3000


@app.function(
    image=image,
    secrets=[auth_secret, aws_secret],
    cpu=2,
    memory=4096,
    timeout=120,
    min_containers=0,
)
@modal.fastapi_endpoint(method="POST", docs=False)
def verify_liveness(payload: dict, x_verify_auth: str = Header(default="")):
    """Server-side liveness verdict on a recorded selfie video.

    Body:
      {"liveness_video_key": "signing/liveness/<contractId>-<ts>"}

    Response (200):
      {
        "is_live": true,
        "score": 0.7421,
        "pass_ratio": 0.83,
        "frames_analyzed": 6,
        "frames_passed": 5,
        "threshold": 0.55,
        "per_frame": [...]   # forensic detail for audit logs
      }

    Response (4xx) on auth failure / decode failure / no face found.
    """
    if not _auth_ok(x_verify_auth):
        raise HTTPException(status_code=401, detail="invalid auth")

    video_key = payload.get("liveness_video_key") or ""
    if not video_key:
        raise HTTPException(status_code=422, detail="missing liveness_video_key")
    _validate_key("liveness_video", video_key)

    bucket = os.environ.get("BUCKET", "")
    if not bucket:
        raise HTTPException(status_code=500, detail="bucket not configured")

    import tempfile

    import boto3
    import cv2
    from deepface import DeepFace
    from app.services.biometrical_liveness_engine import analyze

    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-1"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    video_bytes = _fetch_capped(s3, bucket, "liveness_video", video_key, _MAX_VIDEO_BYTES)

    # cv2.VideoCapture needs a real filesystem path; raw bytes don't work.
    # Suffix only helps ffmpeg when magic bytes are inconclusive; WebM (EBML)
    # and MP4 (ftyp box) both probe conclusively by content, so a stable
    # default with an mp4 override is enough.
    suffix = ".mp4" if (video_key.endswith(".mp4") or video_bytes[4:8] == b"ftyp") else ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=422, detail="video_decode_failed")

        # Frame count: MediaRecorder WebM routinely reports 0/-1 for
        # CAP_PROP_FRAME_COUNT because live-recorded Matroska carries no
        # duration/cues, and CAP_PROP_POS_FRAMES seeks are unreliable on VP8/VP9
        # even when a count IS reported. So we NEVER trust the metadata count or
        # index seeks: we count real frames with a cheap grab()-only pass (no
        # decode), then linear-walk a second time, decoding only at the sample
        # positions. This is the "walk the stream linearly" behavior the old
        # code promised in a comment but never implemented (it just 422'd every
        # cues-less webm as video_too_short, silently disabling the gate).
        real_total = 0
        while real_total <= _LIVENESS_MAX_WALK_FRAMES and cap.grab():
            real_total += 1
        if real_total < _LIVENESS_MIN_TOTAL_FRAMES:
            cap.release()
            raise HTTPException(status_code=422, detail="video_too_short")

        # Reopen — the counting pass consumed the stream.
        cap.release()
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=422, detail="video_decode_failed")

        sample_positions = {
            int(real_total * (i + 0.5) / _LIVENESS_FRAME_SAMPLES)
            for i in range(_LIVENESS_FRAME_SAMPLES)
        }

        per_frame: list[dict] = []
        pos = -1
        while cap.grab():
            pos += 1
            if pos not in sample_positions:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            idx = pos

            # Detect face — opencv backend is the cheapest detector and
            # works fine on selfie frames (frontal face, well lit by the
            # alignment gate). RetinaFace is overkill here.
            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="opencv",
                    enforce_detection=True,
                    align=False,
                )
            except Exception:
                continue
            if not faces:
                continue

            largest = max(
                faces,
                key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"],
            )
            fa = largest["facial_area"]
            x, y, w, h = int(fa["x"]), int(fa["y"]), int(fa["w"]), int(fa["h"])
            # Clamp to frame bounds defensively — DeepFace can occasionally
            # return slight overshoot on edge faces.
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            face_bgr = frame[y:y2, x:x2]
            if face_bgr.size == 0:
                continue

            try:
                analysis = analyze(face_bgr, threshold=_LIVENESS_PASS_THRESHOLD)
                per_frame.append({
                    "idx": idx,
                    **analysis.to_public(),
                })
            except Exception as exc:
                print(f"[verify_liveness] frame {idx} analyze failed: {exc}")
                continue

        cap.release()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # A verdict needs enough face-bearing frames to be meaningful. With too
    # few, pass_ratio is dominated by one or two frames and a single lucky
    # spoof frame passes — so we decline to render a verdict and let the
    # contract soft-skip (same non-fatal path as an outage) rather than
    # emit an attackable is_live=True.
    if len(per_frame) < _LIVENESS_MIN_ANALYZED:
        raise HTTPException(status_code=422, detail="no_face_in_video")

    avg_score = sum(p["score"] for p in per_frame) / len(per_frame)
    passed = sum(1 for p in per_frame if p["is_live"])
    pass_ratio = passed / len(per_frame)
    is_live = avg_score >= _LIVENESS_PASS_THRESHOLD and pass_ratio >= _LIVENESS_PASS_RATIO_MIN

    return {
        "is_live": is_live,
        "score": round(float(avg_score), 4),
        "pass_ratio": round(float(pass_ratio), 4),
        "frames_analyzed": len(per_frame),
        "frames_passed": passed,
        "threshold": _LIVENESS_PASS_THRESHOLD,
        "per_frame": per_frame,
    }


@app.local_entrypoint()
def smoke():
    """`modal run modal_app.py` — basic local check that the app can be parsed."""
    print(f"Modal app '{APP_NAME}' loaded. Deploy with: modal deploy modal_app.py")
