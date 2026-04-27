"""Liveness detection: passive (anti-spoof CNN) + active (challenge response).

Passive uses a Silent-Face-Anti-Spoofing MiniFASNet ONNX model on a single
frame. Active verifies that the user performed the requested challenge, e.g.
blinked at least N times, by tracking eye-aspect-ratio across frames via
MediaPipe FaceMesh.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
_settings = get_settings()

# Eye landmarks (MediaPipe FaceMesh, refined)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Cap how many frames we'll iterate during liveness analysis even on a video
# whose declared length is much larger. The submit endpoint already caps the
# byte size; this is the second line of defence against a worker DoS via a
# pathological media file with a corrupt/inflated frame count.
LIVENESS_MAX_FRAMES = 240


Challenge = Literal["blink_twice", "blink_once", "turn_head", "none"]


@dataclass
class LivenessResult:
    is_live: bool
    score: float
    challenge: Challenge
    challenge_passed: bool | None
    notes: str

    def to_public(self) -> dict:
        return {
            "is_live": self.is_live,
            "score": self.score,
            "challenge": self.challenge,
            "challenge_passed": self.challenge_passed,
            "notes": self.notes,
        }


# ---------- passive (ONNX) ----------

_session = None
_input_name: str | None = None
_session_lock = threading.Lock()


def _get_session():
    global _session
    if _session is not None:
        return _session
    # Two threads racing into first-call would each load the model into RAM.
    # Cheap to lock here; the slow path runs at most once per process.
    with _session_lock:
        if _session is not None:
            return _session
        return _load_session_locked()


def _load_session_locked():
    global _session, _input_name
    if not os.path.exists(_settings.anti_spoof_model_path):
        # In production we refuse to fall back to the sharpness heuristic
        # silently — that fallback approves printed photos and would silently
        # downgrade the entire pipeline if a model download failed.
        if _settings.is_prod:
            raise RuntimeError(
                f"Anti-spoof ONNX model missing at {_settings.anti_spoof_model_path}; "
                "refusing to fall back to sharpness heuristic in production"
            )
        log.warning("anti_spoof_model_missing", path=_settings.anti_spoof_model_path)
        return None

    # If a hash is configured, verify the file contents match exactly. This
    # prevents both bit-rot and a swapped-out model.
    if _settings.anti_spoof_model_sha256:
        import hashlib

        sha = hashlib.sha256()
        with open(_settings.anti_spoof_model_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha.update(chunk)
        actual = sha.hexdigest()
        if actual.lower() != _settings.anti_spoof_model_sha256.lower():
            raise RuntimeError(
                f"Anti-spoof model hash mismatch: expected "
                f"{_settings.anti_spoof_model_sha256}, got {actual}"
            )
    elif _settings.is_prod:
        raise RuntimeError(
            "ANTI_SPOOF_MODEL_SHA256 must be configured in production "
            "so the loaded model is verifiable"
        )

    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _session = ort.InferenceSession(_settings.anti_spoof_model_path, providers=providers)
    _input_name = _session.get_inputs()[0].name
    return _session


def _preprocess(face_bgr: np.ndarray, size: int = 80) -> np.ndarray:
    img = cv2.resize(face_bgr, (size, size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img


def passive_liveness_score(face_bgr: np.ndarray) -> float:
    sess = _get_session()
    if sess is None:
        # Fallback heuristic: texture sharpness — better than nothing in dev,
        # NEVER acceptable in prod (the warning above flags it).
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(1.0, sharp / 250.0))
    x = _preprocess(face_bgr)
    out = sess.run(None, {_input_name: x})[0]
    e = np.exp(out - out.max())
    probs = e / e.sum(axis=1, keepdims=True)
    return float(probs[0, 1])  # class 1 = live


# ---------- active (MediaPipe blink count) ----------

def _ear(landmarks, idxs, w: int, h: int) -> float:
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs])
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return (a + b) / (2.0 * c + 1e-6)


def detect_blinks(
    video_path: str,
    ear_thr: float = 0.21,
    max_frames: int = LIVENESS_MAX_FRAMES,
) -> int:
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video at {video_path}")
    blinks = 0
    closed = False
    seen = 0
    with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as fm:
        try:
            while seen < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                seen += 1
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)
                if not res.multi_face_landmarks:
                    continue
                lm = res.multi_face_landmarks[0].landmark
                ear = (_ear(lm, LEFT_EYE, w, h) + _ear(lm, RIGHT_EYE, w, h)) / 2.0
                if ear < ear_thr and not closed:
                    closed = True
                elif ear >= ear_thr and closed:
                    blinks += 1
                    closed = False
        finally:
            cap.release()
    return blinks


def detect_head_turn(
    video_path: str,
    min_yaw_deg: float = 18.0,
    max_frames: int = LIVENESS_MAX_FRAMES,
) -> bool:
    """Cheap nose-position heuristic: detects sideways head movement."""
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video at {video_path}")
    nose_xs: list[float] = []
    seen = 0
    with mp.solutions.face_mesh.FaceMesh(refine_landmarks=False, max_num_faces=1) as fm:
        try:
            while seen < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                seen += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)
                if not res.multi_face_landmarks:
                    continue
                # Landmark 1 = nose tip (normalised x in 0..1)
                nose_xs.append(res.multi_face_landmarks[0].landmark[1].x)
        finally:
            cap.release()
    if len(nose_xs) < 5:
        return False
    spread = (max(nose_xs) - min(nose_xs)) * 100  # rough proxy for degrees
    return spread >= min_yaw_deg


# ---------- orchestrator ----------

def run_liveness(
    video_path: str,
    best_frame_bgr: np.ndarray,
    challenge: Challenge = "blink_twice",
) -> LivenessResult:
    p_score = passive_liveness_score(best_frame_bgr)

    challenge_passed: bool | None = None
    if challenge == "blink_twice":
        challenge_passed = detect_blinks(video_path) >= 2
    elif challenge == "blink_once":
        challenge_passed = detect_blinks(video_path) >= 1
    elif challenge == "turn_head":
        challenge_passed = detect_head_turn(video_path)
    elif challenge == "none":
        challenge_passed = None

    is_live = p_score >= _settings.liveness_min_score and (
        challenge_passed is None or challenge_passed
    )
    return LivenessResult(
        is_live=is_live,
        score=p_score,
        challenge=challenge,
        challenge_passed=challenge_passed,
        notes=f"passive={p_score:.3f} challenge={challenge}={challenge_passed}",
    )
