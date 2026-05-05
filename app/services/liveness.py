"""Liveness detection: passive (Biometrical Liveness Engine) + active
(challenge response).

Passive analysis runs the in-house Biometrical Liveness Engine v1 (BLE) — a
multi-signal ensemble of HSV variance, LBP entropy, FFT spectral ratio,
YCrCb skin density and Sobel gradient variance. 100% proprietary; no
external pre-trained model weights. See ``biometrical_liveness_engine.py``
for the algorithm and weights.

Active verifies the user performed the requested challenge (e.g. blinked
N times) by tracking eye-aspect-ratio across frames via MediaPipe FaceMesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.biometrical_liveness_engine import analyze as ble_analyze

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


# ---------- passive (Biometrical Liveness Engine) ----------
# The previous implementation depended on a Silent-Face-Anti-Spoofing
# MiniFASNet ONNX model whose source URL went dead. We replaced it with the
# in-house BLE — see app/services/biometrical_liveness_engine.py for the
# algorithm. This wrapper preserves the previous public signature so the
# orchestrator below didn't need to change.


def passive_liveness_score(face_bgr: np.ndarray) -> float:
    """Run BLE on a single face crop and return the fused score in [0, 1]."""
    return ble_analyze(face_bgr).score


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
