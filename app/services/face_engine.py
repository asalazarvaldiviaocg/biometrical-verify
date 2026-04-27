"""Face detection, embedding extraction, and 1:1 comparison.

Wraps DeepFace + ArcFace for production use. Loaded lazily so test runs and
worker boot stay light. Frame selection picks the sharpest frame from a clip
to feed the embedding model — handles motion blur from short selfie videos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

_settings = get_settings()


@dataclass
class MatchResult:
    verified: bool
    distance: float
    similarity: float
    threshold: float
    model: str
    # Embeddings are kept off the result by default to avoid carrying raw
    # biometric vectors through logs / receipts. Re-extract on demand if a
    # caller genuinely needs them (e.g. for an offline audit).

    def to_public(self) -> dict:
        return {
            "verified": self.verified,
            "distance": self.distance,
            "similarity": self.similarity,
            "threshold": self.threshold,
            "model": self.model,
        }


def _decode_image(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes")
    return img


def _largest_face(reps: list[dict]) -> dict:
    return max(reps, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])


def extract_embedding(img: np.ndarray) -> np.ndarray:
    """Detect the dominant face and return its L2-normalised embedding."""
    from deepface import DeepFace  # heavy import — keep lazy

    reps = DeepFace.represent(
        img_path=img,
        model_name=_settings.face_model_name,
        detector_backend=_settings.face_detector,
        enforce_detection=True,
        align=True,
        normalization=_settings.face_model_name,
    )
    if not reps:
        raise ValueError("No face detected")
    rep = _largest_face(reps) if len(reps) > 1 else reps[0]
    vec = np.asarray(rep["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Degenerate embedding (zero vector)")
    return vec / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def compare(id_image_bytes: bytes, selfie_image_bytes: bytes) -> MatchResult:
    id_img = _decode_image(id_image_bytes)
    self_img = _decode_image(selfie_image_bytes)

    emb_id = extract_embedding(id_img)
    emb_self = extract_embedding(self_img)

    dist = cosine_distance(emb_id, emb_self)
    sim = 1.0 - dist
    return MatchResult(
        verified=dist <= _settings.match_threshold,
        distance=dist,
        similarity=sim,
        threshold=_settings.match_threshold,
        model=_settings.face_model_name,
    )


def best_frame_from_video(video_path: str, max_frames: int = 60) -> bytes:
    """Sample up to `max_frames`, return JPEG bytes of the sharpest one."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video at {video_path}")
    # Hard cap on declared frame count to guard against pathological files
    # that report millions of frames and would tie up the worker walking
    # them. Anything beyond ~10× the sample size is suspicious.
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if declared > max_frames * 100:
        cap.release()
        raise RuntimeError(
            f"Video declares {declared} frames; refusing to process (cap={max_frames * 100})"
        )
    best_score = -1.0
    best_jpeg: bytes | None = None
    seen = 0
    try:
        while seen < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness > best_score:
                best_score = sharpness
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok2:
                    best_jpeg = buf.tobytes()
            seen += 1
    finally:
        cap.release()
    if best_jpeg is None:
        raise ValueError("No usable frame in video")
    log.info("best_frame_selected", frames_examined=seen, sharpness=best_score)
    return best_jpeg
