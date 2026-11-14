"""ID document quality + structure validation.

Quality gate for the onboarding Continue button AND the first stage of the
full verification pipeline (both call :func:`assess`, single source of truth).

What each metric measures (calibrated Sep 2026 on the live box):

- Face: Haar frontal-face cascade. Tuned for small printed ID portraits
  (min 30px); it only needs to reject faceless junk (blank sheets, drawings,
  random photos of objects). Real face-matching happens later in the pipeline
  via retinaface + ArcFace, so this stage intentionally does not judge identity.
- Sharpness: max Laplacian variance over a 4x4 grid, NOT the whole-image
  mean. A white card background is legitimately uniform and must not drag the
  score down. Calibrated: sharp card photo ~2500, heavy Gaussian blur ~50.
  Threshold 150 rejects unreadable captures with wide margin.
- Glare: fraction of saturated pixels that form SMALL isolated spots, after
  discarding large connected white regions (the card paper itself is white by
  design — the old global white-pixel ratio rejected nearly every real photo).
  Calibrated: clean card 0.0, coin-sized specular spot on portrait ~0.005.
  Threshold 0.005.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Rejection reason codes. Stable contract: Laundex maps these to user-facing
# messages, so never rename without updating the Laundex handler.
REASON_NO_FACE = "no_face"
REASON_TOO_BLURRY = "too_blurry"
REASON_GLARE = "glare"

MIN_SHARPNESS = 150.0
MAX_GLARE_SPOTS = 0.005
_SATURATION = 248
# Connected white regions larger than this fraction of the frame are card
# paper / background, not reflections.
_PAPER_FRACTION = 0.02


@dataclass
class IdQualityReport:
    acceptable: bool
    blur_score: float
    glare_ratio: float
    has_face: bool
    notes: str
    reasons: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        """Primary rejection code ("" when acceptable)."""
        return self.reasons[0] if self.reasons else ""

    def to_public(self) -> dict:
        return {
            "acceptable": self.acceptable,
            "blur_score": self.blur_score,
            "glare_ratio": self.glare_ratio,
            "has_face": self.has_face,
            "reason": self.reason,
            "reasons": self.reasons,
            "notes": self.notes,
        }


def _decode(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid ID image bytes")
    return img


def blur_score(gray: np.ndarray) -> float:
    """Regional sharpness: max Laplacian variance over a 4x4 grid.

    Uniform areas (white card, desk) score near zero and must not mask the
    textured regions (photo, text, guilloche) that carry identity content.
    """
    h, w = gray.shape
    ys = np.linspace(0, h, 5, dtype=int)
    xs = np.linspace(0, w, 5, dtype=int)
    best = 0.0
    for i in range(4):
        for j in range(4):
            cell = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if cell.size == 0:
                continue
            v = float(cv2.Laplacian(cell, cv2.CV_64F).var())
            best = max(best, v)
    return best


def glare_ratio(gray: np.ndarray, thr: int = _SATURATION) -> float:
    """Specular-reflection fraction: saturated pixels in small isolated spots.

    Large connected white regions (the card itself, a light desk) are paper,
    not glare, and are excluded — otherwise every photo of a white national
    ID is rejected.
    """
    mask = (gray > thr).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    total = gray.size
    spots = 0
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area > _PAPER_FRACTION * total:
            continue
        spots += area
    return float(spots) / float(total)


def has_frontal_face(img_bgr: np.ndarray) -> bool:
    factory = getattr(cv2, "CascadeClassifier", None)
    if factory is None:
        # Wrong OpenCV build (e.g. a 5.x wheel shadowing headless) — fail
        # closed on this signal only; blur/glare still gate, and the
        # pipeline's retinaface match remains authoritative.
        return False
    cascade = factory(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        # Detector data missing — fail closed on this signal only; blur/glare
        # still gate, and the pipeline's retinaface match remains authoritative.
        return False
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
    )
    return len(faces) > 0


def assess(id_image_bytes: bytes) -> IdQualityReport:
    img = _decode(id_image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bs = blur_score(gray)
    gr = glare_ratio(gray)
    face_ok = has_frontal_face(img)
    reasons: list[str] = []
    if not face_ok:
        reasons.append(REASON_NO_FACE)
    if bs < MIN_SHARPNESS:
        reasons.append(REASON_TOO_BLURRY)
    if gr > MAX_GLARE_SPOTS:
        reasons.append(REASON_GLARE)
    return IdQualityReport(
        acceptable=not reasons,
        blur_score=bs,
        glare_ratio=gr,
        has_face=face_ok,
        reasons=reasons,
        notes=f"blur={bs:.1f} glare={gr:.4f} face={face_ok}",
    )
