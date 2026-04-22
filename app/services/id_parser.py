"""ID document quality + structure validation.

This is intentionally minimal — ships sane checks that work without OCR
licences. For production, plug in PaddleOCR / Tesseract for MRZ parsing or
a commercial doc-verification API (Onfido, Veriff, Jumio).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class IdQualityReport:
    acceptable: bool
    blur_score: float
    glare_ratio: float
    has_face: bool
    notes: str

    def to_public(self) -> dict:
        return {
            "acceptable": self.acceptable,
            "blur_score": self.blur_score,
            "glare_ratio": self.glare_ratio,
            "has_face": self.has_face,
            "notes": self.notes,
        }


def _decode(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid ID image bytes")
    return img


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def glare_ratio(gray: np.ndarray, thr: int = 245) -> float:
    return float(np.mean(gray > thr))


def has_frontal_face(img_bgr: np.ndarray) -> bool:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    return len(faces) > 0


def assess(id_image_bytes: bytes) -> IdQualityReport:
    img = _decode(id_image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bs = blur_score(gray)
    gr = glare_ratio(gray)
    face_ok = has_frontal_face(img)
    acceptable = bs >= 80.0 and gr <= 0.05 and face_ok
    return IdQualityReport(
        acceptable=acceptable,
        blur_score=bs,
        glare_ratio=gr,
        has_face=face_ok,
        notes=f"blur={bs:.1f} glare={gr:.3f} face={face_ok}",
    )
