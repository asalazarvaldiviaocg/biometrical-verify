"""Heuristic deepfake / GAN-artifact checks.

These are NOT a substitute for a dedicated deepfake classifier. They catch
common print-and-replay attacks and obvious GAN artifacts. For high-stakes
contracts, layer a commercial detector (FaceTec, iProov) on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DeepfakeReport:
    suspicious: bool
    fft_high_freq_energy: float
    temporal_flicker: float
    notes: str

    def to_public(self) -> dict:
        return {
            "suspicious": self.suspicious,
            "fft_high_freq_energy": self.fft_high_freq_energy,
            "temporal_flicker": self.temporal_flicker,
            "notes": self.notes,
        }


def fft_high_freq_energy(face_bgr: np.ndarray) -> float:
    """GAN faces tend to over-emphasise mid-high spatial frequencies."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    high_band_mask = np.ones_like(mag, dtype=bool)
    high_band_mask[cy - r : cy + r, cx - r : cx + r] = False
    return float(mag[high_band_mask].mean() / (mag.mean() + 1e-6))


def temporal_flicker_score(video_path: str, max_frames: int = 30) -> float:
    """Frame-to-frame brightness diff: deepfakes often show micro-flicker."""
    cap = cv2.VideoCapture(video_path)
    diffs: list[float] = []
    prev: np.ndarray | None = None
    seen = 0
    try:
        while seen < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(float(np.mean(cv2.absdiff(gray, prev))))
            prev = gray
            seen += 1
    finally:
        cap.release()
    if not diffs:
        return 0.0
    return float(np.std(diffs))


def analyse(face_bgr: np.ndarray, video_path: str) -> DeepfakeReport:
    fft_e = fft_high_freq_energy(face_bgr)
    flicker = temporal_flicker_score(video_path)
    suspicious = fft_e > 1.20 or flicker > 12.0
    return DeepfakeReport(
        suspicious=suspicious,
        fft_high_freq_energy=fft_e,
        temporal_flicker=flicker,
        notes=f"fft={fft_e:.3f} flicker={flicker:.2f}",
    )
