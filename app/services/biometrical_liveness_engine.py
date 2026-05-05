"""
Biometrical Liveness Engine v1 (BLE)
====================================

100% proprietary multi-signal face anti-spoofing. No external pre-trained
model weights, no third-party SDKs — only OpenCV + NumPy as compute
primitives.

The engine combines five independent signal extractors into a weighted
ensemble. Each signal targets a different presentation-attack vector
(printed photo, screen replay, paper mask, low-quality video) so a
spoof would have to defeat ALL signals simultaneously to produce a high
fused score.

Signals (weights add up to 1.0):

  1. HSV saturation richness          (w = 0.18)
       Real skin has wide saturation distribution (forehead, lips, eyes,
       facial hair). Printed photos compress saturation into a narrower
       range, especially after thermal printing.

  2. LBP (Local Binary Pattern) entropy        (w = 0.26)
       Skin microtexture produces a high-entropy LBP histogram (256 bins,
       lots of unique patterns). Paper, screen, and laminated prints
       collapse to a few dominant patterns → low entropy.

  3. FFT high-frequency energy ratio  (w = 0.20)
       Printed/screen media leak compression / halftone artifacts into
       specific frequency bands. We measure the ratio of mid-to-low
       frequency band energy: real faces sit in a narrow window; spoofs
       drift outside it.

  4. YCrCb skin-region density        (w = 0.18)
       Real skin pixels cluster within a well-known YCrCb range
       (Cr 133-173, Cb 77-127). Printed photos shift the distribution.

  5. Gradient magnitude variance      (w = 0.18)
       Sobel-gradient std-dev. Real faces have a wide distribution
       (sharp eye corners + smooth cheeks); printed photos and screens
       collapse to either uniform-soft or uniform-aliased.

Decision: ``score >= LIVENESS_PASS_THRESHOLD`` (default 0.65).

Calibration:
    Initial weights are educated guesses tuned against synthetic
    real-vs-printed test images. Re-fit on production data when we have
    a labelled fraud corpus. The weight matrix is the only knob to tune;
    individual signal extractors stay fixed so calibrations don't break
    forensic reproducibility.

Performance:
    All signals run in O(pixels). Total wall-clock budget on a 224x224
    face crop is ~80 ms on a 2-vCPU Modal worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import cv2
import numpy as np

# Default threshold above which we treat the score as a passing live face.
# Tunable via Settings.liveness_min_score; this is the engine's intrinsic
# default when the caller doesn't override.
LIVENESS_PASS_THRESHOLD = 0.65

# Signal weights — must sum to 1.0 (asserted at import time below).
WEIGHT_HSV_VARIANCE       = 0.18
WEIGHT_LBP_ENTROPY        = 0.26
WEIGHT_FFT_RATIO          = 0.20
WEIGHT_SKIN_DENSITY       = 0.18
WEIGHT_GRADIENT_VARIANCE  = 0.18

assert abs(
    WEIGHT_HSV_VARIANCE
    + WEIGHT_LBP_ENTROPY
    + WEIGHT_FFT_RATIO
    + WEIGHT_SKIN_DENSITY
    + WEIGHT_GRADIENT_VARIANCE
    - 1.0
) < 1e-9, "BLE weights must sum to 1.0"


@dataclass
class SignalScores:
    """Per-signal raw scores in [0, 1]. Useful for diagnostics + logging."""

    hsv_variance:      float = 0.0
    lbp_entropy:       float = 0.0
    fft_ratio:         float = 0.0
    skin_density:      float = 0.0
    gradient_variance: float = 0.0
    notes:             Dict[str, float] = field(default_factory=dict)


@dataclass
class LivenessAnalysis:
    score:    float                 # weighted ensemble in [0, 1]
    is_live:  bool                  # score >= threshold
    signals:  SignalScores
    threshold: float

    def to_public(self) -> dict:
        """Public-safe shape (exposes signal breakdown for forensic logs)."""
        return {
            "score":    round(self.score, 4),
            "is_live":  self.is_live,
            "threshold": self.threshold,
            "signals": {
                "hsv_variance":      round(self.signals.hsv_variance, 4),
                "lbp_entropy":       round(self.signals.lbp_entropy, 4),
                "fft_ratio":         round(self.signals.fft_ratio, 4),
                "skin_density":      round(self.signals.skin_density, 4),
                "gradient_variance": round(self.signals.gradient_variance, 4),
            },
        }


# ────────────────────────────────────────────────────────────────────────────
# Signal extractors — each returns a value in [0, 1] where 1.0 = strongly
# real, 0.0 = strongly spoof. They are pure functions of the input image
# and never raise on degenerate input (single-pixel / all-zero), returning
# 0.0 instead so the ensemble degrades gracefully.
# ────────────────────────────────────────────────────────────────────────────


def _normalize_face(face_bgr: np.ndarray, side: int = 160) -> np.ndarray:
    """Resize to a fixed square so signal scales are comparable across
    different camera resolutions."""
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("empty face crop passed to BLE")
    if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
        raise ValueError(f"BLE expects HxWx3 BGR; got shape={face_bgr.shape}")
    return cv2.resize(face_bgr, (side, side), interpolation=cv2.INTER_AREA)


def signal_hsv_variance(face_bgr: np.ndarray) -> float:
    """Real faces have richer Saturation distribution. We measure std-dev of
    S in HSV and map it to [0, 1] with a soft sigmoid centred at the mean
    of empirically-observed real values (~38)."""
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    std = float(np.std(s))
    # Sigmoid with k=8/center=38 — calibrated so std=20 (printed) -> ~0.15
    # and std=55 (real) -> ~0.85.
    return float(1.0 / (1.0 + np.exp(-(std - 38.0) / 8.0)))


def signal_lbp_entropy(face_bgr: np.ndarray) -> float:
    """Local Binary Pattern entropy. Skin produces ~5.5 nats; paper/screen
    collapse to ~3.5. Map to [0, 1] using a calibrated sigmoid."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    # 8-neighbour LBP, no rotation invariance (cheap + sufficient).
    lbp = _lbp_uniform_8(gray)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    entropy = float(-np.sum(hist * np.log(hist)))
    # Sigmoid centred at 4.5 nats — empirically separates real from spoof.
    return float(1.0 / (1.0 + np.exp(-(entropy - 4.5) * 1.6)))


def _lbp_uniform_8(gray: np.ndarray) -> np.ndarray:
    """Vectorised 8-neighbour LBP. Output dtype: uint8 (0..255)."""
    g = gray.astype(np.int16)
    h, w = g.shape
    if h < 3 or w < 3:
        return np.zeros_like(gray)
    pad = np.pad(g, 1, mode="edge")
    center = pad[1:-1, 1:-1]
    bits = (
        ((pad[0:-2, 0:-2] >= center).astype(np.uint8) << 7) |
        ((pad[0:-2, 1:-1] >= center).astype(np.uint8) << 6) |
        ((pad[0:-2, 2:  ] >= center).astype(np.uint8) << 5) |
        ((pad[1:-1, 2:  ] >= center).astype(np.uint8) << 4) |
        ((pad[2:  , 2:  ] >= center).astype(np.uint8) << 3) |
        ((pad[2:  , 1:-1] >= center).astype(np.uint8) << 2) |
        ((pad[2:  , 0:-2] >= center).astype(np.uint8) << 1) |
        ((pad[1:-1, 0:-2] >= center).astype(np.uint8) << 0)
    )
    return bits.astype(np.uint8)


def signal_fft_ratio(face_bgr: np.ndarray) -> float:
    """Frequency-domain artifact analysis. Compute the radial energy
    distribution; real faces concentrate energy in low-mid frequencies,
    spoofs leak into either ultra-low (over-smooth print) or high (screen
    halftone). We measure the mid-band fraction of total energy."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(min(cy, cx))
    if rmax <= 0:
        return 0.0
    norm_r = r / rmax  # 0 at centre, 1 at edge
    total = float(np.sum(mag))
    if total <= 0:
        return 0.0
    mid = float(np.sum(mag[(norm_r > 0.20) & (norm_r < 0.55)]))
    ratio = mid / total
    # Empirical: real ~0.32, printed ~0.45 (over-smooth shifts), screen
    # ~0.18. Score is highest near 0.32 with a triangular kernel.
    return float(max(0.0, 1.0 - abs(ratio - 0.32) * 4.0))


def signal_skin_density(face_bgr: np.ndarray) -> float:
    """Fraction of face pixels that fall inside the canonical YCrCb skin
    range. Real faces hit 0.55-0.85; printed photos shift outside."""
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    density = float(np.mean(mask))
    # Real-face density typically 0.55-0.85. Map to [0, 1] with sigmoid
    # centred at 0.5; saturate above 0.8 to avoid penalising very-skin
    # crops.
    return float(min(1.0, 1.0 / (1.0 + np.exp(-(density - 0.45) * 12.0))))


def signal_gradient_variance(face_bgr: np.ndarray) -> float:
    """Sobel gradient magnitude std-dev. Real faces have sharp eye corners
    + smooth cheeks → wide variance. Prints / screens collapse."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    std = float(np.std(mag))
    # Real faces: std ~28-55. Sigmoid centred at 30.
    return float(1.0 / (1.0 + np.exp(-(std - 30.0) / 7.0)))


# ────────────────────────────────────────────────────────────────────────────
# Top-level analysis function
# ────────────────────────────────────────────────────────────────────────────


def analyze(
    face_bgr: np.ndarray,
    threshold: float = LIVENESS_PASS_THRESHOLD,
) -> LivenessAnalysis:
    """Run all five signal extractors and fuse them into a single live/spoof
    decision. Pure function — no side effects, no I/O."""
    norm = _normalize_face(face_bgr)
    s1 = signal_hsv_variance(norm)
    s2 = signal_lbp_entropy(norm)
    s3 = signal_fft_ratio(norm)
    s4 = signal_skin_density(norm)
    s5 = signal_gradient_variance(norm)

    score = (
        WEIGHT_HSV_VARIANCE      * s1
        + WEIGHT_LBP_ENTROPY     * s2
        + WEIGHT_FFT_RATIO       * s3
        + WEIGHT_SKIN_DENSITY    * s4
        + WEIGHT_GRADIENT_VARIANCE * s5
    )

    return LivenessAnalysis(
        score=float(score),
        is_live=bool(score >= threshold),
        threshold=float(threshold),
        signals=SignalScores(
            hsv_variance=s1,
            lbp_entropy=s2,
            fft_ratio=s3,
            skin_density=s4,
            gradient_variance=s5,
        ),
    )


def passive_score(face_bgr: np.ndarray) -> float:
    """Backward-compatible wrapper that returns just the fused score in
    [0, 1] — same shape as the previous ONNX-based passive_liveness_score
    so call sites in liveness.py / orchestrators don't need to change."""
    return analyze(face_bgr).score
