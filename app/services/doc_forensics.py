"""Document forensics for AI-generated / synthetic ID detection.

Layered, review-only signals — nothing here rejects an applicant. A flag
downgrades APPROVE to REVIEW with the reason recorded; a human decides.

Signals (calibrated Sep 2026 on the live box):

- phash: 64-bit DCT perceptual hash of the ID image. Near-duplicates across
  accounts (AI-ID services reuse outputs; fraud rings share photos) are the
  highest-precision synthetic-ID signal available without GPU classifiers.
- structural: card quadrilateral aspect (~1.586 for ID-1), text-row count,
  face-inside-card. AI layouts drift; screen photos of real cards pass.
- spectral_hf / noise_residual: RECORDED ONLY. Measured Sep 2026 on real
  captures (hf 0.947-0.957), a screen-photo fake (0.954) and pure StyleGAN3
  faces (0.958): whole-image spectra do NOT separate camera from synthetic
  at this scale. No threshold is applied — the numbers land in the audit
  event so a future classifier can be tuned on observed data. Shipping a
  cutoff here would be security theatre.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

# ID-1 card aspect (85.60 x 53.98 mm) with tolerance for perspective + crop.
CARD_ASPECT = 85.60 / 53.98
CARD_ASPECT_TOL = 0.18
# Text-row peaks in the horizontal edge profile. Calibrated Sep 2026:
# real card photos 16-27, blank/heavily-blurred 0. Below 5 means no
# readable row structure at all.
MIN_TEXT_PEAKS = 5

# Garble screen: AI-generated IDs garble microtext into consonant soup.
# wordlike_ratio = vowel-containing, sane-consonant-run tokens / all alpha
# tokens (len>=4). Calibrated Sep 2026 on sidecar OCR: noisy real-card read
# 1.00 (17 tokens), sample-card read 0.92 (12), synthetic garble 0.00.
# Threshold 0.50 with >=4 scorable tokens keeps both real-world reads clear
# with wide margin; only severe garble flags.
GARBLE_WORDLIKE_MIN = 0.50
GARBLE_MIN_TOKENS = 4
_VOWELS = set("aeiouAEIOU")


@dataclass
class DocForensicsReport:
    phash: str
    card_found: bool
    card_aspect: float
    text_rows: int
    face_in_card: bool
    spectral_hf: float
    noise_residual: float
    reasons: list[str] = field(default_factory=list)

    def to_public(self) -> dict:
        return {
            "phash": self.phash,
            "card_found": self.card_found,
            "card_aspect": round(self.card_aspect, 3),
            "text_rows": self.text_rows,
            "face_in_card": self.face_in_card,
            "spectral_hf": round(self.spectral_hf, 4),
            "noise_residual": round(self.noise_residual, 2),
            "reasons": self.reasons,
        }


def phash(img_bgr: np.ndarray) -> str:
    """64-bit DCT perceptual hash, hex-encoded. Robust to JPEG recompression
    and minor crops; Hamming distance <= 6 ~ same image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)
    low = d[:8, :8]
    med = float(np.median(low))
    bits = (low > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


def phash_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hex phashes."""
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def garble_score(text_lines: list[str]) -> tuple[float, int]:
    """Wordlike ratio over OCR alpha tokens. Returns (ratio, n_tokens);
    (1.0, 0) when nothing scorable — inconclusive, never suspicious."""
    toks = [w for ln in text_lines for w in re.split(r"\s+", ln)]
    scores = []
    for w in toks:
        t = re.sub(r"[^A-Za-z]", "", w)
        if len(t) < 4:
            continue
        has_vowel = any(c in _VOWELS for c in t)
        run = maxrun = 0
        for c in t:
            run = 0 if c in _VOWELS else run + 1
            maxrun = max(maxrun, run)
        scores.append(has_vowel and maxrun < 4)
    if len(scores) < GARBLE_MIN_TOKENS:
        return 1.0, len(scores)
    return sum(scores) / len(scores), len(scores)


def is_garbled(text_lines: list[str]) -> tuple[bool, float, int]:
    """(flagged, ratio, n). Flags only severe, well-evidenced garble."""
    ratio, n = garble_score(text_lines)
    return (n >= GARBLE_MIN_TOKENS and ratio < GARBLE_WORDLIKE_MIN), ratio, n


def _decode(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid ID image bytes")
    return img


def spectral_metrics(img_bgr: np.ndarray) -> tuple[float, float]:
    """Whole-image high-frequency energy ratio + sensor-noise residual std.
    Recorded for future tuning; NOT thresholded (see module docstring)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    mask = np.ones_like(mag, dtype=bool)
    mask[cy - r:cy + r, cx - r:cx + r] = False
    hf = float(mag[mask].mean() / (mag.mean() + 1e-6))
    resid = gray - cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
    return hf, float(resid.std())


def _text_peak_count(gray: np.ndarray) -> int:
    """Distinct horizontal-structure peaks over the full frame.

    Counts local maxima (not binary transitions): a uniformly edge-dense
    region — photo texture, screen moiré — yields few clean peaks, while
    printed text lines, rules and card borders each contribute one. Blank
    and heavily blurred captures score 0.
    """
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    rows = (edges > 0).sum(axis=1).astype(float)
    smooth = np.convolve(rows, np.ones(7) / 7, mode="same")
    floor = max(0.02 * w, 0.3 * smooth.max())
    peaks = 0
    i = 10
    while i < len(smooth) - 10:
        if smooth[i] > floor and smooth[i] == smooth[i - 10:i + 11].max():
            peaks += 1
            i += 15  # one line's peak, not its shoulders
        else:
            i += 1
    return peaks


def structural_metrics(img_bgr: np.ndarray) -> tuple[bool, float, int, bool]:
    """(card_found, aspect, text_peaks, face_in_card). All best-effort: any
    step may fail on odd captures and simply reports False/0."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_found, aspect = False, 0.0
    card_box = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 0.08 * gray.size:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(c if len(approx) != 4 else approx)
            if w > 0 and h > 0:
                aspect = max(w, h) / min(w, h)
                card_found = True
                card_box = (x, y, w, h)
    text_rows = _text_peak_count(gray)
    # Face inside card bounds (printed portrait, not a bystander behind).
    face_in_card = False
    # Same Haar detector family as the quality gate: only needs to confirm a
    # printed portrait sits inside the card bounds, not identify anyone.
    try:
        cascade = None
        if hasattr(cv2, "CascadeClassifier"):
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        if cascade is not None and not cascade.empty():
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) > 0 and card_box is not None:
                x, y, w0, h0 = card_box
                for (fx, fy, fw, fh) in faces:
                    cx, cy = fx + fw / 2, fy + fh / 2
                    if x <= cx <= x + w0 and y <= cy <= y + h0:
                        face_in_card = True
                        break
            elif len(faces) > 0:
                face_in_card = True  # no card bounds; face present at least
    except Exception:
        pass
    return card_found, aspect, text_rows, face_in_card


def analyse(id_image_bytes: bytes) -> DocForensicsReport:
    img = _decode(id_image_bytes)
    h = phash(img)
    card_found, aspect, text_rows, face_in_card = structural_metrics(img)
    hf, nz = spectral_metrics(img)
    reasons: list[str] = []
    if card_found and abs(aspect - CARD_ASPECT) > CARD_ASPECT_TOL:
        reasons.append("card geometry off-template — possible synthetic layout")
    if text_rows < MIN_TEXT_PEAKS:
        reasons.append("no readable row structure — possible synthetic")
    return DocForensicsReport(
        phash=h,
        card_found=card_found,
        card_aspect=aspect,
        text_rows=text_rows,
        face_in_card=face_in_card,
        spectral_hf=hf,
        noise_residual=nz,
        reasons=reasons,
    )
