"""Handwritten signature comparison.

Compares a digitally drawn signature (canvas PNG, e.g. from signature_pad on
a touch device) against the printed signature on the back of a Mexican
INE/IFE credential. 100% open-source pipeline:

    - OpenCV (Apache 2.0, https://github.com/opencv/opencv) — image decode,
      crop, adaptive thresholding, morphology, contour detection.
    - scikit-image (BSD-3-Clause, https://github.com/scikit-image/scikit-image)
      — Structural Similarity Index (SSIM), the comparison metric.
    - NumPy (BSD-3-Clause) — array math.

Pipeline
--------
1. Decode both images. Canvas PNGs from signature_pad are alpha-channel
   transparent; we composite them onto a white background so the strokes
   stay dark on light (otherwise IMREAD_COLOR collapses transparent pixels
   to black and the entire image registers as ink).
2. INE back-side has the printed signature in the lower-left band, above
   the MRZ/CIC barcode and to the left of the QR. Crop that band, run an
   adaptive threshold (Gaussian, blockSize=31, C=10) to isolate stroke
   pixels, dilate-and-close to bridge stroke gaps, and pick the largest
   connected component as the signature blob.
3. Tight-crop both images to their ink bounding boxes, letterbox-resize
   to a common 384×192 canvas keeping aspect ratio, Otsu-binarize so SSIM
   compares stroke topology rather than photo texture.
4. Score = max(0, SSIM) × 100 → integer in [0, 100].

The default threshold (60) is informational. Callers control the gate.

Reference / inspiration for the cropping heuristic:
    - https://github.com/Aftaab99/OfflineSignatureVerification (5k★)
      Uses a similar OpenCV + Otsu pipeline, follow-on with a Siamese CNN.
    - https://github.com/luizgh/sigver (3k★)
      Pretrained signature verification networks (CNN feature extractors).

We start with a pure-OpenCV/SSIM baseline to keep the deployment small and
debuggable. A future iteration can swap step 4 for a Siamese network for
higher accuracy on naturally-varying signatures.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


SIGNATURE_THRESHOLD_DEFAULT = 60
MODEL_NAME = "ssim+otsu-v1"


@dataclass
class SignatureMatchResult:
    similarity: int
    ssim_raw: float
    match_pass: bool
    threshold: int
    id_signature_found: bool
    canvas_signature_found: bool
    model: str
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "similarity":              self.similarity,
            "ssim":                    self.ssim_raw,
            "match_pass":              self.match_pass,
            "threshold":               self.threshold,
            "id_signature_found":      self.id_signature_found,
            "canvas_signature_found":  self.canvas_signature_found,
            "model":                   self.model,
            **({"reason": self.reason} if self.reason else {}),
        }


# ── Decoders ───────────────────────────────────────────────────────────────

def _decode_color(buf: bytes, label: str) -> np.ndarray:
    """Decode an opaque image (JPEG/PNG/WebP) as BGR. Raises ValueError on
    failure so the caller can surface a clean 422.
    """
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"image_decode_failed_{label}")
    return img


def _decode_signature_with_alpha(buf: bytes) -> np.ndarray:
    """Decode a canvas signature PNG and composite alpha onto a white
    background. signature_pad emits transparent PNGs with dark strokes;
    cv2.IMREAD_COLOR drops the alpha channel and collapses transparent
    pixels to black, which makes the entire image read as "ink" downstream.
    """
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("image_decode_failed_signature")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        a_f = a.astype(np.float32) / 255.0
        white = np.full_like(b, 255, dtype=np.float32)
        b = (b.astype(np.float32) * a_f + white * (1.0 - a_f)).astype(np.uint8)
        g = (g.astype(np.float32) * a_f + white * (1.0 - a_f)).astype(np.uint8)
        r = (r.astype(np.float32) * a_f + white * (1.0 - a_f)).astype(np.uint8)
        return cv2.merge([b, g, r])
    return img


# ── Croppers ───────────────────────────────────────────────────────────────

def _crop_signature_region_ine(gray: np.ndarray) -> np.ndarray | None:
    """INE/IFE back-side has the printed signature in a lower-left band:
    roughly y in [55%, 92%] of the image height, x in [4%, 62%] of the
    width. Within that band, find the largest dark connected component
    and tight-crop to it.

    Returns None if no significant ink blob is found in the band.
    """
    h, w = gray.shape[:2]
    y0 = int(h * 0.55)
    y1 = int(h * 0.92)
    x0 = int(w * 0.04)
    x1 = int(w * 0.62)
    band = gray[y0:y1, x0:x1]
    if band.size == 0:
        return None

    bin_inv = cv2.adaptiveThreshold(
        band, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=31, C=10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)
    if cw * ch < (band.shape[0] * band.shape[1]) * 0.01:
        return None
    pad = 6
    xa = max(0, x - pad)
    ya = max(0, y - pad)
    xb = min(band.shape[1], x + cw + pad)
    yb = min(band.shape[0], y + ch + pad)
    return band[ya:yb, xa:xb]


def _crop_canvas_signature(gray: np.ndarray) -> np.ndarray | None:
    """Canvas signatures are usually black strokes on white background
    (after alpha compositing). Threshold to isolate ink, then bbox-crop.
    """
    _, bin_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(bin_inv > 0)
    if len(xs) < 50:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad = 6
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(gray.shape[1] - 1, x1 + pad)
    y1 = min(gray.shape[0] - 1, y1 + pad)
    return gray[y0:y1 + 1, x0:x1 + 1]


def _fit_to(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Letterbox-resize keeping aspect ratio. Padding is white (255) so the
    Otsu threshold picks ink vs background cleanly afterward.
    """
    ih, iw = img.shape[:2]
    if iw == 0 or ih == 0:
        return np.full((target_h, target_w), 255, dtype=np.uint8)
    scale = min(target_w / iw, target_h / ih)
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


# ── Main entry point ───────────────────────────────────────────────────────

def compare_signatures(
    id_back_bytes: bytes,
    signature_bytes: bytes,
    threshold: int = SIGNATURE_THRESHOLD_DEFAULT,
) -> SignatureMatchResult:
    """Compare a canvas signature against the signature region cropped from
    the back of an INE/IFE.

    Args:
        id_back_bytes: Raw bytes of the INE/IFE back-side image (JPEG/PNG/WebP).
        signature_bytes: Raw bytes of the canvas signature PNG (transparent OK).
        threshold: Pass/fail threshold on the 0–100 similarity score.

    Returns:
        SignatureMatchResult — always returns a result, never raises beyond
        decode errors. Use the `reason` field to distinguish:
          - None: comparison ran cleanly.
          - "no_signature_in_id_back": couldn't isolate a signature blob on
            the INE back. Likely the upload was the front, blurry, or wrong.
          - "no_canvas_signature": canvas was empty or noise.
    """
    id_back = _decode_color(id_back_bytes, "id_back")
    sig_img = _decode_signature_with_alpha(signature_bytes)

    id_gray  = cv2.cvtColor(id_back, cv2.COLOR_BGR2GRAY)
    sig_gray = cv2.cvtColor(sig_img, cv2.COLOR_BGR2GRAY)

    id_sig_crop  = _crop_signature_region_ine(id_gray)
    canvas_crop  = _crop_canvas_signature(sig_gray)

    if id_sig_crop is None:
        return SignatureMatchResult(
            similarity=0, ssim_raw=-1.0,
            match_pass=False, threshold=threshold,
            id_signature_found=False, canvas_signature_found=canvas_crop is not None,
            model=MODEL_NAME, reason="no_signature_in_id_back",
        )
    if canvas_crop is None:
        return SignatureMatchResult(
            similarity=0, ssim_raw=-1.0,
            match_pass=False, threshold=threshold,
            id_signature_found=True, canvas_signature_found=False,
            model=MODEL_NAME, reason="no_canvas_signature",
        )

    target_w, target_h = 384, 192
    id_norm  = _fit_to(id_sig_crop,  target_w, target_h)
    sig_norm = _fit_to(canvas_crop,  target_w, target_h)

    # Otsu both so SSIM compares stroke topology, not photo texture / shadow
    # / paper colour. Letterbox padding (white) dominates the histogram on
    # both sides so the threshold lands cleanly between ink and background.
    _, id_bin  = cv2.threshold(id_norm,  0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, sig_bin = cv2.threshold(sig_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    score = float(ssim(id_bin, sig_bin, data_range=255))
    similarity = int(max(0, min(100, round(max(0.0, score) * 100))))

    return SignatureMatchResult(
        similarity=similarity, ssim_raw=score,
        match_pass=similarity >= threshold, threshold=threshold,
        id_signature_found=True, canvas_signature_found=True,
        model=MODEL_NAME,
    )


def compare_signatures_b64(
    id_back_bytes: bytes,
    signature_b64: str,
    threshold: int = SIGNATURE_THRESHOLD_DEFAULT,
) -> SignatureMatchResult:
    """Convenience wrapper accepting the canvas signature as a base64 string
    (with or without a `data:image/png;base64,` prefix).
    """
    if signature_b64.startswith("data:"):
        signature_b64 = signature_b64.split(",", 1)[-1]
    try:
        signature_bytes = base64.b64decode(signature_b64)
    except Exception as exc:
        raise ValueError(f"signature_b64_invalid: {exc}") from exc
    return compare_signatures(id_back_bytes, signature_bytes, threshold)
