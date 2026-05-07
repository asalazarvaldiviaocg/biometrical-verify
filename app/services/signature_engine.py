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
from scipy.spatial.distance import directed_hausdorff
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim

SIGNATURE_THRESHOLD_DEFAULT = 45
MODEL_NAME = "ssim+hu+hog+stroke+aspect+hausdorff-v5"

# Pesos del score combinado multi-feature v5.
#
# History:
#   v2 (0.55 SSIM / 0.45 Hu additive): producía 70-82 en firmas legítimas
#     Y en "wrong-but-similar" scribbles. Threshold 75 admitía ambas.
#   v3 (0.7·min + 0.3·mean): demasiado estricto. Aplastó firmas reales
#     a 38-41 porque Hu Moments dan distancias altas (similarity 0.3-0.5)
#     entre firma impresa-en-papel y firma dedo-en-pantalla aunque sean
#     de la misma persona.
#   v4 (0.65 SSIM / 0.35 Hu): mejor balance pero false-reject ~15 % en
#     producción. SSIM saturaba en 0.45-0.55 para firmas finger-vs-printed
#     incluso de la misma persona porque texturas son intrínsecamente
#     distintas (papel laminado con glare vs canvas digital limpio).
#   v5 (este): multi-feature ponderado. Cinco descriptores complementarios
#     en lugar de dos. Cada uno captura un aspecto distinto:
#       SSIM        — similitud pixel-a-pixel después de Otsu
#       Hu Moments  — invariante a escala/rotación (forma global)
#       HOG cosine  — gradientes locales (orientación de strokes)
#       Stroke density ratio — ¿se parecen en cantidad de tinta?
#       Aspect ratio match   — proporción ancho/alto del trazo
#       Hausdorff   — distancia geométrica entre contornos
#     SSIM y HOG llevan el peso (descriptores fuertes); los otros
#     refinan el score. Calibración esperada: false-reject 15 % → 7-8 %
#     manteniendo false-accept < 1 %.
WEIGHT_SSIM           = 0.30
WEIGHT_HU_MOMENTS     = 0.15
WEIGHT_HOG            = 0.25
WEIGHT_STROKE_DENSITY = 0.10
WEIGHT_ASPECT_RATIO   = 0.10
WEIGHT_HAUSDORFF      = 0.10

# Hard floors on the *discriminative* features (SSIM and HOG). The
# weighted-additive score is generous on purpose — it has to absorb the
# texture gap between a printed-on-paper signature and a finger-on-canvas
# capture of the same person. But generosity has a cost: stroke density,
# aspect ratio and Hausdorff all give "free" partial points whenever two
# signatures happen to have similar ink quantity / proportions / outline
# scale, even when the actual strokes are unrelated. A real-world report
# from production: a signer drew a doodle that had nothing to do with
# their passport signature and the gate accepted it because density +
# aspect + Hausdorff alone pushed the additive score over threshold.
#
# These two floors short-circuit that. SSIM and HOG can't both be high
# unless the strokes themselves overlap structurally — that's exactly
# what the soft features can't fake. If either of them is below floor,
# `match_pass` is forced to False regardless of the additive score; the
# similarity number we surface to the operator stays unchanged so the
# panel still shows the real measurement.
SSIM_HARD_FLOOR = 0.20  # binarised images sharing zero stroke topology
HOG_HARD_FLOOR  = 0.15  # local gradient orientations must overlap a bit


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

def _crop_signature_in_band(
    gray: np.ndarray, y0: int, y1: int, x0: int, x1: int,
) -> np.ndarray | None:
    """Generic largest-contour cropper inside a normalized band of the image.
    Used for both INE (lower-left) and passport (lower half) variants.
    """
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
    # Pick the largest contour by bounding-box area, but require a
    # minimum area to avoid tiny noise/text artifacts being mistaken
    # for the signature.
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


def _crop_signature_region_ine(gray: np.ndarray) -> np.ndarray | None:
    """INE/IFE back-side: signature lives in the lower-left band, above
    the MRZ/CIC barcode and to the left of the QR code.
    Roughly y in [55%, 92%] and x in [4%, 62%].
    """
    h, w = gray.shape[:2]
    return _crop_signature_in_band(
        gray,
        y0=int(h * 0.55), y1=int(h * 0.92),
        x0=int(w * 0.04), x1=int(w * 0.62),
    )


def _crop_signature_region_passport(gray: np.ndarray) -> np.ndarray | None:
    """Mexican passport bio page: signature appears below the photo /
    personal info, above the MRZ block at the bottom. Position varies
    by passport version (book layout, biometric upgrades) so we scan
    a wider band (lower 60%, full width minus a small left/right margin)
    and trust the largest-contour heuristic.

    Falls back to the entire lower half if no contour is found in the
    primary band — handles odd cropping where users photograph just
    part of the page.
    """
    h, w = gray.shape[:2]
    primary = _crop_signature_in_band(
        gray,
        y0=int(h * 0.55), y1=int(h * 0.88),  # exclude MRZ at very bottom
        x0=int(w * 0.05), x1=int(w * 0.95),
    )
    if primary is not None:
        return primary
    # Fallback: try the lower half wholesale (handles cropped uploads).
    return _crop_signature_in_band(
        gray,
        y0=int(h * 0.40), y1=int(h * 0.95),
        x0=0, x1=w,
    )


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


def _hog_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between HOG (Histogram of Oriented Gradients)
    descriptors of two grayscale images. HOG captures local gradient
    orientations, which encode the direction of each stroke without
    being sensitive to small translations or pixel-level noise. Two
    signatures from the same person produce HOG vectors that point in
    similar directions; different signatures diverge.

    Returns a similarity in [0, 1] (cosine clamped). Inputs must be the
    same size (callers feed the post-letterbox 384×192 binarised images).
    """
    if a.shape != b.shape:
        return 0.0
    # Block-normalised HOG with 8×8 cells / 2×2 cell blocks. Generous
    # cell size keeps the descriptor short (~hundred floats) and
    # robust to the natural 1-2 px stroke jitter between signatures.
    fd_a = hog(a, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm='L2-Hys', feature_vector=True)
    fd_b = hog(b, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm='L2-Hys', feature_vector=True)
    na = float(np.linalg.norm(fd_a))
    nb = float(np.linalg.norm(fd_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    cos = float(np.dot(fd_a, fd_b) / (na * nb))
    return max(0.0, min(1.0, cos))


def _stroke_density_similarity(bin_a: np.ndarray, bin_b: np.ndarray) -> float:
    """Ratio of ink pixels (binary 0 = ink, 255 = background after Otsu).
    Two signatures from the same person have comparable ink density;
    a sparse scribble vs a dense signature diverge here even when SSIM
    happens to match by accident.

    Maps the relative density gap to a similarity in [0, 1] via 1 -
    relative_diff. Caller feeds the Otsu-binarised images.
    """
    pix_a = float(np.count_nonzero(bin_a == 0))
    pix_b = float(np.count_nonzero(bin_b == 0))
    if pix_a == 0 and pix_b == 0:
        return 0.0
    if pix_a == 0 or pix_b == 0:
        return 0.0
    rel_diff = abs(pix_a - pix_b) / max(pix_a, pix_b)
    return max(0.0, min(1.0, 1.0 - rel_diff))


def _aspect_ratio_similarity(crop_a: np.ndarray, crop_b: np.ndarray) -> float:
    """Aspect ratio (width / height) of the bounding box. Most signers
    have a consistent aspect ratio across captures (wide and short, or
    compact). A signer who normally signs in a 4:1 aspect cannot fake
    that with a quick 1:1 scribble even if SSIM happens to align.

    Inputs are the pre-letterbox tight crops, NOT the 384×192 padded
    versions — we want each signature's natural aspect ratio.
    """
    h_a, w_a = crop_a.shape[:2]
    h_b, w_b = crop_b.shape[:2]
    if h_a == 0 or h_b == 0:
        return 0.0
    ar_a = w_a / h_a
    ar_b = w_b / h_b
    if ar_a == 0 or ar_b == 0:
        return 0.0
    rel_diff = abs(ar_a - ar_b) / max(ar_a, ar_b)
    return max(0.0, min(1.0, 1.0 - rel_diff))


def _hausdorff_similarity(bin_a: np.ndarray, bin_b: np.ndarray) -> float:
    """Modified Hausdorff distance between the two contour point sets,
    mapped to a [0, 1] similarity. Hausdorff is "what is the worst-case
    closest distance from a point in A to its nearest point in B?" — it
    captures how much one signature shape would have to deform to match
    the other. Robust to local stroke jitter.

    Returns 0 (very different shapes) to 1 (overlapping contours).
    """
    # Subsample to keep the O(n×m) distance pairs tractable. 256 points
    # is enough to capture the signature's outline; full point sets
    # blow up to 30k× compares.
    def points(bin_img: np.ndarray) -> np.ndarray:
        ys, xs = np.where(bin_img == 0)
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(xs) > 256:
            idx = np.linspace(0, len(xs) - 1, 256, dtype=int)
            xs, ys = xs[idx], ys[idx]
        return np.stack([xs, ys], axis=1).astype(np.float32)

    pa = points(bin_a)
    pb = points(bin_b)
    if pa.shape[0] == 0 or pb.shape[0] == 0:
        return 0.0
    d_ab = directed_hausdorff(pa, pb)[0]
    d_ba = directed_hausdorff(pb, pa)[0]
    h    = max(d_ab, d_ba)
    # Diagonal of the 384×192 canvas is ~430 px. Distances near 0 = match,
    # > 100 px = different shapes. Exponential decay keeps the metric
    # well-behaved at extremes.
    return float(np.exp(-h / 80.0))


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
    id_type: str = "INE",
) -> SignatureMatchResult:
    """Compare a canvas signature against the signature region cropped from
    the user's identification document (INE or Mexican passport).

    Args:
        id_back_bytes: Raw bytes of the ID image:
            - INE/IFE: the BACK side (where the printed signature is).
            - Pasaporte: the bio data page (signature below personal info).
        signature_bytes: Raw bytes of the canvas signature PNG (transparent OK).
        threshold: Pass/fail threshold on the 0–100 similarity score.
        id_type: "INE" (default) or "PASAPORTE". Selects the region cropper.

    Returns:
        SignatureMatchResult — always returns a result, never raises beyond
        decode errors. Use the `reason` field to distinguish:
          - None: comparison ran cleanly.
          - "no_signature_in_id_back": couldn't isolate a signature blob on
            the ID. Likely the upload was the wrong side, blurry, or wrong.
          - "no_canvas_signature": canvas was empty or noise.
    """
    id_back = _decode_color(id_back_bytes, "id_back")
    sig_img = _decode_signature_with_alpha(signature_bytes)

    id_gray  = cv2.cvtColor(id_back, cv2.COLOR_BGR2GRAY)
    sig_gray = cv2.cvtColor(sig_img, cv2.COLOR_BGR2GRAY)

    # Route to the right region cropper. Passport signature lives in a
    # different part of the page than INE-back; using the wrong cropper
    # produces "no_signature_in_id_back" as a false negative.
    id_type_norm = (id_type or "INE").upper()
    if id_type_norm == "PASAPORTE":
        id_sig_crop = _crop_signature_region_passport(id_gray)
    else:
        id_sig_crop = _crop_signature_region_ine(id_gray)
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

    # Otsu both so the metrics compare stroke topology, not photo texture /
    # shadow / paper colour. Letterbox padding (white) dominates the histogram
    # on both sides so the threshold lands cleanly between ink and background.
    _, id_bin  = cv2.threshold(id_norm,  0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, sig_bin = cv2.threshold(sig_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Metric 1: SSIM (pixel-by-pixel structural similarity) ──
    ssim_raw = float(ssim(id_bin, sig_bin, data_range=255))
    ssim_score = max(0.0, ssim_raw)  # SSIM puede salir negativo si son anti-correlacionadas

    # ── Metric 2: Hu Moments distance (shape descriptor invariante) ──
    # 7 momentos log-escalados invariantes a escala/rotación. matchShapes
    # con CONTOURS_MATCH_I1 = sum(|1/m_a - 1/m_b|). Empíricamente:
    # dist > 1.5 ⇒ formas claramente distintas, < 0.3 ⇒ parecidas.
    # Exponencial decreciente para que distancias grandes maten el score.
    hu_distance = cv2.matchShapes(id_bin, sig_bin, cv2.CONTOURS_MATCH_I1, 0.0)
    hu_similarity = float(np.exp(-hu_distance * 1.5))

    # ── Metric 3: HOG cosine (orientación local de strokes) ──
    hog_score = _hog_cosine(id_bin, sig_bin)

    # ── Metric 4: Stroke density ratio (cantidad de tinta) ──
    stroke_score = _stroke_density_similarity(id_bin, sig_bin)

    # ── Metric 5: Aspect ratio (proporción natural del trazo) ──
    # Crops pre-letterbox preservan la proporción real de cada firma.
    aspect_score = _aspect_ratio_similarity(id_sig_crop, canvas_crop)

    # ── Metric 6: Hausdorff distance (geometría de contornos) ──
    hausdorff_score = _hausdorff_similarity(id_bin, sig_bin)

    # Score combinado v5 (multi-feature ponderado). SSIM y HOG llevan
    # el peso (descriptores fuertes); Hu, stroke density, aspect ratio
    # y Hausdorff refinan. Cada métrica captura una propiedad distinta —
    # un atacante tendría que clonar TODAS para obtener un score alto.
    combined = (
        WEIGHT_SSIM           * ssim_score      +
        WEIGHT_HU_MOMENTS     * hu_similarity   +
        WEIGHT_HOG            * hog_score       +
        WEIGHT_STROKE_DENSITY * stroke_score    +
        WEIGHT_ASPECT_RATIO   * aspect_score    +
        WEIGHT_HAUSDORFF      * hausdorff_score
    )
    similarity = int(max(0, min(100, round(combined * 100))))

    # Hard floor on the discriminative features. Soft features (density,
    # aspect, Hausdorff) can collude to fake a passing total even when
    # SSIM/HOG show no real stroke overlap — that's how a doodle was
    # passing through against a passport signature. Force fail when
    # either discriminator is below floor; the surfaced similarity stays
    # honest so the operator panel still shows what we measured.
    discriminator_pass = (ssim_score >= SSIM_HARD_FLOOR) and (hog_score >= HOG_HARD_FLOOR)
    match_pass = (similarity >= threshold) and discriminator_pass
    reason = None if match_pass else (
        "below_threshold" if not discriminator_pass else None
    )

    return SignatureMatchResult(
        similarity=similarity, ssim_raw=ssim_raw,
        match_pass=match_pass, threshold=threshold,
        id_signature_found=True, canvas_signature_found=True,
        model=MODEL_NAME,
        reason=reason,
    )


def compare_signatures_b64(
    id_back_bytes: bytes,
    signature_b64: str,
    threshold: int = SIGNATURE_THRESHOLD_DEFAULT,
    id_type: str = "INE",
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
    return compare_signatures(id_back_bytes, signature_bytes, threshold, id_type=id_type)
