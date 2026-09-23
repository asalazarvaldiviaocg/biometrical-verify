"""Unit tests for doc forensics (app.services.doc_forensics).

Synthetic fixtures only — no PII, no network. Pins: phash stability under
recompression, Hamming behaviour, structural sanity, and the rule that
spectral metrics are recorded but never verdicts.
"""

import cv2
import numpy as np

from app.services import doc_forensics


def _jpg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return bytes(buf)


def _card() -> np.ndarray:
    img = np.full((900, 1400, 3), 235, np.uint8)
    cv2.rectangle(img, (150, 100), (1250, 800), (250, 250, 250), -1)
    cv2.rectangle(img, (250, 200), (470, 430), (80, 80, 80), -1)
    for y in range(250, 700, 60):
        cv2.rectangle(img, (700, y), (1150, y + 22), (60, 60, 60), -1)
    return img


def _decode_phash(jpg_bytes: bytes) -> str:
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    return doc_forensics.phash(cv2.imdecode(arr, cv2.IMREAD_COLOR))


def test_phash_stable_under_recompression():
    h1 = _decode_phash(_jpg(_card(), 90))
    h2 = _decode_phash(_jpg(_card(), 60))
    assert doc_forensics.phash_distance(h1, h2) <= 6


def test_phash_distinguishes_images():
    h1 = _decode_phash(_jpg(_card()))
    h2 = _decode_phash(_jpg(np.zeros((900, 1400, 3), np.uint8)))
    assert doc_forensics.phash_distance(h1, h2) > 6


def test_phash_distance_symmetry():
    assert doc_forensics.phash_distance("0" * 16, "f" * 16) == 64


def test_spectral_recorded_never_flagged():
    rep = doc_forensics.analyse(_jpg(_card()))
    pub = rep.to_public()
    assert "spectral_hf" in pub and "noise_residual" in pub
    assert not any("spectral" in r or "noise" in r for r in rep.reasons)


def test_off_template_geometry_flagged():
    # Tall portrait-aspect card on dark background: geometry must flag.
    img = np.full((1200, 600, 3), 40, np.uint8)
    cv2.rectangle(img, (50, 50), (550, 1150), (250, 250, 250), -1)
    rep = doc_forensics.analyse(_jpg(img))
    assert rep.card_found
    assert any("geometry" in r for r in rep.reasons)


def test_blank_has_no_text_rows():
    rep = doc_forensics.analyse(_jpg(np.full((600, 800, 3), 255, np.uint8)))
    assert rep.text_rows < doc_forensics.MIN_TEXT_PEAKS
    assert any("row structure" in r for r in rep.reasons)


def test_textured_card_has_row_structure():
    rep = doc_forensics.analyse(_jpg(_card()))
    assert rep.text_rows >= doc_forensics.MIN_TEXT_PEAKS, rep.text_rows
    assert not any("row structure" in r for r in rep.reasons)


def test_noisy_real_ocr_not_garbled():
    real = ["ECOWAS IDENTITY CARD.", "ES REPUBLIC OF 2 oe",
            "Ee CEDRIC DRIC. AKUE TTEH MENSAH", "ae ee RR LON Ste"]
    flagged, ratio, n = doc_forensics.is_garbled(real)
    assert not flagged, (ratio, n)


def test_sample_card_ocr_not_garbled():
    fake = ["py) IDENTIFICATION CARD *", "Pennsylvania", "visitPA.com USA",
            "yin Hf) RARRISBURG, PA 17101-0000", "hei SMM SEX:F 18 EYES: BRO"]
    flagged, ratio, n = doc_forensics.is_garbled(fake)
    assert not flagged, (ratio, n)


def test_consonant_soup_flagged():
    flagged, ratio, n = doc_forensics.is_garbled(
        ["XKQV ZHTPR MNBWK", "QWRTZXCV BNMDLK", "PLKJHGFDSAQW ZXCV"])
    assert flagged and ratio == 0.0


def test_too_little_text_never_flags():
    flagged, _, n = doc_forensics.is_garbled(["ABC"])
    assert not flagged and n < doc_forensics.GARBLE_MIN_TOKENS
