"""Unit tests for the ID quality gate (app.services.id_parser).

All fixtures are synthetic numpy images — no PII, no network. They pin the
calibrated behaviour: white card paper is NOT glare, uniform areas do NOT
mask sharpness, and each failure mode reports its own reason code.
"""

import cv2
import numpy as np

from app.services import id_parser


def _jpg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return bytes(buf)


def _card() -> np.ndarray:
    """Simulated phone photo of a white ID card: paper, dark portrait block,
    text lines."""
    img = np.full((900, 1400, 3), 235, np.uint8)
    cv2.rectangle(img, (150, 100), (1250, 800), (250, 250, 250), -1)
    cv2.rectangle(img, (250, 200), (470, 430), (80, 80, 80), -1)
    for y in range(250, 700, 60):
        cv2.rectangle(img, (700, y), (1150, y + 22), (60, 60, 60), -1)
    return img


def test_blank_white_is_no_face_not_glare():
    q = id_parser.assess(_jpg(np.full((600, 800, 3), 255, np.uint8)))
    assert not q.acceptable
    assert id_parser.REASON_NO_FACE in q.reasons
    assert id_parser.REASON_GLARE not in q.reasons


def test_blank_black_is_no_face():
    q = id_parser.assess(_jpg(np.zeros((600, 800, 3), np.uint8)))
    assert not q.acceptable
    assert id_parser.REASON_NO_FACE in q.reasons


def test_white_card_paper_is_not_glare():
    # Faceless card: the only legal complaint is the missing face — paper
    # must never count as a reflection.
    q = id_parser.assess(_jpg(_card()))
    assert id_parser.REASON_NO_FACE in q.reasons
    assert id_parser.REASON_GLARE not in q.reasons
    # ... and the textured regions keep the sharpness score high.
    assert q.blur_score >= id_parser.MIN_SHARPNESS


def test_heavy_blur_reports_too_blurry():
    blurred = cv2.GaussianBlur(_card(), (31, 31), 0)
    q = id_parser.assess(_jpg(blurred))
    assert not q.acceptable
    assert id_parser.REASON_TOO_BLURRY in q.reasons


def test_specular_spot_reports_glare():
    img = _card()
    cv2.circle(img, (360, 315), 55, (255, 255, 255), -1)  # flash on portrait
    q = id_parser.assess(_jpg(img))
    assert q.glare_ratio > 0
    assert id_parser.REASON_GLARE in q.reasons


def test_clean_card_has_no_glare_spots():
    q = id_parser.assess(_jpg(_card()))
    assert q.glare_ratio == 0.0


def test_report_carries_primary_reason():
    q = id_parser.assess(_jpg(np.zeros((600, 800, 3), np.uint8)))
    assert q.reason == q.reasons[0]
    pub = q.to_public()
    assert pub["reason"] == q.reason
    assert pub["reasons"] == q.reasons
