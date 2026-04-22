from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services import face_engine


def _jpeg_bytes(color=(127, 127, 127), size=(200, 200)) -> bytes:
    img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_decode_image_rejects_garbage():
    with pytest.raises(ValueError):
        face_engine._decode_image(b"not-an-image")


def test_compare_returns_match_with_stub_embeddings():
    # conftest stubs DeepFace.represent → identical embeddings → similarity ~1.0
    res = face_engine.compare(_jpeg_bytes(), _jpeg_bytes())
    assert res.verified is True
    assert res.similarity == pytest.approx(1.0, abs=1e-3)
    assert res.distance == pytest.approx(0.0, abs=1e-3)
    assert res.model == "ArcFace"


def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert face_engine.cosine_distance(a, b) == pytest.approx(1.0)
