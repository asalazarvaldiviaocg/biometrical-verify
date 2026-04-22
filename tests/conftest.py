"""Test fixtures.

Heavy deps (DeepFace, MediaPipe, ONNX, S3) are stubbed so the suite runs
fast in CI without GPUs, models, or buckets. Integration tests can be added
later under tests/integration with real services.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---- env defaults BEFORE app imports ----
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "memory://")
os.environ.setdefault("KMS_BACKEND", "local")
os.environ.setdefault("MASTER_KEY", base64.b64encode(b"\x11" * 32).decode())
os.environ.setdefault("JWT_SECRET", "test-secret-" + "x" * 48)
os.environ.setdefault("CORS_ORIGINS", "http://localhost")
os.environ.setdefault("S3_ENDPOINT", "http://localhost:9000")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---- stub heavy modules so imports don't trigger them ----

class _StubDeepFace:
    @staticmethod
    def represent(img_path, **kwargs):
        # Return two consistent unit-norm vectors → high similarity
        return [{"embedding": [0.1] * 512, "facial_area": {"x": 0, "y": 0, "w": 100, "h": 100}}]


sys.modules.setdefault("deepface", MagicMock())
sys.modules["deepface"].DeepFace = _StubDeepFace
sys.modules.setdefault("mediapipe", MagicMock())
sys.modules.setdefault("onnxruntime", MagicMock())


@pytest.fixture(scope="session")
def settings():
    from app.core.config import get_settings
    return get_settings()


@pytest.fixture
def jwt_token(settings):
    from app.core.security import create_access_token
    return create_access_token("user-123", {"email": "test@biometrical.org"})


@pytest.fixture
def auth_headers(jwt_token):
    return {"Authorization": f"Bearer {jwt_token}"}
