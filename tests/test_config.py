"""Config validators — production refusal-to-boot + jwt_alg=none rejection."""
from __future__ import annotations

import importlib

import pytest


def _fresh_settings(**env):
    """Reload Settings under custom env without contaminating other tests."""
    import os

    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        from app.core import config as cfg_mod
        importlib.reload(cfg_mod)
        # Bypass the lru_cache so we get fresh validation
        return cfg_mod.Settings()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        from app.core import config as cfg_mod
        importlib.reload(cfg_mod)


def test_jwt_alg_none_rejected():
    """Settings refuses to construct with JWT_ALG='none' even in dev."""
    with pytest.raises(Exception) as ei:
        _fresh_settings(JWT_ALG="none")
    msg = str(ei.value).lower()
    assert "jwt_alg" in msg or "none" in msg


def test_jwt_alg_blank_rejected():
    with pytest.raises(Exception) as ei:
        _fresh_settings(JWT_ALG="")
    msg = str(ei.value).lower()
    assert "jwt_alg" in msg or "none" in msg or "empty" in msg


def test_jwt_alg_normal_value_ok():
    s = _fresh_settings(JWT_ALG="HS256")
    assert s.jwt_alg == "HS256"
