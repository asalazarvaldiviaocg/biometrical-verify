from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_healthz():
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_readyz():
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code == 200


def test_challenge_requires_auth():
    with TestClient(app) as c:
        r = c.get("/api/v1/verify/challenge")
    assert r.status_code == 401


def test_challenge_with_auth(auth_headers):
    with TestClient(app) as c:
        r = c.get("/api/v1/verify/challenge", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["challenge"] in {"blink_twice", "blink_once", "turn_head", "none"}
    assert body["nonce"]
    assert body["instruction"]
