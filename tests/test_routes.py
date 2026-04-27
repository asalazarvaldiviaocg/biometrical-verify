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


def test_readyz_reports_per_component():
    """In CI Redis/MinIO are absent, so /readyz returns 503 with a per-
    component breakdown. The point is that the probe actually exercises
    deps now — the previous version always returned 200 even when broken."""
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code in (200, 503)
    body = r.json()
    assert "checks" in body
    assert {"db", "redis", "object_store", "anti_spoof_model"}.issubset(body["checks"].keys())


def test_keys_endpoint_returns_active_pubkey():
    with TestClient(app) as c:
        r = c.get("/api/v1/keys")
    assert r.status_code == 200
    body = r.json()
    assert body["active"]["alg"] == "Ed25519"
    assert body["active"]["public_key_b64"]
    assert body["active"]["key_id"]


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


def test_submit_rejects_unknown_nonce(auth_headers):
    """The /submit handler must reject any nonce that wasn't issued by /challenge."""
    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 64
    fake_mp4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64

    with TestClient(app) as c:
        r = c.post(
            "/api/v1/verify/submit",
            headers=auth_headers,
            data={
                "contract_id": "ctr-001",
                "challenge": "blink_twice",
                "nonce": "this-was-never-issued",
            },
            files={
                "id_image": ("id.jpg", fake_jpeg, "image/jpeg"),
                "selfie_video": ("v.mp4", fake_mp4, "video/mp4"),
            },
        )
    assert r.status_code == 400
    assert "Invalid nonce" in r.json()["detail"]


def test_submit_rejects_mime_lie(auth_headers):
    """Bytes that are clearly not an image must be refused even if header lies."""
    # Get a valid nonce first.
    with TestClient(app) as c:
        chal = c.get("/api/v1/verify/challenge", headers=auth_headers).json()
        r = c.post(
            "/api/v1/verify/submit",
            headers=auth_headers,
            data={
                "contract_id": "ctr-002",
                "challenge": chal["challenge"],
                "nonce": chal["nonce"],
            },
            files={
                "id_image": ("id.jpg", b"this is plain text", "image/jpeg"),
                "selfie_video": ("v.mp4", b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64, "video/mp4"),
            },
        )
    assert r.status_code == 415


def test_jwt_requires_audience():
    """A token without an audience claim is rejected."""
    import jwt as pyjwt

    from app.core.config import get_settings

    s = get_settings()
    token = pyjwt.encode(
        {"sub": "user-x", "exp": 9999999999, "iat": 1, "iss": s.jwt_issuer},
        s.jwt_secret,
        algorithm=s.jwt_alg,
    )
    with TestClient(app) as c:
        r = c.get("/api/v1/verify/challenge", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_submit_happy_path_enqueues_and_persists(auth_headers, monkeypatch):
    """Full /submit happy path: nonce issued → bytes uploaded → row persisted →
    Celery task enqueued. Stubs the S3 client + Celery .delay() since both
    require external services CI doesn't have. Locks in the schema fix that
    lets Verification rows live in SQLite."""
    from unittest.mock import MagicMock

    fake_s3 = MagicMock()
    monkeypatch.setattr("app.services.storage._client", lambda: fake_s3)

    enqueue_calls: list[str] = []

    def fake_delay(*, job_id):
        enqueue_calls.append(job_id)
        return MagicMock(id=job_id)

    import app.workers.tasks as tasks_mod
    monkeypatch.setattr(tasks_mod.verify_identity_task, "delay", fake_delay)

    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 256
    fake_mp4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 256

    with TestClient(app) as c:
        chal = c.get("/api/v1/verify/challenge", headers=auth_headers).json()
        r = c.post(
            "/api/v1/verify/submit",
            headers=auth_headers,
            data={
                "contract_id": "ctr-happy",
                "challenge": chal["challenge"],
                "nonce": chal["nonce"],
            },
            files={
                "id_image": ("id.jpg", fake_jpeg, "image/jpeg"),
                "selfie_video": ("v.mp4", fake_mp4, "video/mp4"),
            },
        )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert body["job_id"]
    assert enqueue_calls == [body["job_id"]]

    from app.db.session import SessionLocal
    from app.models.verification import AuditEvent, Verification

    db = SessionLocal()
    try:
        rec = db.get(Verification, body["job_id"])
        assert rec is not None
        assert rec.contract_id == "ctr-happy"
        assert rec.status == "queued"
        assert rec.nonce == chal["nonce"]
        audit = db.query(AuditEvent).filter_by(
            verification_id=body["job_id"], event="submitted"
        ).first()
        assert audit is not None
    finally:
        db.close()


def test_jwt_rejects_unsafe_subject():
    """A subject that would traverse S3 prefixes must be refused at the token layer."""
    import jwt as pyjwt

    from app.core.config import get_settings

    s = get_settings()
    token = pyjwt.encode(
        {
            "sub": "../tenantB/job",
            "iss": s.jwt_issuer,
            "aud": s.jwt_audience,
            "iat": 1,
            "exp": 9999999999,
        },
        s.jwt_secret,
        algorithm=s.jwt_alg,
    )
    with TestClient(app) as c:
        r = c.get("/api/v1/verify/challenge", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401
