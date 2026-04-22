# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-21

### Added

Initial open-source release by [biometrical.org](https://biometrical.org).

- FastAPI service with `/healthz`, `/readyz`, `/api/v1/verify/challenge`,
  `/api/v1/verify/submit`, `/api/v1/verify/{job_id}`.
- Celery worker pipeline:
  - ID quality assessment (blur, glare, face presence)
  - Best-frame selection from short selfie videos
  - Passive liveness via Silent-Face-Anti-Spoofing ONNX (graceful fallback to
    sharpness heuristic when the model is absent)
  - Active liveness via MediaPipe FaceMesh blink counter
  - Deepfake heuristics (FFT high-frequency energy + temporal flicker)
  - Face matching via DeepFace + ArcFace embeddings, cosine distance
  - Decision engine: APPROVE / REVIEW / REJECT thresholds
- Envelope encryption (AES-256-GCM) with pluggable KMS backend (`LocalKMS`,
  `AWSKMS`).
- Ed25519-signed receipts over canonical JSON for legal evidence.
- Postgres persistence + append-only audit log.
- MinIO / S3 object storage for encrypted blobs.
- React + Vite demo capture component using `getUserMedia` + `MediaRecorder`.
- Docker Compose stack (api + worker + db + redis + minio + bucket-init).
- Multi-stage Dockerfile, non-root container user.
- GitHub Actions CI: ruff, mypy, pytest, Docker build, frontend build.
- Pytest suite (10/10 passing): crypto roundtrip + tamper detection,
  receipt sign + verify, health probes, auth gating.
- Bootstrap script: generates fresh `.env` secrets, attempts model download
  with mirror fallback.
- Documentation: README, API.md, SECURITY.md, COMPLIANCE.md, CONTRIBUTING.md.

### Security

- AES-256-GCM with per-blob fresh DEK, KEK-wrapped via KMS.
- Crypto-shred path for GDPR Art. 17 right-to-erasure.
- Rate limiting per user (5/min default).
- File-size caps (8 MB ID, 25 MB video).
- MIME validation on uploads.
- JWT auth on all `/api/v1` endpoints.
- Server-issued nonce for anti-replay.
