# Security Architecture

## Threat model

| Threat | Mitigation |
|--------|------------|
| Stolen photo of victim used as "selfie" | Active liveness challenge (blink/turn) + passive anti-spoof CNN |
| Replay of valid prior video | Server-issued nonce in `/challenge`, echoed in `submit` and bound to the receipt |
| Deepfake / face-swap | FFT high-frequency band check + temporal flicker score; commercial layer recommended for high-value flows |
| ID tampering | Document quality gate (blur/glare/face presence); MRZ/OCR plug-in recommended |
| MITM during upload | TLS 1.3, HSTS, optional cert pinning on mobile |
| Compromised storage | AES-256-GCM with per-blob DEK wrapped under KMS-managed KEK |
| Insider exfil of biometrics | Operators see decisions only — raw blobs require KMS access; crypto-shred on user erasure |
| Replay via reused JWT | Short JWT TTL + per-user rate limit; revoke list optional |
| Brute force submissions | Rate limit (5/min/user by default), file-size caps |

## Data flow

1. Client captures ID + 4-5 s selfie video.
2. Browser computes SHA-256 of the video for tamper evidence.
3. POST → API gateway. Both blobs are sealed in memory with AES-256-GCM
   using a fresh DEK; the DEK is wrapped under the KEK (local in dev,
   AWS KMS in prod) and stored beside the ciphertext in S3.
4. A Celery worker decrypts in memory, runs the pipeline, and never writes
   plaintext to disk except a short-lived video tempfile (unlinked in
   `finally:`).
5. Decision + signed receipt persist to Postgres. Raw blobs auto-purge
   after `BLOB_RETENTION_DAYS`.

## Crypto-shredding

Right-to-erasure: deleting a user's KEK from KMS renders every wrapped
DEK unrecoverable. Storage records may remain for audit purposes but are
plaintext-equivalent to random bytes.

## Receipt signing

Decisions are signed with Ed25519 over canonical-JSON of the payload
(`job_id`, `contract_id`, `decision`, `similarity`, `model`, hashes).
Verifiable offline using the public key from the receipt envelope.

## Operational hardening

- Run API + workers as non-root (Dockerfile sets UID 1000).
- Minimal image (python:3.11-slim + libgl/glib only).
- HTTP middleware: CORS allow-list, no wildcard.
- Add CSP, X-Content-Type-Options, Referrer-Policy: no-referrer at the
  edge (NGINX / Cloudflare).
- Mount `app/` and `models/` read-only in the worker container.
- Never log image bytes, base64 payloads, embeddings, or DEKs.

## Known gaps in the open-source baseline

- No commercial deepfake classifier — heuristics only.
- No ID OCR / MRZ parser by default — plug in PaddleOCR or a vendor.
- `LocalKMS` is not for production — switch `KMS_BACKEND=aws`.
- Rate limiter is in-memory — replace with Redis for HA.
