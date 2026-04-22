# API Reference

Base URL: `http://localhost:8000`. All `/api/v1/verify/*` routes require a
Bearer JWT issued by your app (see `scripts/issue_dev_token.py` for a dev token).

## `GET /healthz`

Liveness probe. Returns `{"status": "ok", "version": "..."}`.

## `GET /readyz`

Readiness probe. Same shape as `/healthz`.

## `GET /api/v1/verify/challenge`

Returns an anti-replay challenge to embed in the next `submit`.

```json
{
  "challenge": "blink_twice",
  "instruction": "Mira a la cámara y parpadea dos veces lentamente.",
  "nonce": "lQ8...",
  "expires_at": "2026-04-21T15:05:00+00:00"
}
```

## `POST /api/v1/verify/submit`

Multipart form fields:

| Field | Type | Notes |
|-------|------|-------|
| `contract_id`   | text | Your contract identifier (≤64 chars). |
| `challenge`     | text | Echoed challenge name (`blink_twice` etc). |
| `nonce`         | text | Echoed nonce from `/challenge`. |
| `id_image`      | file | JPEG/PNG/WEBP, ≤8 MB. |
| `selfie_video`  | file | MP4/WEBM, ≤25 MB, ≥3s. |

Returns `202 Accepted`:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "sha256_id": "...",
  "sha256_video": "..."
}
```

## `GET /api/v1/verify/{job_id}`

Poll for outcome.

```json
{
  "job_id": "uuid",
  "status": "done",
  "decision": "APPROVE",
  "similarity": 0.84,
  "distance": 0.16,
  "threshold": 0.68,
  "liveness_score": 0.97,
  "challenge_passed": true,
  "deepfake_suspicious": false,
  "model": "ArcFace",
  "receipt_hash": "sha256-of-canonical-receipt",
  "receipt_signature": "base64-ed25519-sig",
  "reason": "Match above approve threshold"
}
```

`status` cycles `queued → running → done|error`. Decisions: `APPROVE`,
`REVIEW`, `REJECT`.

## Receipt verification

Receipts are Ed25519-signed canonical JSON. Verify offline with the public
key returned in `receipt.public_key` (or fetched once and pinned).
