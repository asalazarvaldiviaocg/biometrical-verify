# Tier-3 server-side liveness — deploy steps

The Modal endpoint `verify_liveness` (added to `modal_app.py`) runs the
canonical OSS biometrical_liveness_engine on a sample of decoded video
frames, server-side. It complements the client-side blink + alignment
gate already shipped in `biometrical-contract/components/signing/selfie-capture.tsx`.

## What this catches

| Spoof modality      | Signal that fires                                      |
|---------------------|--------------------------------------------------------|
| Printed photo       | Low LBP entropy + skewed FFT toward over-smooth        |
| Phone screen replay | Moiré pattern in FFT high-band + low gradient variance |
| 3D silicone mask    | Skin density mismatch + HSV uniformity                 |
| Deepfake video      | Texture flatness + frame-to-frame entropy collapse     |

## Deploy

From the `biometrical-verify` folder:

```bash
modal deploy modal_app.py
```

Modal will print URLs for each `@modal.fastapi_endpoint` function — there
are now THREE:

```
✓ Created web function verify_face          => https://...--verify-face.modal.run
✓ Created web function verify_signature     => https://...--verify-signature.modal.run
✓ Created web function verify_liveness      => https://...--verify-liveness.modal.run     ← new
```

Copy the `verify_liveness` URL.

## Wire into biometrical-contract

Add to Netlify environment variables:

```
BIOMETRICAL_VERIFY_LIVENESS_URL = <verify_liveness URL from Modal>
```

The `BIOMETRICAL_VERIFY_TOKEN` already in use for face-match is reused —
both functions check the same `SHARED_SECRET` Modal secret.

After Netlify redeploys, the `/api/signing/[token]/selfie` route will:

1. Run client-side blink + alignment gate (unchanged)
2. Run Tier-2 face-match against the ID portrait (unchanged)
3. **NEW** — Run Tier-3 server-side liveness on the recorded video. If
   the verdict is `is_live=false`, the request is rejected with
   `reason: 'server_liveness_failed'` and the signer sees an
   anti-spoofing-specific error message.

## Soft-fail behavior

If `BIOMETRICAL_VERIFY_LIVENESS_URL` is unset, or Modal is unreachable,
or the endpoint returns a non-fatal reason (e.g. `no_face_in_video`,
`video_too_short`), the gate is **skipped** and audit-logged. This is
intentional: the client-side and Tier-2 gates already provide strong
defense, and a Modal outage shouldn't block legitimate signers.

## Verifying live

Look in the dashboard's Errors / AuditLog table after a real signing:

- `signing.liveness.server_skipped` — verdict was `skipped` (auditing only)
- `signing.liveness.server_failed`  — gate caught a spoof (signer rejected)
- `signing.face_match.passed` with `serverLivenessVerdict: 'passed'` and
  `serverLivenessScore: 0.74` — full pipeline cleared

## Rollback

If the new gate causes any issue, unset `BIOMETRICAL_VERIFY_LIVENESS_URL`
in Netlify env. The route will then soft-fail every liveness check and
fall through to the previous behavior. No code rollback needed.
