# Master KEK rotation runbook

The service uses three keys, each with its own rotation story:

| Key | What it wraps | Where it lives | Rotation cadence |
|-----|---------------|----------------|------------------|
| **Master KEK** | Per-user KEKs | `MASTER_KEY` env (local) or AWS KMS CMK | Annual or on suspected compromise |
| **Per-user KEK** | Per-blob DEKs | `user_keks.wrapped_kek` (DB row, wrapped under master) | On compromise — automatic via crypto-shred |
| **Receipt signing key** | Receipt signatures | `RECEIPT_SIGNING_KEY` env (Ed25519, base64 32B) | On suspected compromise (invalidates older receipts) |

This document covers **master KEK rotation** because it is the only one
that's non-obvious and that touches every wrapped DEK in the system. The
other two keys are isolated changes (revoke a row / update an env var).

---

## Why we don't auto-rotate the master

Each `user_keks.wrapped_kek` row carries a `kms_key_id` column that pins
the master CMK that wrapped it. When the master rotates we have two
options:

1. **Lazy rotation** — keep the old CMK active for unwrap-only, write all
   *new* per-user KEKs under the new CMK. Old rows quietly age out as
   their users do new verifications (which create fresh DEKs anyway, but
   not fresh KEKs — so old rows live until that user is erased).
2. **Eager rotation** — re-wrap every existing row under the new CMK.
   Single sweep, takes ~1ms per row. After completion the old CMK is
   purely cosmetic and can be scheduled for deletion.

Eager is the right call when:

- The old master is *known* compromised (lazy keeps it useful).
- Compliance requires "all data under current key" within a window.
- Row count is moderate (tens of millions tops; beyond that, batch).

Lazy is fine for routine annual rotation.

---

## Eager rotation procedure (recommended for compromise)

### 1. Pre-flight

```bash
# Inventory of current wrapping
psql $DATABASE_URL -c "
  SELECT kms_key_id, count(*)
  FROM   user_keks
  WHERE  revoked_at IS NULL
  GROUP  BY kms_key_id;
"
```

Confirm:
- Backup of `user_keks` exists (it's the *only* reference for unwrapping).
- New master CMK has `Encrypt`/`Decrypt` IAM permission for the API role.
- A maintenance window of ~10 minutes per million rows.

### 2. Provision the new master

**AWS KMS:** create a new CMK. Note its ARN.

**Local KMS (dev only — don't run rotation in dev seriously):**

```bash
python -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
```

### 3. Run the rotation

A reference script lives at `scripts/rotate_master_kek.py` (see below).
It:

1. Loads the OLD CMK via `kms_backend.unwrap`.
2. Loads the NEW CMK via a separately-instantiated backend.
3. For each `user_keks` row with `revoked_at IS NULL`:
   - Unwrap with old.
   - Wrap with new.
   - `UPDATE user_keks SET wrapped_kek = $new_wrap, kms_key_id = $new_id
      WHERE user_id = $uid AND kms_key_id = $old_id;`
4. Refuses to touch rows already on the new key id (idempotent).

```bash
python scripts/rotate_master_kek.py \
    --old-key-id arn:aws:kms:us-east-1:.../old \
    --new-key-id arn:aws:kms:us-east-1:.../new \
    --batch 1000
```

### 4. Switch the runtime

After the script reports `kms_key_id` distribution = 100 % new:

1. Update `AWS_KMS_KEY_ID` (or `MASTER_KEY` for local) in your secret store.
2. Roll the API + worker pods. They re-read `_kms` lazily on first call.
3. Schedule the OLD CMK for deletion in AWS KMS (≥30 day grace
   recommended — gives time to discover any missed rotations).

### 5. Verify

```bash
psql $DATABASE_URL -c "
  SELECT kms_key_id, count(*) FROM user_keks GROUP BY kms_key_id;
"
# Should show only the new key id for non-revoked rows.

# Smoke test a verification end-to-end against a test user. The pipeline
# unwraps a per-user KEK to read past blobs — if rotation worked, this
# succeeds; if it didn't, you'll see UserKekRevoked or KMS errors in the
# worker log.
```

---

## Lazy rotation (routine annual rotate)

1. Provision new CMK.
2. Update `AWS_KMS_KEY_ID` in env.
3. Keep IAM permission on the OLD CMK (decrypt only) so legacy rows still
   open. The wrap path always uses the current `_kms` instance which now
   points at the new CMK; unwrap uses the `kms_key_id` recorded on each
   row, so it transparently selects the right CMK.

Drawback: until every user is erased or you run an eager pass, the old
CMK must stay decryptable. If that key was compromised, eager rotation
is the only safe path.

---

## Recovery from a botched rotation

Failure mode: script crashed midway, half the rows are on new, half on
old. The system keeps working because each row knows which key wrapped
it — `unwrap` looks at `kms_key_id` per row.

To complete:

```bash
python scripts/rotate_master_kek.py \
    --old-key-id <old> \
    --new-key-id <new> \
    --batch 1000 \
    --resume    # Skips rows already on new key id
```

If you've already deleted the old CMK and have rows still under it: those
rows are dead. The user_keks for them are unrecoverable, which means
their blobs are too — operationally equivalent to crypto-shred. Mark the
rows revoked and let the sweeper clean up.

```sql
UPDATE user_keks
   SET revoked_at = now()
 WHERE kms_key_id = '<orphaned-old-key-id>';
```

---

## Monitoring during/after rotation

Watch:

- API error rate on `verify_identity_task` — `UserKekRevoked` spikes
  indicate rows that lost access to their wrapping key.
- KMS API metrics (AWS CloudWatch) — `Decrypt` calls against the old key
  should drop to ~zero a few hours post-switch.
- Audit table `admin_audit_events` should have `action='kek.rotate.started'`
  + `action='kek.rotate.complete'` rows from the rotation script.

A rotation that goes well is *invisible*: no failed verifications, no
elevated errors, latency unchanged.
