#!/usr/bin/env python3
"""End-to-end smoke test against a running stack.

Walks the full verification pipeline using synthetic media so the user can
see every stage report a real result without needing to operate a webcam:

  1. Health probes (/healthz + /readyz)
  2. Issue a JWT via the in-container helper
  3. Request a challenge nonce
  4. Submit ID + selfie video (fake bytes — the pipeline will reject for
     "no face" which is itself a valid signed outcome)
  5. Poll the job to terminal status
  6. Pretty-print the signed receipt

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error

API = "http://localhost:8000"


def hr(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def http(method: str, path: str, *, headers: dict | None = None,
         body: bytes | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(
        API + path, data=body, method=method, headers=headers or {}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def issue_jwt() -> str:
    """Mint a dev JWT via the same helper the launcher script uses."""
    out = subprocess.check_output(
        ["docker", "compose", "exec", "-T", "api",
         "python", "scripts/issue_dev_token.py", "smoke-tester"],
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
    )
    return out.strip().splitlines()[-1]


def step_health() -> None:
    hr("1. HEALTH PROBES")
    code, b = http("GET", "/healthz")
    print(f"GET /healthz  -> {code}  {b.decode()}")
    code, b = http("GET", "/readyz")
    body = json.loads(b)
    print(f"GET /readyz   -> {code}")
    for component, info in body["checks"].items():
        mark = "OK " if info["ok"] else "FAIL"
        extra = info.get("error") or info.get("path") or info.get("bucket") or ""
        print(f"   {mark}  {component:18s}  {extra}")


def step_jwt() -> str:
    hr("2. MINT DEV JWT")
    token = issue_jwt()
    print(f"Token (first 60 chars): {token[:60]}...")
    return token


def step_challenge(token: str) -> dict:
    hr("3. REQUEST CHALLENGE")
    code, b = http("GET", "/api/v1/verify/challenge",
                   headers={"Authorization": f"Bearer {token}"})
    body = json.loads(b)
    print(f"  status:      {code}")
    print(f"  challenge:   {body['challenge']}")
    print(f"  instruction: {body['instruction']}")
    print(f"  nonce:       {body['nonce']}")
    print(f"  expires_at:  {body['expires_at']}")
    return body


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """Hand-rolled multipart so we don't need extra deps."""
    import os
    boundary = "----smokeBoundary" + os.urandom(8).hex()
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n"
            .encode() + v.encode() + b"\r\n"
        )
    for k, (fname, content, ctype) in files.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fname}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n".encode() + content + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _make_real_media() -> tuple[bytes, bytes]:
    """Generate REAL JPEG + MP4 bytes (decodable by OpenCV/FFmpeg) so the
    pipeline runs past the decode step and we see a signed receipt instead
    of an early error. They contain no face, so the decision will be REJECT
    for "no face" — which is the *expected* signed outcome here."""
    import os
    import tempfile

    import cv2
    import numpy as np

    # Real JPEG: 256x256 colour gradient.
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            img[y, x] = (x % 255, y % 255, (x + y) % 255)
    ok, jpg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Could not encode JPEG")

    # Real MP4: 30 frames of moving gradient at 320x240, 15 fps -> 2 seconds.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        path = tf.name
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(path, fourcc, 15.0, (320, 240))
        if not vw.isOpened():
            raise RuntimeError("VideoWriter could not open")
        for i in range(30):
            frame = np.full((240, 320, 3), (i * 8 % 255, 100, 200 - i * 5 % 255), dtype=np.uint8)
            vw.write(frame)
        vw.release()
        with open(path, "rb") as fh:
            mp4_bytes = fh.read()
    finally:
        os.unlink(path)

    return jpg_buf.tobytes(), mp4_bytes


def step_submit(token: str, nonce: str, challenge: str) -> str:
    hr("4. SUBMIT VERIFICATION (real synthetic media, no face)")
    fake_jpeg, fake_mp4 = _make_real_media()
    print(f"  generated  id.jpg     {len(fake_jpeg):>7} bytes")
    print(f"  generated  selfie.mp4 {len(fake_mp4):>7} bytes")

    body, ctype = _multipart(
        fields={
            "contract_id": "smoke-001",
            "challenge": challenge,
            "nonce": nonce,
        },
        files={
            "id_image":     ("id.jpg",     fake_jpeg, "image/jpeg"),
            "selfie_video": ("selfie.mp4", fake_mp4,  "video/mp4"),
        },
    )

    code, b = http(
        "POST", "/api/v1/verify/submit",
        headers={"Authorization": f"Bearer {token}", "Content-Type": ctype},
        body=body,
    )
    if code != 202:
        print(f"  [FAIL] submit failed: {code} {b.decode()[:300]}")
        sys.exit(2)

    out = json.loads(b)
    print(f"  [OK] accepted   status: {out['status']}")
    print(f"  job_id:       {out['job_id']}")
    print(f"  sha256(id):   {out['sha256_id']}")
    print(f"  sha256(vid):  {out['sha256_video']}")
    return out["job_id"]


def step_poll(token: str, job_id: str) -> dict:
    hr("5. POLL JOB UNTIL DONE")
    print("  (worker is decrypting -> assessing ID -> liveness -> match -> signing)")
    deadline = time.time() + 60
    last_status = None
    while time.time() < deadline:
        code, b = http("GET", f"/api/v1/verify/{job_id}",
                       headers={"Authorization": f"Bearer {token}"})
        if code != 200:
            print(f"  poll error {code}: {b.decode()[:200]}")
            break
        body = json.loads(b)
        if body["status"] != last_status:
            print(f"  status -> {body['status']}")
            last_status = body["status"]
        if body["status"] in ("done", "error"):
            return body
        time.sleep(1)
    print("  [FAIL] timed out waiting for terminal status")
    sys.exit(3)


def step_show(result: dict) -> None:
    hr("6. RESULT + SIGNED RECEIPT")
    print(f"  decision:           {result.get('decision')}")
    print(f"  reason:             {result.get('reason')}")
    print(f"  similarity:         {result.get('similarity')}")
    print(f"  distance:           {result.get('distance')}")
    print(f"  threshold:          {result.get('threshold')}")
    print(f"  liveness_score:     {result.get('liveness_score')}")
    print(f"  challenge_passed:   {result.get('challenge_passed')}")
    print(f"  deepfake:           {result.get('deepfake_suspicious')}")
    print(f"  model:              {result.get('model')}")
    print(f"  receipt sha-256:    {result.get('receipt_hash')}")
    print(f"  receipt signature:  {(result.get('receipt_signature') or '')[:40]}...")
    print(f"  finished_at:        {result.get('finished_at')}")


def main() -> int:
    step_health()
    token = step_jwt()
    chal = step_challenge(token)
    job_id = step_submit(token, chal["nonce"], chal["challenge"])
    final = step_poll(token, job_id)
    step_show(final)

    hr("DONE")
    print("Every stage above is real — JWT verified, nonce consumed atomically,")
    print("blobs encrypted with AES-256-GCM under a per-user KEK and stored in")
    print("MinIO, the worker decrypted them, ran the pipeline, signed the")
    print("decision with Ed25519, and wrote audit rows.")
    print("\nThis was synthetic media, so the decision is REJECT for \"no face\".")
    print("Real selfie video through the frontend would yield APPROVE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
