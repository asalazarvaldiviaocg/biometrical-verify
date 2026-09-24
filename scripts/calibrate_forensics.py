"""Forensics calibration harness: AI-generated vs real ID captures.

Runs doc_forensics.analyse (+ optional sidecar garble screen) over labelled
samples and prints a metrics table with group separation. Groups are derived
from the label prefix before ':' (real:, ai:, junk:).

Usage (on the KYC box, images in /tmp):
  python scripts/calibrate_forensics.py \
    --sample real:front=/tmp/real_front.jpg \
    --sample ai:mockup=/tmp/ai_mockup.png \
    --sidecar http://10.138.0.3:8001

Thresholds live in app/services/doc_forensics.py; change them only on the
basis of this table, never by gut feel. Spectral metrics are recorded but
unthresholded until a sample set shows genuine separation.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from app.services import doc_forensics as F


def ocr_lines(sidecar: str | None, path: str) -> list[str] | None:
    if not sidecar:
        return None
    try:
        import io
        import mimetypes

        with open(path, "rb") as f:
            data = f.read()
        boundary = "CALIBBOUNDARY"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="img.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            sidecar.rstrip("/") + "/text", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read().decode())
        lines = out.get("text_lines")
        return lines if isinstance(lines, list) else None
    except Exception as e:
        print(f"  [sidecar skip for {path}: {e}]")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="append", default=[],
                    help="GROUP:name=path (repeatable)")
    ap.add_argument("--sidecar", default=None)
    args = ap.parse_args()

    rows = []
    for spec in args.sample:
        group, rest = spec.split(":", 1)
        name, path = rest.split("=", 1)
        with open(path, "rb") as f:
            rep = F.analyse(f.read())
        lines = ocr_lines(args.sidecar, path)
        garble = None
        if lines is not None:
            flagged, ratio, n = F.is_garbled([str(l) for l in lines])
            garble = {"flagged": flagged, "ratio": round(ratio, 3), "tokens": n}
        rows.append({"group": group, "name": name, "rep": rep.to_public(), "garble": garble})
        g = f" garble={garble}" if garble else ""
        print(f"{group:5} {name:10} peaks={rep.to_public()['text_rows']:3} "
              f"card={rep.card_found!s:5} aspect={rep.card_aspect:.2f} "
              f"face={rep.face_in_card!s:5} hf={rep.spectral_hf:.3f} "
              f"noise={rep.noise_residual:6.2f} flags={len(rep.reasons)}{g}")

    print("\n-- group ranges (min..max) --")
    for metric in ("text_rows", "spectral_hf", "noise_residual"):
        parts = []
        for group in sorted({r["group"] for r in rows}):
            vals = [r["rep"][metric] for r in rows if r["group"] == group]
            parts.append(f"{group}=[{min(vals):.3f}..{max(vals):.3f}]")
        print(f"{metric:15} " + "  ".join(parts))
    print(f"current thresholds: peaks>={F.MIN_TEXT_PEAKS} "
          f"garble_wordlike>={F.GARBLE_WORDLIKE_MIN} (tokens>={F.GARBLE_MIN_TOKENS}) "
          f"phash_dup_dist<=6")


if __name__ == "__main__":
    main()
