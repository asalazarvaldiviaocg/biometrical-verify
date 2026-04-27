"""Minimal magic-byte sniffer for image and video uploads.

We only accept JPEG, PNG, WEBP for IDs and MP4, WEBM for selfie videos. The
`Content-Type` header is attacker-controlled; relying on it lets a bad client
upload arbitrary bytes labelled as an image. Sniffing the leading bytes
forces the file to actually be what it claims.

Returns the detected MIME, or None if nothing matched.
"""

from __future__ import annotations


def sniff(blob: bytes) -> str | None:
    if len(blob) < 12:
        return None
    h = blob[:16]

    # JPEG: FF D8 FF
    if h[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if h[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # WEBP: RIFF....WEBP
    if h[:4] == b"RIFF" and h[8:12] == b"WEBP":
        return "image/webp"
    # MP4 / ISO base media: ....ftyp
    if h[4:8] == b"ftyp":
        major = h[8:12]
        if major in (b"isom", b"iso2", b"avc1", b"mp41", b"mp42", b"M4V ", b"mp4v"):
            return "video/mp4"
        # Some QuickTime brands map to mp4 well enough for this gate.
        if major == b"qt  ":
            return "video/mp4"
    # WEBM / Matroska: 1A 45 DF A3
    if h[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    return None
