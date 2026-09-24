"""Utility-bill name match for guarantor reachability proof.

Pure verdict logic (no I/O): given sidecar OCR lines and the guarantor's
full name, decides verified vs manual_review. Verified requires the surname
(last name token) plus at least one more name token (len>=3) present in the
bill text. Everything else — OCR noise, abbreviated bills, married names —
is manual_review with a note, NEVER a rejection: OCR misreads are routine
and a machine must not strand a real guarantor.
"""

from __future__ import annotations

import re


def _norm(s: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", " ", s.upper())
    return " ".join(cleaned.split())


def _lev1(a: str, b: str) -> bool:
    """True when a and b differ by at most one single-character edit.
    Bridges single-glyph OCR misreads (MENSAH vs MENSAR) without opening
    the door to different names: only for tokens of length >= 5."""
    if a == b:
        return True
    if len(a) < 5 or len(b) < 5 or abs(len(a) - len(b)) > 1:
        return False
    # Classic one-row DP, early-exits above distance 1.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(v)
            row_min = min(row_min, v)
        if row_min > 1:
            return False
        prev = cur
    return prev[-1] <= 1


def match_bill_name(bill_lines: list[str], full_name: str) -> tuple[bool, str]:
    want = [t for t in _norm(full_name).split() if len(t) >= 3]
    if not want:
        return False, "guarantor name has no matchable tokens - admin review"
    haystack = " " + _norm("\n".join(bill_lines)) + " "
    hits = 0
    surname_hit = False
    for i, tok in enumerate(want):
        if f" {tok} " in haystack:
            hits += 1
            if i == len(want) - 1:
                surname_hit = True
    if surname_hit and hits >= 2:
        return True, ""
    if hits == 0:
        return False, "no guarantor name tokens on the bill - admin review"
    return False, "partial name match on the bill - admin review"


def match_id_name(ocr_lines: list[str], full_name: str) -> tuple[bool, str]:
    """Typed-name vs ID-printed-name check for guarantor acceptance.

    Same token rule as the bill match, but tokens may differ by one
    single-character edit (length >= 5): machine-printed MRZ/front text is
    structurally known, so the only failure mode is an OCR glyph misread,
    and exact matching would strand real people on noise. Anything weaker
    than surname + one more token still fails closed to manual review.
    """
    want = [t for t in _norm(full_name).split() if len(t) >= 3]
    if not want:
        return False, "guarantor name has no matchable tokens - admin review"
    hay = _norm("\n".join(ocr_lines)).split()
    if not hay:
        return False, "no readable text on the ID - admin review"

    def present(tok: str) -> bool:
        return any(tok == h or _lev1(tok, h) for h in hay if len(h) >= 3)

    hits = sum(present(t) for t in want)
    surname_hit = present(want[-1])
    if surname_hit and hits >= 2:
        return True, ""
    if hits == 0:
        return False, "name does not match the name on the ID - admin review"
    return False, "partial name match on the ID - admin review"
