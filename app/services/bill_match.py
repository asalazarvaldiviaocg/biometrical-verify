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
