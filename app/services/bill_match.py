"""Utility-bill name match for guarantor reachability proof.

Pure verdict logic (no I/O): given sidecar OCR lines and the guarantor's
full name, decides verified vs manual_review.

`match_bill_name` verified on the surname plus one more token, because a
utility bill is a formal document with its own conventions and the surname is
the stable part of it.

`match_id_name` verified on any two tokens, because an identity card does not
follow the person's typing convention and the old surname requirement refused
real guarantors for a suffix, a hyphen, or the order the card printed. See its
docstring.

Everything else - OCR noise, abbreviated documents, married names, unreadable
images - is manual_review with a note naming which case it was, NEVER a
rejection: OCR misreads are routine and a machine must not strand a real
guarantor.
"""

from __future__ import annotations

import re


_NAME_TOKEN_MIN = 3
# Single-glyph tolerance starts at four characters, not five. KOFI, AMA, YAW,
# ABI and NANA are everyday given names here, and a tolerance that skipped them
# gave the one case this exists for - a one-glyph OCR misread - no tolerance at
# all on the names most likely to be short. At three it would collide (AMA vs
# AMP), which is why it stops here.
_GLYPH_TOLERANCE_MIN = 4
# The rule: two of the person's names on the card is the same human.
#
# It was "the surname matched, plus one more". That required the LAST token the
# person typed to appear on the card, which is a guess about naming convention
# rather than a fact about identity, and it turned away three ordinary shapes:
# a suffix ("Kwadwo Mensah Jnr"), a hyphenated surname, and a card that prints
# the surname first while the person typed it last. Each of those has two names
# on the card and was still refused, in front of an applicant who could do
# nothing about it.
_REQUIRED_MATCHES = 2


def _norm(s: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", " ", s.upper())
    return " ".join(cleaned.split())


def _lev1(a: str, b: str) -> bool:
    """True when a and b differ by at most one single-character edit.
    Bridges single-glyph OCR misreads (MENSAH vs MENSAR, KOFI vs KOF1) without
    opening the door to different names: only for tokens of length >= 4."""
    if a == b:
        return True
    if len(a) < _GLYPH_TOLERANCE_MIN or len(b) < _GLYPH_TOLERANCE_MIN:
        return False
    if abs(len(a) - len(b)) > 1:
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


def _matched_tokens(want: list[str], hay_tokens: list[str]) -> list[str]:
    """Which of the typed name tokens appear in the OCR text.

    Tolerant of a single misread glyph, because machine-printed card text has
    exactly one realistic failure mode and exact matching stranded real people
    on noise.
    """
    matched = []
    for tok in want:
        if any(tok == h or _lev1(tok, h) for h in hay_tokens if len(h) >= _NAME_TOKEN_MIN):
            matched.append(tok)
    return matched


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

    Verified when at least two of the typed name's tokens are on the card,
    each exact or one misread glyph off. Nothing else gates it: not which token
    is last, not whether the surname specifically appeared. Which token a person
    types last is a naming convention, not a fact about who they are, and
    requiring it refused real guarantors for using a suffix, a hyphen, or the
    order the card happened to print.

    Every other outcome is manual_review with a note that says which one it was.
    The distinction matters to the person reading it: "we could not read the
    card" means retake the photo, "the names did not match" means check the
    spelling, and reporting the second when the truth is the first sends people
    to fix something that was never wrong.

    NEVER a rejection. A guarantor is somebody vouching for a stranger, with
    nothing to gain and every reason to walk away; a machine that turns a misread
    into a refusal does not lose an applicant, it loses the vouch.
    """
    want = [t for t in _norm(full_name).split() if len(t) >= _NAME_TOKEN_MIN]
    if not want:
        return False, "guarantor name has no matchable tokens - admin review"
    hay = _norm("\n".join(ocr_lines)).split()
    if not hay:
        return False, "could not read the name on the ID - admin review"

    matched = _matched_tokens(want, hay)
    if len(matched) >= _REQUIRED_MATCHES:
        return True, ""

    if not matched:
        return False, "the name on the ID did not match - admin review"
    return (
        False,
        f"only {len(matched)} of {len(want)} names on the ID matched "
        f"({len(matched)} needed) - admin review",
    )
