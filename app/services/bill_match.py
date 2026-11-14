"""Utility-bill name match for guarantor reachability proof.

Pure verdict logic (no I/O): given sidecar OCR lines and the guarantor's
full name, decides verified vs manual_review.

`match_bill_name` verified on the surname plus one more token, because a
utility bill is a formal document with its own conventions and the surname is
the stable part of it.

`grade_id_name` verified on any two tokens, because an identity card does not
follow the person's typing convention and the old surname requirement refused
real guarantors for a suffix, a hyphen, or the order the card printed. See its
docstring.

Everything else - OCR noise, abbreviated documents, married names, unreadable
images - is manual_review with a note naming which case it was, NEVER a
rejection: OCR misreads are routine and a machine must not strand a real
guarantor.

The one thing the caller needs on top of that is the difference between a name
that is plainly NOT on the document and a name the machine could not read well
enough to judge. Those are facts about different things - the document and the
photograph - and they have different consequences: the first is something the
person can act on by correcting what they typed, the second is something only a
person can decide. `grade_id_name` grades the name three ways so the caller
never has to infer that difference from a count.
"""

from __future__ import annotations

import re


# The three grades a name can come back with.
#
# verified - the card carries the person's name.
# mismatch - the card was read and carries a DIFFERENT name.
# unclear  - the card could not be read well enough to say either way.
#
# Only `mismatch` is a fact about the document, so only `mismatch` may refuse
# anybody. `unclear` is a fact about a photograph.
NAME_VERIFIED = "verified"
NAME_MISMATCH = "mismatch"
NAME_UNCLEAR = "unclear"

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
# A token shorter than this is never judged absent.
#
# ABI, NANA, AMA and JNR are ordinary given names and suffixes. A short token
# that is not on the card is far more likely to be a suffix the person typed
# than a false name, and refusing on it turned away people who were carrying
# their own card. Long enough to mean something, or it goes to a person.
_ABSENCE_MIN_LEN = 4
# How many edits a card token may be from a typed name and still be that name,
# mangled by the OCR rather than replaced by a different one.
#
# One edit is a misread. Two is a transposition or a doubled stroke. Three is
# where a name stops being the same name: past it, the card does not carry
# what the person typed, and that is a fact about the document.
_NEAR_MISS_MAX = 3


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


def _lev_n(a: str, b: str, limit: int) -> int:
    """Edit distance between a and b, capped at limit + 1.

    Anything further apart than limit is reported as limit + 1, because the
    only question asked of it is whether the two are within limit of each
    other. The band keeps the cost proportional to the limit rather than to
    the length of the names.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(v)
            row_min = min(row_min, v)
        if row_min > limit:
            return limit + 1
        prev = cur
    return prev[-1]


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


def _near_miss(tok: str, hay_tokens: list[str]) -> bool:
    """Is this typed name on the card, mangled by the OCR rather than replaced?

    A near miss means the card probably carries the name and the machine read
    it badly - a person's call, never a refusal. No near miss means the card
    does not carry it, which is a fact about the document.
    """
    for h in hay_tokens:
        if len(h) < _ABSENCE_MIN_LEN:
            continue
        if _lev_n(tok, h, _NEAR_MISS_MAX) <= _NEAR_MISS_MAX:
            return True
    return False


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


def grade_id_name(ocr_lines: list[str], full_name: str) -> tuple[str, str]:
    """Grade a typed name against an ID card: verified, mismatch or unclear.

    Verified when at least two of the typed name's tokens are on the card,
    each exact or one misread glyph off. Nothing else gates it: not which token
    is last, not whether the surname specifically appeared. Which token a person
    types last is a naming convention, not a fact about who they are, and
    requiring it refused real people for using a suffix, a hyphen, or the order
    the card happened to print.

    The grade below that line is the one the caller needs:

    mismatch - the card was read, and it does not carry the name that was
        typed. One of the typed names is not on the card at all, and it is not
        there misread either - it is simply absent. This is a fact about the
        document, and the person can act on it by typing the name their card
        actually carries.

    unclear - the card could not be read well enough to say. The name may be
        there and misread (a near miss), it may be shorter than the bar for
        judging absence, it may be a suffix the person typed, or nothing may
        have been read at all. None of those is a fact about the document, and
    a person decides those.

    The distinction is the whole point of this function. A count cannot carry
    it: "one of two names matched" is the same number whether the second name
    was misread or is somebody else's, and a caller reading only that number
    has to guess - and guessing wrong means either refusing a person whose
    card is in their hand, or accepting a card that belongs to somebody else.
    """
    want = [t for t in _norm(full_name).split() if len(t) >= _NAME_TOKEN_MIN]
    if not want:
        return NAME_UNCLEAR, "guarantor name has no matchable tokens - admin review"
    hay = _norm("\n".join(ocr_lines)).split()
    if not hay:
        return NAME_UNCLEAR, "could not read the name on the ID - admin review"

    matched = _matched_tokens(want, hay)
    if len(matched) >= _REQUIRED_MATCHES:
        return NAME_VERIFIED, ""

    missing = [t for t in want if t not in matched]
    if not missing:
        # Every typed name is on the card, but there are fewer than two of them
        # to compare: a one-token name cannot reach the bar. That is not a
        # mismatch, it is a name this check cannot grade.
        return NAME_UNCLEAR, (
            f"only {len(matched)} of {len(want)} names on the ID matched "
            f"({_REQUIRED_MATCHES} needed) - admin review"
        )

    # Some of the names are on the card and some are not. The question that
    # decides this case is whether the missing name is there but misread, or is
    # simply not on the card.
    absent = [
        t for t in missing
        if len(t) >= _ABSENCE_MIN_LEN and not _near_miss(t, hay)
    ]
    if absent:
        return NAME_MISMATCH, "the name on the ID did not match - admin review"
    return NAME_UNCLEAR, (
        f"only {len(matched)} of {len(want)} names on the ID matched "
        f"({_REQUIRED_MATCHES} needed) - admin review"
    )


def match_id_name(ocr_lines: list[str], full_name: str) -> tuple[bool, str]:
    """Typed-name vs ID-printed-name check, as a yes/no.

    Kept for callers that only need "is this verified". The grade is
    `grade_id_name`'s, and the notes are the ones it produces, so a caller
    reading this still learns whether the name was wrong or merely unread.
    """
    status, note = grade_id_name(ocr_lines, full_name)
    return status == NAME_VERIFIED, note
