"""The guarantor-card endpoint, exercised through HTTP.

Why this file exists
--------------------
The reported symptom was not a bad verdict. It was "every guarantor submission
returns 422", for every guarantor, every time - and it took a production log to
find, because the only tests for this decision were unit tests of a pure
function called directly with OCR lines someone made up.

The bug lived between that function and the applicant. `match_id_name` refused
anything without a surname match, and every non-verified verdict reached
Laundex as "name does not match", which Laundex turned into a hard rejection.
So a card we could not read, a card with one glyph misread, and a card belonging
to somebody else all presented to the applicant identically: you are lying
about your name.

These tests go through the real route - real multipart parsing, real quality
gate, real sidecar call - with only two boundaries stubbed: what the OCR sidecar
says, and what the quality gate decided. Everything between them is the code
that was wrong.

The sidecar is stubbed at its own client, not faked at the HTTP layer, because
its output is the input under test. What matters is not that an OCR engine reads
a card; it is that whatever the engine says becomes the right verdict for the
right reason, and is reported honestly when it is not.
"""

from __future__ import annotations

import io
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _png(width: int = 64, height: int = 64) -> bytes:
    """A real, decodable PNG.

    `_check_mime` sniffs the bytes, so a fake payload is refused at the door with
    a 415 and never reaches the logic under test - these tests would then pass
    for the wrong reason, which is the failure this whole exercise is about.
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes([200, 200, 200] * width) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def acceptable_card(monkeypatch):
    """The image-quality gate says the photos are usable.

    Left real, it runs OpenCV face detection over a grey square and rejects it,
    which is a different test - the quality gate has its own coverage. Folding it
    in here means a 422 from the wrong gate reads as a name bug, which is
    precisely the confusion this file exists to remove.
    """
    from app.services import id_parser

    class _Report:
        acceptable = True
        reason = ""
        reasons: list = []
        notes: list = []

    monkeypatch.setattr(id_parser, "assess", lambda _b: _Report())


@pytest.fixture
def ocr_says(monkeypatch):
    """Feed the route whatever the sidecar 'read', per endpoint."""
    import httpx

    state = {"front": [], "back": []}

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, files=None):
            if url.endswith("/text"):
                return _Resp({"text_lines": state["front"]})
            return _Resp({"mrz_lines": state["back"]})

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    def _set(front=None, back=None):
        if front is not None:
            state["front"] = front
        if back is not None:
            state["back"] = back

    return _set


def post_card(client, auth_headers, name: str):
    png = _png()
    return client.post(
        "/api/v1/verify/guarantor-card",
        headers=auth_headers,
        files={
            "card_front_image": ("front.png", io.BytesIO(png), "image/png"),
            "card_back_image": ("back.png", io.BytesIO(png), "image/png"),
        },
        data={"full_name": name},
    )


def test_two_names_on_the_card_verify_despite_the_typed_surname_not_being_there(
    client, auth_headers, acceptable_card, ocr_says
):
    """The reported bug, at the boundary.

    The person typed "Kwadwo Mensah Jnr". Two of those three names are on the
    card. The old rule demanded the last token specifically, so this was refused
    - and refused as "the name does not match on their ID", telling the
    applicant their name was wrong when it was not.
    """
    ocr_says(
        front=["SURNAME: MENSAH", "GIVEN NAMES: KWADWO"],
        back=["P<GHAMENSAH<<KWADWO<<<<<<<<<<<<<<<<<<<<<<<<"],
    )

    r = post_card(client, auth_headers, "Kwadwo Mensah Jnr")

    assert r.status_code == 200, r.text
    assert r.json()["name"] == "verified", r.json()


def test_a_four_letter_given_name_survives_one_misread_glyph(
    client, auth_headers, acceptable_card, ocr_says
):
    """KOFI, AMA, YAW are four letters. The single-glyph tolerance started at
    five, so the everyday short name was the one with no tolerance at all."""
    ocr_says(front=["MENSAH", "KOF1"], back=[])

    r = post_card(client, auth_headers, "Kofi Mensah")

    assert r.status_code == 200, r.text
    assert r.json()["name"] == "verified", r.json()


def test_one_shared_name_is_review_not_a_verdict(
    client, auth_headers, acceptable_card, ocr_says
):
    ocr_says(front=["MENSAH", "GHANA CARD"], back=[])

    r = post_card(client, auth_headers, "Kwadwo Mensah")

    assert r.status_code == 200, r.text
    body = r.json()
    # Never a rejection. The endpoint answers 200 and hands back a verdict; what
    # the caller does with a non-verified verdict is the other half of this bug,
    # fixed and pinned on the Laundex side.
    assert body["name"] == "manual_review", body
    assert body["card"] == "verified"


def test_an_unreadable_card_is_not_reported_as_a_wrong_name(
    client, auth_headers, acceptable_card, ocr_says
):
    """The distinction that makes the next diagnosis possible.

    "We could not read your card" and "your name does not match" have different
    fixes - retake the photo, versus check the spelling - and they were reported
    as the same thing.
    """
    ocr_says(front=[], back=[])

    r = post_card(client, auth_headers, "Kofi Mensah")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "manual_review", body
    assert "read" in body["name_note"].lower(), body["name_note"]


def test_the_response_says_how_many_names_matched(
    client, auth_headers, acceptable_card, ocr_says
):
    """Counts only, never the name. This is what turns the next failure of this
    check into something one log line answers."""
    ocr_says(front=["MENSAH", "GHANA CARD"], back=[])

    body = post_card(client, auth_headers, "Kwadwo Mensah").json()

    assert body["name_matched"] == 1
    assert body["name_typed"] == 2
    # The whole point: no name is echoed back into a response or a log.
    assert "KWADWO" not in str(body).upper()