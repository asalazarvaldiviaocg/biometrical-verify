"""Unit tests for the guarantor bill-name verdict (app.services.bill_match).

Pure function, synthetic fixtures. Pins: real-name bills verify despite
case/punctuation noise; foreign, partial, and empty reads route to
manual_review with a note — never verify, never reject.
"""

from app.services import bill_match as bm


def test_matching_bill_verifies():
    bill = [
        "ELECTRICITY COMPANY OF GHANA",
        "Account Name: ODONKOR CEDRIC A",
        "Meter No 0123456789",
        "Amount Due: GHS 45.20",
    ]
    ok, note = bm.match_bill_name(bill, "Cedric Akuetteh Mensa Odonkor")
    assert ok and note == ""


def test_noise_tolerant():
    ok, _ = bm.match_bill_name(
        ["Acct Name - odonkor, cedric!!!"], "Cedric Akuetteh Mensa Odonkor")
    assert ok


def test_foreign_bill_manual():
    other = ["Pennsylvania", "visitPA.com USA", "SAMPLE",
             "RARRISBURG, PA 17101-0000", "SEX:F 18 EYES: BRO"]
    ok, note = bm.match_bill_name(other, "Cedric Akuetteh Mensa Odonkor")
    assert not ok and note


def test_single_token_manual():
    ok, _ = bm.match_bill_name(
        ["Meter No 0123456789", "Amount Due GHS 45.20 Cedric"],
        "Cedric Akuetteh Mensa Odonkor")
    assert not ok


def test_empty_manual():
    ok, note = bm.match_bill_name([], "Cedric Akuetteh Mensa Odonkor")
    assert not ok and note
