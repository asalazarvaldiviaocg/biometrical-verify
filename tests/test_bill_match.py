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


def test_id_name_match_verifies():
    ocr = ["ECOWAS IDENTITY CARD", "ODONKOR<<CEDRIC<AKUETTEH<MENSA",
           "REPUBLIC OF GHANA"]
    ok, note = bm.match_id_name(ocr, "Cedric Mensa Odonkor")
    assert ok and note == ""


def test_id_name_single_glyph_tolerance():
    # OCR misread one letter of the surname and no exact copy exists:
    # still the same name via single-edit tolerance.
    ok, _ = bm.match_id_name(["ODONKAR<<CEDRIC"], "Cedric Odonkor")
    assert ok


def test_id_name_mismatch_manual():
    ok, note = bm.match_id_name(
        ["ODONKOR<<CEDRIC<AKUETTEH<MENSA"], "Ama Serwaa")
    assert not ok and "match" in note


def test_id_name_empty_manual():
    ok, _ = bm.match_id_name([], "Cedric Odonkor")
    assert not ok


# ---------------------------------------------------------------------------
# The rule the guarantor step actually needs
#
# A guarantor is a person vouching for somebody else. They are the least
# motivated participant in the whole flow and the most likely to be turned away
# by a machine being pedantic about spelling, so the bar for "this is the same
# human" is low on purpose.
#
# The old rule was `surname matched AND at least two tokens matched`. That
# silently required the LAST token the person typed to be on the card, which is
# a guess about naming convention and not a fact about identity. It rejected
# three shapes that are all the same person:
#
#     "Kwadwo Mensah Jnr"        2 of 3 tokens on the card, "Jnr" is not
#     "Kwadwo Mensah-Boateng"    hyphen splits into two tokens, surname is the second half
#     "Cedric Odonkor Mensah"    card prints MENSAH first, so "Mensah" is not last
#
# The rule is now simply: two of the typed name's tokens are on the card. Any
# two, not a particular two. A surname match still counts and is still recorded,
# but it is not a gate - nothing about which token is last is evidence.
# ---------------------------------------------------------------------------


def test_two_matches_verify_without_the_typed_surname_being_present():
    # "Jnr" is a suffix, not a name on the card. KWADWO and MENSAH both are.
    ocr = ["SURNAME: MENSAH", "GIVEN NAMES: KWADWO", "P<GHAMENSAH<<KWADWO"]
    ok, note = bm.match_id_name(ocr, "Kwadwo Mensah Jnr")
    assert ok, f"two of three names are on the card: {note}"


def test_two_matches_verify_when_the_card_orders_the_name_differently():
    # The card prints the surname first. The person typed it last. Same human.
    ocr = ["MENSAH, KWADWO YAW", "P<GHAMENSAH<<KWADWO<YAW"]
    ok, note = bm.match_id_name(ocr, "Kwadwo Yaw Mensah")
    assert ok, note


def test_hyphenated_surname_verifies_on_its_two_halves():
    ocr = ["MENSAH BOATENG KWADWO", "P<GHAMENSAH<BOATENG<<KWADWO"]
    ok, note = bm.match_id_name(ocr, "Kwadwo Mensah-Boateng")
    assert ok, note


def test_one_match_is_still_not_enough():
    # The floor is two. One shared token is a coincidence, not an identity.
    ocr = ["MENSAH", "GHANA CARD"]
    ok, note = bm.match_id_name(ocr, "Kwadwo Mensah")
    assert not ok
    assert "match" in note


def test_no_match_is_still_not_enough():
    ocr = ["AMENYO", "GHANA CARD"]
    ok, note = bm.match_id_name(ocr, "Kwadwo Mensah")
    assert not ok


def test_four_letter_names_get_single_glyph_tolerance():
    # KOFI, AMA, YAW, ABI, NANA - four-letter given names are extremely common
    # here, and the old tolerance only applied at five characters or more. So
    # the name this document exists to catch, a one-glyph OCR misread, was the
    # one case with none: "KOF1" read as a different name entirely.
    ok, note = bm.match_id_name(["MENSAH", "KOF1"], "Kofi Mensah")
    assert ok, f"one glyph off a four-letter given name: {note}"


def test_three_letter_noise_still_does_not_match():
    # The tolerance must not widen so far that short tokens collide. "AMA"
    # and "AMP" are two edits apart, and they are different names.
    ok, _ = bm.match_id_name(["MENSAH", "AMP"], "Ama Mensah")
    assert not ok


def test_the_note_says_what_actually_happened():
    # "name does not match" was reported to the user for a card we simply could
    # not read, which is a different problem with a different fix. The note has
    # to distinguish them, or the next diagnosis is another round of guessing.
    _, unreadable = bm.match_id_name([], "Kofi Mensah")
    assert "read" in unreadable.lower()

    _, wrong = bm.match_id_name(["AMENYO", "GHANA CARD"], "Kofi Mensah")
    assert "did not match" in wrong.lower() or "does not match" in wrong.lower()
    assert "read" not in wrong.lower()


# ---------------------------------------------------------------------------
# The grade: verified, mismatch, unclear.
#
# A count cannot carry the difference between "this card carries a different
# name" and "this photograph could not be read well enough to say". One of two
# names matched is the same number in both cases, so a caller reading only the
# count has to guess - and guessing wrong means refusing a person whose card is
# in their hand, or accepting a card that belongs to somebody else.
#
# The grade is the difference, and it can only be computed here: this is where
# the card text is.
# ---------------------------------------------------------------------------


def test_the_card_carries_the_name_so_it_verifies():
    status, note = bm.grade_id_name(
        ["ODONKOR<<CEDRIC<AKUETTEH<MENSA"], "Cedric Odonkor")
    assert status == bm.NAME_VERIFIED, note


def test_a_name_the_card_does_not_carry_is_a_mismatch():
    """Nothing of the typed name is on a card we could read."""
    status, note = bm.grade_id_name(
        ["ODONKOR<<CEDRIC<AKUETTEH<MENSA"], "Kwadwo Mensah")
    assert status == bm.NAME_MISMATCH, note


def test_one_name_on_the_card_and_one_absent_is_a_mismatch():
    """The case that had no answer at all.

    "Cedric" is on the card and the surname the person typed is not. That is
    somebody else's card, and it was indistinguishable from a surname the OCR
    had mangled - the same "1 of 2 matched" in both cases. One of those is a
    fact about the document and the other is a fact about a photograph.
    """
    status, note = bm.grade_id_name(
        ["ODONKOR<<CEDRIC<AKUETTEH<MENSA"], "Cedric Asare")
    assert status == bm.NAME_MISMATCH, note


def test_a_mangled_surname_is_unclear_not_a_mismatch():
    """The same shape, but the surname on the card was read badly - two edits
    from what was typed. The card carries the name; the machine did not read
    it. A person decides that one."""
    status, note = bm.grade_id_name(["CEDRIC", "MENSHA"], "Cedric Mensah")
    assert status == bm.NAME_UNCLEAR, note


def test_a_suffix_the_person_typed_is_not_a_wrong_name():
    """"Jnr" is not on the card because the card does not carry suffixes. It is
    three letters long, which is also too short to call a name absent: ABI,
    NANA and AMA are ordinary names, and a short token that is missing is far
    more likely to be a suffix than a false name."""
    status, note = bm.grade_id_name(["KWADWO", "MENSAH"], "Kwadwo Jnr")
    assert status == bm.NAME_UNCLEAR, note


def test_a_card_we_could_not_read_is_unclear():
    """Not a mismatch. We did not read the card; we know nothing about what is
    on it."""
    status, note = bm.grade_id_name([], "Kofi Mensah")
    assert status == bm.NAME_UNCLEAR, note
    assert "read" in note.lower()


def test_a_one_token_name_the_card_does_not_carry_is_a_mismatch():
    """Long enough to mean something. "Kofi" is not on that card."""
    status, note = bm.grade_id_name(["MENSAH", "GHANA CARD"], "Kofi")
    assert status == bm.NAME_MISMATCH, note


def test_a_one_token_name_too_short_to_judge_is_unclear():
    """"Ama" is three letters. The card may carry it and the machine may have
    read it badly; that is not a fact about the document."""
    status, note = bm.grade_id_name(["MENSAH", "GHANA CARD"], "Ama")
    assert status == bm.NAME_UNCLEAR, note


def test_a_one_token_name_that_is_on_the_card_is_unclear_not_verified():
    """The bar is two names, and one name cannot reach it. That is a name this
    check cannot grade, not a wrong name."""
    status, note = bm.grade_id_name(["Kofi", "GHANA CARD"], "Kofi")
    assert status == bm.NAME_UNCLEAR, note


def test_the_grade_never_echoes_the_name():
    """Counts and grades only. The response and the logs must never carry the
    person's name."""
    status, note = bm.grade_id_name(
        ["ODONKOR<<CEDRIC<AKUETTEH<MENSA"], "Cedric Asare")
    assert status == bm.NAME_MISMATCH
    assert "CEDRIC" not in note.upper()
    assert "ASARE" not in note.upper()
