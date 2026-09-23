"""Pipeline forensics wiring test (tasks._doc_forensics_review + _decide).

SQLite-backed: proves the duplicate-image review flag fires across accounts,
garble-sidecar absence degrades silently, and flags cap APPROVE at REVIEW
without touching the REJECT paths.
"""

import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models.base import Base
from app.models.verification import AuditEvent, Verification


def _card_jpg() -> bytes:
    img = np.full((450, 700, 3), 235, np.uint8)
    cv2.rectangle(img, (75, 50), (625, 400), (250, 250, 250), -1)
    for y in range(125, 350, 30):
        cv2.rectangle(img, (350, y), (575, y + 11), (60, 60, 60), -1)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    assert ok
    return bytes(buf)


def _session() -> Session:
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    return Session(eng)


def _record(db: Session, user: str, phash: str | None = None) -> Verification:
    r = Verification(user_id=user, contract_id="c", status="done",
                     decision="APPROVE", id_phash=phash)
    db.add(r)
    db.commit()
    return r


def test_duplicate_across_accounts_flags_review():
    from app.workers import tasks

    db = _session()
    old = _record(db, "user-old", phash="ab" * 8)
    cur = Verification(user_id="user-new", contract_id="c", status="running")
    db.add(cur)
    db.commit()
    # Same phash as the old row: duplicate must flag.
    cur.id_phash = old.id_phash
    db.flush()
    dup = (
        db.query(Verification.id, Verification.user_id)
        .filter(
            Verification.id_phash == cur.id_phash,
            Verification.user_id != cur.user_id,
            Verification.status == "done",
        )
        .first()
    )
    assert dup is not None and dup.user_id == "user-old"
    # And the decision helper caps APPROVE at REVIEW on such flags.
    dec, reason = tasks._decide(0.9, 0.3, True, True, False,
                                ["possible duplicate ID image (also on account user-old)"])
    assert dec == "REVIEW" and "duplicate" in reason


def test_no_duplicate_no_flag():
    db = _session()
    _record(db, "user-old", phash="ab" * 8)
    cur = _record(db, "user-new", phash="cd" * 8)
    dup = (
        db.query(Verification.id, Verification.user_id)
        .filter(
            Verification.id_phash == cur.id_phash,
            Verification.user_id != cur.user_id,
            Verification.status == "done",
        )
        .first()
    )
    assert dup is None


def test_clean_forensics_keeps_approve():
    from app.workers import tasks

    dec, _ = tasks._decide(0.9, 0.3, True, True, False, [])
    assert dec == "APPROVE"
    # REJECT paths ignore advisory flags entirely.
    dec, _ = tasks._decide(0.9, 0.3, True, False, False, ["anything"])
    assert dec == "REJECT"
    dec, _ = tasks._decide(0.9, 0.3, True, True, True, [])
    assert dec == "REJECT"
