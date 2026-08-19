"""PPE-to-person binding. Pure geometry — no GPU, camera, or weights."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.associator import associate
from vision.detector import (
    HARDHAT,
    NO_HARDHAT,
    NO_SAFETY_VEST,
    PERSON,
    SAFETY_VEST,
    Detection,
)

MONITORED = [NO_HARDHAT, NO_SAFETY_VEST]

# A person standing at x=100..200, y=0..400. Head band is y < 160, torso band y=40..320.
PERSON_A = Detection(bbox=(100, 0, 200, 400), label=PERSON, conf=0.95, track_id=1)
HEAD_BOX = (120, 10, 180, 60)
TORSO_BOX = (110, 150, 190, 260)
FOOT_BOX = (110, 360, 190, 395)


def det(bbox, label, conf=0.9) -> Detection:
    return Detection(bbox=bbox, label=label, conf=conf)


def test_hardhat_binds_and_clears_violation():
    people, unbound = associate([PERSON_A, det(HEAD_BOX, HARDHAT)], MONITORED)
    assert len(people) == 1
    assert HARDHAT in people[0].ppe
    assert people[0].violations == {}
    assert unbound == []


def test_no_hardhat_binds_and_raises_violation():
    people, _ = associate([PERSON_A, det(HEAD_BOX, NO_HARDHAT, 0.88)], MONITORED)
    assert people[0].violations == {NO_HARDHAT: 0.88}


def test_positive_wins_when_more_confident():
    dets = [PERSON_A, det(HEAD_BOX, HARDHAT, 0.92), det(HEAD_BOX, NO_HARDHAT, 0.80)]
    people, _ = associate(dets, MONITORED)
    assert people[0].violations == {}, "higher-confidence Hardhat should cancel NO-Hardhat"


def test_negative_wins_when_more_confident():
    dets = [PERSON_A, det(HEAD_BOX, HARDHAT, 0.78), det(HEAD_BOX, NO_HARDHAT, 0.91)]
    people, _ = associate(dets, MONITORED)
    assert people[0].violations == {NO_HARDHAT: 0.91}


def test_vest_binds_on_torso():
    people, _ = associate([PERSON_A, det(TORSO_BOX, NO_SAFETY_VEST, 0.83)], MONITORED)
    assert people[0].violations == {NO_SAFETY_VEST: 0.83}


def test_hardhat_at_foot_level_is_rejected_by_band_gate():
    people, unbound = associate([PERSON_A, det(FOOT_BOX, NO_HARDHAT)], MONITORED)
    assert people[0].violations == {}, "a hardhat box at the feet must not bind"
    assert len(unbound) == 1


def test_ppe_outside_every_person_is_unbound():
    people, unbound = associate([PERSON_A, det((600, 10, 660, 60), NO_HARDHAT)], MONITORED)
    assert people[0].ppe == {}
    assert len(unbound) == 1


def test_overlapping_people_bind_to_the_right_head():
    """B stands behind and lower; A's head box must not land on B."""
    person_b = Detection(bbox=(150, 120, 250, 500), label=PERSON, conf=0.9, track_id=2)
    people, _ = associate([PERSON_A, person_b, det(HEAD_BOX, NO_HARDHAT)], MONITORED)
    by_id = {p.track_id: p for p in people}
    assert by_id[1].violations == {NO_HARDHAT: 0.9}
    assert by_id[2].violations == {}, "head box sits outside B's head band"


def test_untracked_person_is_dropped():
    ghost = Detection(bbox=(100, 0, 200, 400), label=PERSON, conf=0.9, track_id=None)
    people, _ = associate([ghost, det(HEAD_BOX, NO_HARDHAT)], MONITORED)
    assert people == [], "a person with no track_id cannot carry a violation timer"


def test_unmonitored_violation_is_not_reported():
    people, _ = associate([PERSON_A, det(TORSO_BOX, NO_SAFETY_VEST)], [NO_HARDHAT])
    assert people[0].violations == {}
    assert NO_SAFETY_VEST in people[0].ppe, "still bound, just not monitored"


def test_higher_confidence_duplicate_replaces_lower():
    dets = [PERSON_A, det(HEAD_BOX, NO_HARDHAT, 0.77), det(HEAD_BOX, NO_HARDHAT, 0.94)]
    people, _ = associate(dets, MONITORED)
    assert people[0].violations == {NO_HARDHAT: 0.94}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
