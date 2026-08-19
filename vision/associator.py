from __future__ import annotations

from dataclasses import dataclass, field

from vision.detector import (
    HARDHAT,
    NO_HARDHAT,
    NO_SAFETY_VEST,
    PERSON,
    SAFETY_VEST,
    Detection,
)

# Which positive class cancels which violation class.
COUNTERPART = {NO_HARDHAT: HARDHAT, NO_SAFETY_VEST: SAFETY_VEST}

# Vertical band of the person box a class is allowed to occupy, as (top, bottom) fractions.
# Stops a hardhat on person A binding to person B's legs when boxes overlap.
BAND = {
    HARDHAT: (0.0, 0.40),
    NO_HARDHAT: (0.0, 0.40),
    SAFETY_VEST: (0.10, 0.80),
    NO_SAFETY_VEST: (0.10, 0.80),
}

MIN_CONTAINMENT = 0.55


@dataclass
class TrackedPerson:
    track_id: int
    bbox: tuple[int, int, int, int]
    conf: float
    ppe: dict[str, Detection] = field(default_factory=dict)
    violations: dict[str, float] = field(default_factory=dict)


def associate(
    detections: list[Detection], monitored_violations: list[str]
) -> tuple[list[TrackedPerson], list[Detection]]:
    """Bind PPE boxes to person tracks. Returns (people, unbound PPE detections)."""
    people = [
        TrackedPerson(track_id=d.track_id, bbox=d.bbox, conf=d.conf)
        for d in detections
        if d.label == PERSON and d.track_id is not None
    ]
    by_id = {p.track_id: p for p in people}
    unbound: list[Detection] = []

    for det in detections:
        if det.label == PERSON:
            continue
        owner = _best_owner(det, people)
        if owner is None:
            unbound.append(det)
            continue
        current = by_id[owner.track_id].ppe.get(det.label)
        if current is None or det.conf > current.conf:
            by_id[owner.track_id].ppe[det.label] = det

    for person in people:
        person.violations = _resolve_violations(person.ppe, monitored_violations)

    return people, unbound


def _best_owner(det: Detection, people: list[TrackedPerson]) -> TrackedPerson | None:
    best: TrackedPerson | None = None
    best_score = MIN_CONTAINMENT
    for person in people:
        score = _containment(det.bbox, person.bbox)
        if score >= best_score and _in_band(det.bbox, person.bbox, det.label):
            best, best_score = person, score
    return best


def _containment(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inner_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return (iw * ih) / inner_area


def _in_band(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int], label: str) -> bool:
    band = BAND.get(label)
    if band is None:
        return True
    _, by1, _, by2 = outer
    height = max(1, by2 - by1)
    center_y = (inner[1] + inner[3]) / 2
    frac = (center_y - by1) / height
    return band[0] <= frac <= band[1]


def _resolve_violations(ppe: dict[str, Detection], monitored: list[str]) -> dict[str, float]:
    violations: dict[str, float] = {}
    for label in monitored:
        negative = ppe.get(label)
        if negative is None:
            continue
        positive = ppe.get(COUNTERPART.get(label, ""))
        if positive is not None and positive.conf >= negative.conf:
            continue
        violations[label] = negative.conf
    return violations
