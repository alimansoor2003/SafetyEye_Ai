from __future__ import annotations

import cv2
import numpy as np

from vision.associator import TrackedPerson
from vision.detector import Detection

GREEN = (60, 200, 60)
RED = (40, 40, 235)
AMBER = (30, 170, 240)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# Overlays are sized relative to a 720p reference so they stay legible on a 1080p capture,
# where fixed pixel sizes render too small to read at a distance.
REFERENCE_HEIGHT = 720
BANNER_H = 46


def annotate(
    frame: np.ndarray,
    people: list[TrackedPerson],
    unbound: list[Detection],
    elapsed_by_track: dict[int, dict[str, float]],
    zone_label: str,
    fps: float | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    s = max(1.0, canvas.shape[0] / REFERENCE_HEIGHT)

    for det in unbound:
        _box(canvas, det.bbox, GREY, round(1 * s))

    for person in people:
        violating = bool(person.violations)
        color = RED if violating else GREEN
        _box(canvas, person.bbox, color, round((3 if violating else 2) * s))

        for label, det in person.ppe.items():
            _box(canvas, det.bbox, RED if label in person.violations else GREEN, round(1 * s))

        lines = [f"ID {person.track_id}"]
        for vtype, conf in person.violations.items():
            held = elapsed_by_track.get(person.track_id, {}).get(vtype, 0.0)
            lines.append(f"{vtype}  {conf:.2f}  {held:.1f}s")
        if not violating:
            lines.append("COMPLIANT")
        _label(canvas, person.bbox, lines, color, s)

    _banner(canvas, any(p.violations for p in people), zone_label, len(people), fps, s)
    return canvas


def _box(img: np.ndarray, bbox: tuple[int, int, int, int], color, thickness: int) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _label(
    img: np.ndarray, bbox: tuple[int, int, int, int], lines: list[str], color, s: float
) -> None:
    x1, y1, _, _ = bbox
    scale, thick, pad = 0.5 * s, max(1, round(s)), round(4 * s)
    sizes = [cv2.getTextSize(t, FONT, scale, thick)[0] for t in lines]
    box_w = max(w for w, _ in sizes) + pad * 2
    line_h = max(h for _, h in sizes) + round(6 * s)
    box_h = line_h * len(lines) + pad

    # Heads sit near the top of frame, so an unclamped label would slide under the status banner.
    top = max(round(BANNER_H * s), y1 - box_h)
    cv2.rectangle(img, (x1, top), (x1 + box_w, top + box_h), color, -1)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (x1 + pad, top + line_h * (i + 1) - pad),
                    FONT, scale, WHITE, thick, cv2.LINE_AA)


def _banner(
    img: np.ndarray, hazard: bool, zone_label: str, people: int, fps: float | None, s: float
) -> None:
    h, w = img.shape[:2]
    color = RED if hazard else GREEN
    text = "HAZARD DETECTED" if hazard else "COMPLIANT"
    banner_h = round(BANNER_H * s)

    cv2.rectangle(img, (0, 0), (w, banner_h), color, -1)
    cv2.putText(img, text, (round(14 * s), round(32 * s)),
                FONT, 0.9 * s, WHITE, max(1, round(2 * s)), cv2.LINE_AA)

    right = f"{zone_label}   people: {people}"
    if fps is not None:
        right += f"   {fps:.1f} fps"
    size = cv2.getTextSize(right, FONT, 0.6 * s, max(1, round(s)))[0]
    cv2.putText(img, right, (w - size[0] - round(14 * s), round(30 * s)),
                FONT, 0.6 * s, WHITE, max(1, round(s)), cv2.LINE_AA)
