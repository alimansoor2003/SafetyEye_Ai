"""End-to-end wiring test: synthetic clip + scripted detections, no GPU or weights.

Substitutes a fake Detector for YOLOv8 so capture -> associate -> state machine -> evidence -> JSON
can be exercised on any machine. Only the model forward pass is unverified.
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import vision.run as run
from vision.detector import HARDHAT, NO_HARDHAT, PERSON, Detection

CLIP_FPS = 15
CLIP_FRAMES = 150                 # 10s
VIOLATION_FRAMES = range(30, 105)  # 5s non-compliant, from t=2.0s to t=7.0s

PERSON_BOX = (200, 60, 340, 440)
HEAD_BOX = (240, 70, 300, 130)


class FakeDetector:
    """Scripted stand-in for YOLOv8 + ByteTrack. Track ID 1 is stable throughout."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = 0

    def track(self, frame):
        index = self.calls
        self.calls += 1
        ppe = NO_HARDHAT if index in VIOLATION_FRAMES else HARDHAT
        return [
            Detection(bbox=PERSON_BOX, label=PERSON, conf=0.95, track_id=1),
            Detection(bbox=HEAD_BOX, label=ppe, conf=0.88),
        ]


def make_clip(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), CLIP_FPS, (640, 480))
    assert writer.isOpened(), "could not open VideoWriter — MJPG encoder missing"
    for i in range(CLIP_FRAMES):
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, PERSON_BOX[:2], PERSON_BOX[2:], (90, 90, 90), -1)
        cv2.putText(frame, str(i), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def make_config(workdir: Path, clip: Path) -> Path:
    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    cfg["cameras"][0]["source"] = str(clip)
    cfg["cameras"][0]["capture_fps"] = CLIP_FPS
    cfg["storage"]["evidence_dir"] = str(workdir / "evidence")
    path = workdir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def drive(workdir: Path, extra: list[str] | None = None) -> tuple[list[dict], Path]:
    """Run the M1 entrypoint against the synthetic clip; return (emitted events, evidence dir)."""
    clip = workdir / "clip.avi"
    if not clip.exists():
        make_clip(clip)
    config = make_config(workdir, clip)

    real_detector = run.Detector
    run.Detector = FakeDetector
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            code = run.main(["--camera", "CAM-01", "--config", str(config)] + (extra or []))
    finally:
        run.Detector = real_detector

    assert code == 0, f"run.main exited {code}"
    events = [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]
    return events, workdir / "evidence"


def test_single_incident_with_evidence_jpeg():
    workdir = Path(tempfile.mkdtemp())
    try:
        events, evidence_dir = drive(workdir)
        incidents = [e for e in events if e["type"] == "incident.created"]
        assert len(incidents) == 1, f"expected 1 incident from one 5s violation, got {len(incidents)}"

        data = incidents[0]["data"]
        assert data["violation_type"] == NO_HARDHAT
        assert data["track_id"] == 1
        assert data["status"] == "pending"
        assert data["zone_id"] == "ZONE-01-MAIN-ENTRANCE"
        assert 2.0 <= data["duration_seconds"] < 2.3, f"fired at {data['duration_seconds']}s"
        assert data["detected_at_utc"].endswith("Z"), "timestamps must be UTC ISO-8601"

        jpeg = evidence_dir / f"{data['report_id']}.jpg"
        assert jpeg.exists(), f"no evidence written at {jpeg}"
        assert cv2.imread(str(jpeg)) is not None, "evidence JPEG is not decodable"
        assert data["evidence_url"] == f"/evidence/{jpeg.name}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_report_id_format_and_sequence():
    workdir = Path(tempfile.mkdtemp())
    try:
        first, _ = drive(workdir)
        second, _ = drive(workdir)
        ids = [e["data"]["report_id"] for e in first + second if e["type"] == "incident.created"]
        assert len(ids) == 2, f"expected one incident per run, got {ids}"

        import re
        assert re.fullmatch(r"INC-\d{8}-ZONE01-0001", ids[0]), ids[0]
        assert re.fullmatch(r"INC-\d{8}-ZONE01-0002", ids[1]), ids[1]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_detection_frame_payload_shape():
    workdir = Path(tempfile.mkdtemp())
    try:
        events, _ = drive(workdir, ["--frames"])
        frames = [e for e in events if e["type"] == "detection.frame"]
        assert frames, "no detection.frame events emitted with --frames"

        # 10s of clip at 5Hz, allowing for the throttle boundary.
        assert 45 <= len(frames) <= 55, f"throttle produced {len(frames)} events over 10s"

        track = frames[0]["data"]["tracks"][0]
        assert set(track) == {"track_id", "bbox", "label", "conf", "violations", "violation_elapsed"}
        assert track["label"] == "Person" and len(track["bbox"]) == 4

        assert any(f["data"]["compliant"] is False for f in frames)
        assert any(f["data"]["compliant"] is True for f in frames)

        held = [f["data"]["tracks"][0]["violation_elapsed"] for f in frames]
        assert max(held) >= 4.0, f"violation_elapsed never accumulated (max {max(held)})"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


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
