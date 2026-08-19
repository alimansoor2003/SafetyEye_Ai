from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

import cv2
from dotenv import load_dotenv

from config import REPO_ROOT, CameraConfig, load_config
from vision.annotate import annotate
from vision.associator import associate
from vision.capture import VideoSource
from vision.detector import Detector
from vision.state_machine import ViolationEvent, ViolationStateMachine

log = logging.getLogger("vision.run")

FRAME_EVENT_HZ = 5.0
# Frame timestamps on a clip clock land exactly on the throttle boundary, where float error would
# silently drop every other tick. The epsilon keeps the rate at the advertised ~5/s.
FRAME_EVENT_INTERVAL = 1.0 / FRAME_EVENT_HZ - 1e-6


def zone_code(zone_id: str) -> str:
    parts = [p for p in re.split(r"[-_]", zone_id) if p]
    return ("".join(parts[:2]) if len(parts) >= 2 else zone_id).upper()


def next_report_id(evidence_dir: Path, zone_id: str, detected_at_utc) -> str:
    """M1 sequences from evidence filenames. M2 takes this over via the SQLite UNIQUE report_id."""
    day = detected_at_utc.strftime("%Y%m%d")
    prefix = f"INC-{day}-{zone_code(zone_id)}-"
    seq = 0
    for path in evidence_dir.glob(f"{prefix}*.jpg"):
        try:
            seq = max(seq, int(path.stem[len(prefix):]))
        except ValueError:
            continue
    return f"{prefix}{seq + 1:04d}"


def incident_payload(event: ViolationEvent, cam: CameraConfig, report_id: str, evidence: Path) -> dict:
    return {
        "type": "incident.created",
        "data": {
            "report_id": report_id,
            "camera_id": cam.camera_id,
            "zone_id": cam.zone_id,
            "track_id": event.track_id,
            "violation_type": event.violation_type,
            "confidence": round(event.confidence, 3),
            "duration_seconds": event.duration_seconds,
            "detected_at_utc": event.detected_at_utc.isoformat().replace("+00:00", "Z"),
            "evidence_url": f"/evidence/{evidence.name}",
            "status": "pending",
            "suppressed_since_last": event.suppressed_since_last,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m vision.run", description="EdgeSentinel M1 vision core")
    p.add_argument("--camera", default="CAM-01", help="camera_id from config.yaml")
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--source", default=None, help="override the configured source (index or path/URL)")
    p.add_argument("--device", default=None, help="override detection.device, e.g. cuda:0")
    p.add_argument("--show", action="store_true", help="open a preview window with overlays")
    p.add_argument("--frames", action="store_true", help="also print throttled detection.frame events")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames (smoke testing)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stderr,
    )

    cfg = load_config(args.config)
    cam = cfg.camera(args.camera)
    if args.source is not None:
        cam.source = int(args.source) if args.source.isdigit() else args.source
    if args.device is not None:
        cfg.detection.device = args.device

    evidence_dir = REPO_ROOT / cfg.storage.evidence_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector(cfg.detection)
    machine = ViolationStateMachine(cfg.violation)

    frame_times: deque[float] = deque(maxlen=30)
    last_frame_event = 0.0
    log.info("monitoring %s (%s) — Ctrl+C to stop", cam.camera_id, cam.zone_id)

    try:
        with VideoSource(cam) as source:
            # A recorded clip decodes far faster than realtime, so violation timers must run on the
            # clip's own timebase or the 2s rule can never be satisfied. Live sources use the wall clock.
            clip_clock = source.is_file
            if clip_clock:
                log.info("file source — timing violations at the clip's %.1ffps", source.fps)

            for index, frame in enumerate(source.frames()):
                wall = time.monotonic()
                now = index / source.fps if clip_clock else wall
                frame_times.append(wall)

                detections = detector.track(frame)
                people, unbound = associate(detections, cfg.detection.monitored_violations)

                active = {p.track_id: dict(p.violations) for p in people if p.violations}
                seen = {p.track_id for p in people}
                events = machine.update(now, active, seen)

                elapsed_by_track = {
                    p.track_id: {v: machine.elapsed(now, p.track_id, v) for v in p.violations}
                    for p in people
                }

                if events or args.show:
                    canvas = annotate(
                        frame, people, unbound, elapsed_by_track, cam.zone_label_en, _fps(frame_times)
                    )
                else:
                    canvas = None

                for event in events:
                    report_id = next_report_id(evidence_dir, cam.zone_id, event.detected_at_utc)
                    path = evidence_dir / f"{report_id}.jpg"
                    cv2.imwrite(str(path), canvas)
                    _emit(incident_payload(event, cam, report_id, path))

                if args.frames and now - last_frame_event >= FRAME_EVENT_INTERVAL:
                    last_frame_event = now
                    _emit(_frame_payload(cam, people, elapsed_by_track))

                if args.show:
                    cv2.imshow(f"EdgeSentinel — {cam.camera_id}", canvas)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

                if args.max_frames is not None and index + 1 >= args.max_frames:
                    log.info("reached --max-frames %d", args.max_frames)
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if args.show:
            cv2.destroyAllWindows()
        _log_stats(machine)

    return 0


def _fps(times: deque[float]) -> float | None:
    if len(times) < 2:
        return None
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 0 else None


def _frame_payload(cam: CameraConfig, people, elapsed_by_track) -> dict:
    from datetime import datetime, timezone

    return {
        "type": "detection.frame",
        "data": {
            "camera_id": cam.camera_id,
            "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "compliant": not any(p.violations for p in people),
            "tracks": [
                {
                    "track_id": p.track_id,
                    "bbox": list(p.bbox),
                    "label": "Person",
                    "conf": round(p.conf, 3),
                    "violations": list(p.violations),
                    "violation_elapsed": max(
                        elapsed_by_track.get(p.track_id, {}).values(), default=0.0
                    ),
                }
                for p in people
            ],
        },
    }


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _log_stats(machine: ViolationStateMachine) -> None:
    s = machine.stats
    log.info(
        "tracks_seen=%d emitted=%d suppressed_cooldown=%d suppressed_breaker=%d reid_churn_discards=%d",
        s.tracks_seen, s.events_emitted, s.events_suppressed_cooldown,
        s.events_suppressed_breaker, s.arming_discarded_track_lost,
    )


if __name__ == "__main__":
    raise SystemExit(main())
