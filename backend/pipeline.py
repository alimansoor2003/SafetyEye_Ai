"""Vision loop -> asyncio bridge.

The M1 loop runs in a worker thread (YOLO inference is blocking and GPU-bound). It never touches
the database or the event loop directly: it pushes plain payloads across a thread-safe hand-off,
and an async drain task does persistence and broadcast. That keeps the detection loop free of
awaits, which is the whole point of spec §1's "detection loop never awaits the API".
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from config import CameraConfig, Config
from vision.annotate import annotate
from vision.associator import associate
from vision.capture import VideoSource
from vision.detector import Detector
from vision.state_machine import ViolationStateMachine

log = logging.getLogger(__name__)

FRAME_EVENT_HZ = 5.0
FRAME_EVENT_INTERVAL = 1.0 / FRAME_EVENT_HZ - 1e-6
EVIDENCE_QUALITY = 90


def zone_code(zone_id: str) -> str:
    parts = [p for p in re.split(r"[-_]", zone_id) if p]
    return ("".join(parts[:2]) if len(parts) >= 2 else zone_id).upper()


@dataclass
class IncidentDraft:
    """A fired violation batch, before it has a report_id or a row."""

    track_id: int
    violations: list[str]
    confidence: float
    duration_seconds: float
    detected_at_utc: datetime
    evidence_jpeg: bytes


@dataclass
class FrameEvent:
    payload: dict
    detection_rows: list[tuple] = field(default_factory=list)


class VisionPipeline:
    def __init__(self, cfg: Config, camera: CameraConfig, evidence_dir: Path):
        self.cfg = cfg
        self.camera = camera
        self.evidence_dir = evidence_dir

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None

        self._frame_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None

        self._times: deque[float] = deque(maxlen=45)
        self.camera_connected = False
        self.gpu_name: str | None = None
        self.last_error: str | None = None

    # ---------------------------------------------------------------- lifecycle

    def start(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        self._loop = loop
        self._queue = queue
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning("vision thread did not stop within %.0fs", timeout)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fps(self) -> float | None:
        with self._frame_lock:
            times = list(self._times)
        if len(times) < 2:
            return None
        span = times[-1] - times[0]
        return round((len(times) - 1) / span, 1) if span > 0 else None

    @property
    def latest_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    # ---------------------------------------------------------------- worker

    def _run(self) -> None:
        try:
            detector = Detector(self.cfg.detection)
            self.gpu_name = _gpu_name(self.cfg.detection.device)
            machine = ViolationStateMachine(self.cfg.violation)
        except Exception as exc:
            self.last_error = str(exc)
            log.exception("vision thread failed to initialise")
            return

        monitored = self.cfg.detection.monitored_violations
        last_frame_event = 0.0
        last_publish = 0.0
        # Capture runs at 30fps but the stream only needs 15. Annotating, resizing and
        # JPEG-encoding every captured frame halved the loop rate for frames nobody sees.
        publish_interval = 1.0 / max(1, self.cfg.stream.fps) - 1e-6

        try:
            with VideoSource(self.camera) as source:
                self.camera_connected = True
                clip_clock = source.is_file
                for index, frame in enumerate(source.frames()):
                    if self._stop.is_set():
                        break

                    wall = time.monotonic()
                    now = index / source.fps if clip_clock else wall

                    detections = detector.track(frame)
                    people, unbound = associate(detections, monitored)

                    active = {p.track_id: dict(p.violations) for p in people if p.violations}
                    seen = {p.track_id for p in people}
                    events = machine.update(now, active, seen)

                    elapsed = {
                        p.track_id: {v: machine.elapsed(now, p.track_id, v) for v in p.violations}
                        for p in people
                    }

                    self._mark_loop(wall)

                    publish_due = wall - last_publish >= publish_interval
                    canvas = None
                    if publish_due or events:
                        canvas = annotate(
                            frame, people, unbound, elapsed, self.camera.zone_label_en, self.fps
                        )

                    if publish_due:
                        last_publish = wall
                        self._publish_frame(canvas)

                    if events:
                        self._publish_incidents(events, canvas)

                    if now - last_frame_event >= FRAME_EVENT_INTERVAL:
                        last_frame_event = now
                        self._publish_detection_frame(people, elapsed)
        except Exception as exc:
            self.last_error = str(exc)
            log.exception("vision loop crashed")
        finally:
            self.camera_connected = False
            _log_stats(machine)

    # ---------------------------------------------------------------- publishing

    def _mark_loop(self, wall: float) -> None:
        """Loop rate, measured on every captured frame — not the throttled publish rate."""
        with self._frame_lock:
            self._times.append(wall)

    def _publish_frame(self, canvas: np.ndarray) -> None:
        stream = self.cfg.stream
        resized = cv2.resize(canvas, (stream.width, stream.height), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(
            ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), stream.jpeg_quality]
        )
        if not ok:
            return
        with self._frame_lock:
            self._latest_jpeg = buffer.tobytes()

    def _publish_incidents(self, events, canvas: np.ndarray) -> None:
        """Amendment A2: simultaneous violations on one track become a single incident."""
        ok, buffer = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), EVIDENCE_QUALITY]
        )
        evidence = buffer.tobytes() if ok else b""

        by_track: dict[int, list] = {}
        for event in events:
            by_track.setdefault(event.track_id, []).append(event)

        for track_id, batch in by_track.items():
            self._emit(
                IncidentDraft(
                    track_id=track_id,
                    violations=[e.violation_type for e in batch],
                    confidence=max(e.confidence for e in batch),
                    duration_seconds=max(e.duration_seconds for e in batch),
                    detected_at_utc=min(e.detected_at_utc for e in batch),
                    evidence_jpeg=evidence,
                )
            )

    def _publish_detection_frame(self, people, elapsed: dict) -> None:
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        tracks = [
            {
                "track_id": p.track_id,
                "bbox": list(p.bbox),
                "label": "Person",
                "conf": round(p.conf, 3),
                "violations": list(p.violations),
                "violation_elapsed": max(elapsed.get(p.track_id, {}).values(), default=0.0),
            }
            for p in people
        ]
        payload = {
            "camera_id": self.camera.camera_id,
            "ts_utc": ts,
            "compliant": not any(p.violations for p in people),
            "tracks": tracks,
        }

        rows: list[tuple] = []
        if self.cfg.storage.persist_detections:
            import json

            # Only non-compliant tracks are persisted; storing every clean frame would add
            # ~430k rows/day for no forensic value.
            rows = [
                (
                    self.camera.camera_id, ts, t["track_id"], json.dumps(t["bbox"]),
                    t["conf"], json.dumps(t["violations"]), t["violation_elapsed"], None,
                )
                for t in tracks
                if t["violations"]
            ]

        self._emit(FrameEvent(payload=payload, detection_rows=rows))

    def _emit(self, item) -> None:
        if self._loop is None or self._queue is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            pass  # loop closed during shutdown
        except asyncio.QueueFull:
            log.warning("event queue full — dropped %s", type(item).__name__)


def _gpu_name(device: str) -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = int(device.split(":")[1]) if ":" in device else 0
        return torch.cuda.get_device_name(index)
    except Exception:
        return None


def _log_stats(machine: ViolationStateMachine) -> None:
    s = machine.stats
    log.info(
        "tracks_seen=%d emitted=%d suppressed_cooldown=%d suppressed_breaker=%d reid_churn=%d",
        s.tracks_seen, s.events_emitted, s.events_suppressed_cooldown,
        s.events_suppressed_breaker, s.arming_discarded_track_lost,
    )
