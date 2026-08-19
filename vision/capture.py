from __future__ import annotations

import logging
import time
from typing import Iterator

import cv2
import numpy as np

from config import CameraConfig

log = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    pass


class VideoSource:
    """Camera / RTSP / file source with reconnect for live sources."""

    def __init__(self, camera: CameraConfig, reconnect_delay: float = 2.0):
        self.camera = camera
        self.reconnect_delay = reconnect_delay
        self._cap: cv2.VideoCapture | None = None
        self.is_file = isinstance(camera.source, str) and not _is_stream_url(camera.source)
        self.fps = float(camera.capture_fps)

    def open(self) -> None:
        src = self.camera.source
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) else cv2.VideoCapture(src)
        if not cap.isOpened():
            raise CaptureError(f"cannot open source {src!r} for {self.camera.camera_id}")

        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera.capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera.capture_height)
            cap.set(cv2.CAP_PROP_FPS, self.camera.capture_fps)

        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported_fps if reported_fps and reported_fps > 0 else float(self.camera.capture_fps)
        log.info(
            "%s opened %r at %dx%d @ %.1ffps",
            self.camera.camera_id,
            src,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.fps,
        )
        self._cap = cap

    def frames(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            self.open()

        while True:
            ok, frame = self._cap.read()
            if ok:
                yield frame
                continue

            if self.is_file:
                log.info("%s reached end of file", self.camera.camera_id)
                return

            log.warning(
                "%s read failed, reconnecting in %.1fs", self.camera.camera_id, self.reconnect_delay
            )
            self.release()
            time.sleep(self.reconnect_delay)
            try:
                self.open()
            except CaptureError as exc:
                log.error("%s reconnect failed: %s", self.camera.camera_id, exc)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _is_stream_url(source: str) -> bool:
    return source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
