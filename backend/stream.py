"""MJPEG generator. Pixels only — this transport never carries events (spec §1)."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

BOUNDARY = "frame"
MEDIA_TYPE = f"multipart/x-mixed-replace; boundary={BOUNDARY}"


def placeholder_jpeg(width: int, height: int, text: str = "NO SIGNAL") -> bytes:
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    cv2.putText(
        frame, text, ((width - size[0]) // 2, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA,
    )
    return cv2.imencode(".jpg", frame)[1].tobytes()


async def mjpeg_stream(pipeline, fps: int, width: int, height: int) -> AsyncIterator[bytes]:
    interval = 1.0 / max(1, fps)
    fallback = placeholder_jpeg(width, height)
    last_sent: bytes | None = None

    try:
        while True:
            jpeg = pipeline.latest_jpeg or fallback
            # Re-send the current frame even when unchanged; an MJPEG client that receives
            # nothing for seconds will time out the connection.
            last_sent = jpeg
            yield (
                f"--{BOUNDARY}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n\r\n"
            ).encode() + jpeg + b"\r\n"
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.debug("mjpeg client disconnected", exc_info=True)
