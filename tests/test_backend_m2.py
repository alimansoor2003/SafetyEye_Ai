"""M2 end-to-end: synthetic clip -> pipeline -> SQLite -> REST/WS/MJPEG.

Runs the real FastAPI app with a scripted stand-in for YOLOv8, so no GPU or camera is needed.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from vision.detector import HARDHAT, NO_HARDHAT, NO_SAFETY_VEST, PERSON, Detection

TOKEN = "test-token-m2"
CLIP_FPS = 15
CLIP_FRAMES = 120
VIOLATION_FROM = 15          # both violations start together at t=1.0s

PERSON_BOX = (200, 60, 340, 440)
HEAD_BOX = (240, 70, 300, 130)
TORSO_BOX = (205, 180, 335, 300)


class FakeDetector:
    """Emits NO-Hardhat and NO-Safety Vest simultaneously to exercise batching (amendment A2)."""

    def __init__(self, cfg):
        self.calls = 0

    def track(self, frame):
        index = self.calls
        self.calls += 1
        dets = [Detection(bbox=PERSON_BOX, label=PERSON, conf=0.95, track_id=1)]
        if index >= VIOLATION_FROM:
            dets.append(Detection(bbox=HEAD_BOX, label=NO_HARDHAT, conf=0.91))
            dets.append(Detection(bbox=TORSO_BOX, label=NO_SAFETY_VEST, conf=0.84))
        else:
            dets.append(Detection(bbox=HEAD_BOX, label=HARDHAT, conf=0.93))
        return dets


def build_env(workdir: Path) -> Path:
    clip = workdir / "clip.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), CLIP_FPS, (640, 480))
    assert writer.isOpened(), "MJPG encoder unavailable"
    for i in range(CLIP_FRAMES):
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, PERSON_BOX[:2], PERSON_BOX[2:], (90, 90, 90), -1)
        writer.write(frame)
    writer.release()

    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    cfg["cameras"][0]["source"] = str(clip)
    cfg["cameras"][0]["capture_fps"] = CLIP_FPS
    cfg["storage"]["evidence_dir"] = str(workdir / "evidence")
    cfg["storage"]["db_path"] = str(workdir / "test.db")
    # The app's lifespan calls load_dotenv, so a configured .env would make every test run send
    # real alert emails for its synthetic violations. Alerting is covered by test_notify.py.
    cfg["email_alerts"]["enabled"] = False
    path = workdir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


class Harness:
    """Boots the app against a temp workdir with the detector patched out."""

    def __init__(self):
        self.workdir = Path(tempfile.mkdtemp())
        os.environ["EDGESENTINEL_CONFIG"] = str(build_env(self.workdir))
        os.environ["EDGESENTINEL_AUTH_TOKEN"] = TOKEN

        import backend.pipeline as pipeline_module
        import backend.main as main_module

        importlib.reload(pipeline_module)
        importlib.reload(main_module)
        pipeline_module.Detector = FakeDetector

        from fastapi.testclient import TestClient

        self.main = main_module
        self.client = TestClient(main_module.app)

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, *exc):
        self.client.__exit__(*exc)
        shutil.rmtree(self.workdir, ignore_errors=True)

    def wait_for_incident(self, timeout: float = 40.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            response = self.client.get("/api/incidents", params={"token": TOKEN})
            if response.status_code == 200 and response.json()["total"] > 0:
                return response.json()["items"][0]
            time.sleep(0.25)
        raise AssertionError(f"no incident persisted within {timeout}s")


def test_incident_persisted_with_batched_violations():
    with Harness() as h:
        incident = h.wait_for_incident()
        assert set(incident["violations"]) == {NO_HARDHAT, NO_SAFETY_VEST}, incident["violations"]
        assert incident["status"] == "pending", "row must land as pending before enrichment"
        assert incident["track_id"] == 1
        assert incident["confidence"] == 0.91, "confidence is the max across the batch"
        assert incident["duration_seconds"] >= 2.0

        total = h.client.get("/api/incidents", params={"token": TOKEN}).json()["total"]
        assert total == 1, f"batching failed: {total} incidents for one track"


def test_report_id_and_timezone_rendering():
    with Harness() as h:
        incident = h.wait_for_incident()
        import re

        assert re.fullmatch(r"INC-\d{8}-ZONE01-0001", incident["report_id"]), incident["report_id"]
        assert incident["detected_at_utc"].endswith("Z")
        # Asia/Riyadh is UTC+3 (locked decision), so the rendered offset must say so.
        assert incident["detected_at_local"].endswith("+03:00"), incident["detected_at_local"]


def test_evidence_written_and_served():
    with Harness() as h:
        incident = h.wait_for_incident()
        assert incident["evidence_url"], "incident has no evidence_url"

        response = h.client.get(incident["evidence_url"], params={"token": TOKEN})
        assert response.status_code == 200
        assert response.content[:2] == b"\xff\xd8", "evidence is not a JPEG"

        decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape[0] > 0


def test_auth_enforced_on_api_and_media():
    with Harness() as h:
        h.wait_for_incident()
        for path in ("/api/incidents", "/video_feed", "/evidence/anything.jpg"):
            assert h.client.get(path).status_code == 401, f"{path} allowed an unauthenticated read"
            assert h.client.get(path, params={"token": "wrong"}).status_code == 401

        assert h.client.get("/api/incidents", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
        assert h.client.get("/api/health").status_code == 200, "health must stay probe-friendly"


def test_evidence_rejects_path_traversal():
    with Harness() as h:
        response = h.client.get("/evidence/..%2F..%2Fconfig.yaml", params={"token": TOKEN})
        assert response.status_code == 404, "traversal escaped the evidence directory"


def test_mjpeg_generator_emits_jpeg_parts():
    """Drives the generator directly. /video_feed is an endless response, and httpx's ASGI
    transport buffers a whole body, so a TestClient GET against it would never return."""
    import asyncio

    from backend.stream import MEDIA_TYPE, mjpeg_stream

    assert "multipart/x-mixed-replace" in MEDIA_TYPE

    with Harness() as h:
        h.wait_for_incident()

        async def collect() -> list[bytes]:
            generator = mjpeg_stream(h.main.app.state.pipeline, 15, 1280, 720)
            chunks = []
            async for chunk in generator:
                chunks.append(chunk)
                if len(chunks) >= 2:
                    break
            await generator.aclose()
            return chunks

        parts = asyncio.run(collect())

    assert len(parts) == 2
    for part in parts:
        assert part.startswith(b"--frame\r\n")
        assert b"Content-Type: image/jpeg" in part
        body = part.split(b"\r\n\r\n", 1)[1]
        assert body[:2] == b"\xff\xd8", "part body is not a JPEG"
        decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape[:2] == (720, 1280), "stream is not 720p"


def test_websocket_replays_history_on_connect():
    with Harness() as h:
        h.wait_for_incident()
        # The clip has ended by now, so replay is the only guaranteed traffic — read exactly
        # what replay owes us rather than blocking on a live event that will never arrive.
        with h.client.websocket_connect(f"/ws/events?token={TOKEN}") as ws:
            message = json.loads(ws.receive_text())

        assert message["type"] == "incident.created", message["type"]
        assert set(message["data"]["violations"]) == {NO_HARDHAT, NO_SAFETY_VEST}
        assert message["data"]["status"] == "pending"


def test_websocket_rejects_bad_token():
    from starlette.websockets import WebSocketDisconnect

    with Harness() as h:
        try:
            with h.client.websocket_connect("/ws/events?token=nope") as ws:
                ws.receive_text()
            raise AssertionError("bad token was accepted")
        except WebSocketDisconnect as exc:
            assert exc.code == 1008, f"unexpected close code {exc.code}"


def test_detections_table_populated():
    with Harness() as h:
        h.wait_for_incident()
        time.sleep(0.5)
        db_path = h.workdir / "test.db"
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM detections ORDER BY id DESC LIMIT 5"
            ).fetchall()
            cameras = conn.execute("SELECT COUNT(*) AS n FROM cameras").fetchone()["n"]

        assert cameras == 1, "camera row not upserted"
        assert rows, "no detection rows persisted"
        for row in rows:
            assert json.loads(row["violations"]), "only non-compliant tracks should be stored"
            assert len(json.loads(row["bbox"])) == 4


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
        except Exception as exc:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
