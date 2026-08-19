"""FastAPI application: lifespan, WebSocket endpoint, and the async drain task."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.client import HSEAgent
from agent.worker import EnrichmentPool
from backend.db import open_database
from backend.pipeline import FrameEvent, IncidentDraft, VisionPipeline, zone_code
from backend.routes import check_token, expected_token, router
from backend.schemas import SystemStatus
from backend.ws import ConnectionManager
from config import load_config

log = logging.getLogger("backend")

HEARTBEAT_SECONDS = 5.0
QUEUE_MAXSIZE = 512


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    cfg = load_config(os.getenv("EDGESENTINEL_CONFIG") or REPO_ROOT / "config.yaml")
    if expected_token() is None:
        log.warning(
            "EDGESENTINEL_AUTH_TOKEN is not set — /api, /video_feed, /ws/events and /evidence "
            "are UNAUTHENTICATED. Set it in .env before the demo."
        )

    camera = cfg.cameras[0]
    if len(cfg.cameras) > 1:
        log.warning("%d cameras configured; v1 serves only %s", len(cfg.cameras), camera.camera_id)

    evidence_dir = REPO_ROOT / cfg.storage.evidence_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)

    db = open_database(cfg)
    await db.connect()
    await db.upsert_camera(camera)

    app.state.config = cfg
    app.state.camera = camera
    app.state.db = db
    app.state.evidence_dir = evidence_dir
    app.state.manager = ConnectionManager()
    app.state.queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    app.state.pipeline = VisionPipeline(cfg, camera, evidence_dir)

    app.state.pool = _build_pool(cfg, db, app.state.manager, camera)
    if app.state.pool is not None:
        app.state.pool.start()
        await app.state.pool.requeue_pending()
    else:
        pending = await db.count_pending()
        if pending:
            log.warning("%d incident(s) pending but enrichment is disabled", pending)

    loop = asyncio.get_running_loop()
    app.state.pipeline.start(loop, app.state.queue)
    app.state.tasks = [
        asyncio.create_task(_drain(app), name="drain"),
        asyncio.create_task(_heartbeat(app), name="heartbeat"),
    ]
    log.info("listening on http://%s:%d", cfg.server.host, cfg.server.port)

    try:
        yield
    finally:
        app.state.pipeline.stop()
        if app.state.pool is not None:
            await app.state.pool.stop()
        for task in app.state.tasks:
            task.cancel()
        for task in app.state.tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await db.close()
        log.info("shutdown complete")


app = FastAPI(title="safetyeye AI", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo-scoped: the dashboard is served from the same host
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.include_router(router)


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket, token: str | None = Query(default=None)) -> None:
    supplied = token
    header = websocket.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        supplied = header[7:].strip()
    if not check_token(supplied):
        await websocket.close(code=1008, reason="invalid or missing token")
        return

    manager: ConnectionManager = websocket.app.state.manager
    await websocket.accept()
    try:
        # History first, then live traffic — see ConnectionManager.register.
        await _replay_history(websocket)
        await manager.register(websocket)
        while True:
            # No client->server protocol; receiving is how we notice a disconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        log.debug("ws error", exc_info=True)
    finally:
        await manager.disconnect(websocket)


async def _replay_history(websocket: WebSocket) -> None:
    """Replay recent incidents so a client that connects late still renders the board."""
    app = websocket.app
    manager: ConnectionManager = app.state.manager
    for incident in await app.state.db.recent_incidents(app.state.config.server.history_replay):
        await manager.send(websocket, "incident.created", {
            "report_id": incident["report_id"],
            "camera_id": incident["camera_id"],
            "zone_id": incident["zone_id"],
            "track_id": incident["track_id"],
            "violations": incident["violations"],
            "confidence": incident["confidence"],
            "duration_seconds": incident["duration_seconds"],
            "detected_at_utc": incident["detected_at_utc"],
            "evidence_url": incident["evidence_url"],
            "status": incident["status"],
        })
        if incident["status"] == "enriched":
            await manager.send(websocket, "incident.enriched", {
                "report_id": incident["report_id"],
                "status": incident["status"],
                "risk_level": incident["risk_level"],
                "recommended_protocol": incident["recommended_protocol"],
                "summary_en": incident["summary_en"],
                "report_ar": incident["report_ar"],
                "model_used": incident["model_used"],
            })


def _build_pool(cfg, db, manager, camera) -> EnrichmentPool | None:
    """A missing key disables enrichment rather than killing the app.

    Spec §11 risk 5: if the model API is unreachable at the booth the dashboard must still look
    alive — incidents appear and badges flip. Refusing to boot would fail that outright.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not gemini_key and not anthropic_key:
        log.warning(
            "No GEMINI_API_KEY or ANTHROPIC_API_KEY — incidents will be recorded but never "
            "enriched. Set one in .env to enable the HSE agent."
        )
        return None
    if not anthropic_key:
        log.warning(
            "ANTHROPIC_API_KEY is not set — no cross-provider fallback. Enrichment will pause "
            "when the Gemini budget is spent."
        )
    try:
        agent = HSEAgent(cfg.agent, gemini_key, anthropic_key, cfg.locale.display_timezone)
    except Exception:
        log.exception("could not build the HSE agent — enrichment disabled")
        return None
    return EnrichmentPool(agent, db, manager, camera, cfg.agent.worker_count)


async def _drain(app: FastAPI) -> None:
    """Single consumer of the vision thread's hand-off queue."""
    db = app.state.db
    manager: ConnectionManager = app.state.manager
    camera = app.state.camera
    evidence_dir: Path = app.state.evidence_dir

    while True:
        item = await app.state.queue.get()
        try:
            if isinstance(item, FrameEvent):
                await manager.broadcast("detection.frame", item.payload)
                if item.detection_rows:
                    await db.insert_detections(item.detection_rows)

            elif isinstance(item, IncidentDraft):
                report_id = await _persist_incident(db, manager, camera, evidence_dir, item)
                if app.state.pool is not None:
                    app.state.pool.submit(report_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("drain failed to handle %s", type(item).__name__)
        finally:
            app.state.queue.task_done()


async def _persist_incident(db, manager, camera, evidence_dir: Path, draft: IncidentDraft) -> str:
    """Row and evidence are written BEFORE any enrichment — spec §5's critical ordering."""
    report_id = await db.next_report_id(zone_code(camera.zone_id), draft.detected_at_utc)

    evidence_path: str | None = None
    if draft.evidence_jpeg:
        target = evidence_dir / f"{report_id}.jpg"
        await asyncio.to_thread(target.write_bytes, draft.evidence_jpeg)
        evidence_path = str(target)

    await db.insert_incident(
        report_id=report_id,
        camera_id=camera.camera_id,
        zone_id=camera.zone_id,
        track_id=draft.track_id,
        violations=draft.violations,
        confidence=draft.confidence,
        duration_seconds=draft.duration_seconds,
        detected_at_utc=draft.detected_at_utc,
        evidence_path=evidence_path,
    )
    await db.mark_camera_seen(camera.camera_id)

    log.info("incident %s track=%d %s", report_id, draft.track_id, draft.violations)
    await manager.broadcast("incident.created", {
        "report_id": report_id,
        "camera_id": camera.camera_id,
        "zone_id": camera.zone_id,
        "track_id": draft.track_id,
        "violations": draft.violations,
        "confidence": round(draft.confidence, 3),
        "duration_seconds": draft.duration_seconds,
        "detected_at_utc": draft.detected_at_utc.isoformat().replace("+00:00", "Z"),
        "evidence_url": f"/evidence/{report_id}.jpg" if evidence_path else None,
        "status": "pending",
    })
    return report_id


async def _heartbeat(app: FastAPI) -> None:
    manager: ConnectionManager = app.state.manager
    pipeline = app.state.pipeline
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        pool = app.state.pool
        status = SystemStatus(
            fps=pipeline.fps,
            gpu=pipeline.gpu_name,
            queue_depth=app.state.queue.qsize() + (pool.depth if pool else 0),
            clients=manager.count,
            pending_incidents=await app.state.db.count_pending(),
            enrichment_paused_seconds=round(pool.paused_for, 1) if pool else 0.0,
            camera_connected=pipeline.camera_connected,
        )
        await manager.broadcast("system.status", status.model_dump())


def main() -> int:
    import uvicorn

    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(os.getenv("EDGESENTINEL_CONFIG") or REPO_ROOT / "config.yaml")
    uvicorn.run(
        "backend.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
