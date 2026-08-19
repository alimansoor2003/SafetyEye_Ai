"""REST + media routes. Auth accepts a bearer header or a ?token= query parameter.

The query form is not laziness: an MJPEG <img src> and a browser WebSocket cannot set request
headers, so a header-only scheme would leave /video_feed and /ws/events unauthenticated.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse

from backend.schemas import IncidentPage, SystemStatus
from backend.stream import MEDIA_TYPE, mjpeg_stream

log = logging.getLogger(__name__)
router = APIRouter()


def expected_token() -> str | None:
    token = os.getenv("EDGESENTINEL_AUTH_TOKEN", "").strip()
    return token or None


def check_token(supplied: str | None) -> bool:
    expected = expected_token()
    if expected is None:
        return True  # auth disabled; startup logs a warning
    return bool(supplied) and supplied == expected


async def require_token(request: Request, token: str | None = Query(default=None)) -> None:
    supplied = token
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        supplied = header[7:].strip()
    if not check_token(supplied):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing token")


@router.get("/api/health", response_model=SystemStatus)
async def health(request: Request) -> SystemStatus:
    """Unauthenticated on purpose: a liveness probe that needs a secret is not a liveness probe."""
    app = request.app
    pipeline = app.state.pipeline
    pool = app.state.pool
    return SystemStatus(
        fps=pipeline.fps,
        gpu=pipeline.gpu_name,
        queue_depth=app.state.queue.qsize() + (pool.depth if pool else 0),
        clients=app.state.manager.count,
        pending_incidents=await app.state.db.count_pending(),
        enrichment_paused_seconds=round(pool.paused_for, 1) if pool else 0.0,
        camera_connected=pipeline.camera_connected,
    )


@router.get("/api/incidents", response_model=IncidentPage, dependencies=[Depends(require_token)])
async def list_incidents(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status_filter: str | None = Query(default=None, alias="status"),
    camera_id: str | None = Query(default=None),
) -> IncidentPage:
    total, items = await request.app.state.db.list_incidents(
        limit=limit, offset=offset, status=status_filter, camera_id=camera_id
    )
    return IncidentPage(total=total, limit=limit, offset=offset, items=items)


@router.get("/api/incidents/{report_id}", dependencies=[Depends(require_token)])
async def get_incident(request: Request, report_id: str) -> dict:
    incident = await request.app.state.db.get_incident(report_id)
    if incident is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no incident {report_id}")
    return incident


@router.get("/video_feed", dependencies=[Depends(require_token)])
async def video_feed(request: Request) -> StreamingResponse:
    stream = request.app.state.config.stream
    return StreamingResponse(
        mjpeg_stream(request.app.state.pipeline, stream.fps, stream.width, stream.height),
        media_type=MEDIA_TYPE,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@router.get("/evidence/{filename}", dependencies=[Depends(require_token)])
async def evidence(request: Request, filename: str) -> FileResponse:
    directory: Path = request.app.state.evidence_dir
    # Resolve and confine: never let a crafted name escape the evidence directory.
    target = (directory / filename).resolve()
    if not str(target).startswith(str(directory.resolve())) or not target.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such evidence file")
    return FileResponse(target, media_type="image/jpeg")
