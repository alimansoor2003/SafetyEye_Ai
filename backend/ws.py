"""WebSocket fan-out. Carries detections and incidents only — never frames (spec §1)."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket

log = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self, send_timeout: float = 2.0):
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self.send_timeout = send_timeout

    @property
    def count(self) -> int:
        return len(self._clients)

    async def register(self, websocket: WebSocket) -> None:
        """Start delivering broadcasts to an already-accepted socket.

        Registration is deliberately separate from accept so a caller can finish replaying
        history first. Registering earlier lets a live event overtake the backlog, which shows
        up on the dashboard as an incident card appearing above older ones.
        """
        async with self._lock:
            self._clients.add(websocket)
        log.info("ws client connected (%d total)", self.count)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)
        log.info("ws client disconnected (%d total)", self.count)

    async def send(self, websocket: WebSocket, type_: str, data: dict) -> None:
        await websocket.send_text(json.dumps({"type": type_, "data": data}, ensure_ascii=False))

    async def broadcast(self, type_: str, data: dict) -> None:
        """A stalled client must never block the detection pipeline, so sends are bounded."""
        async with self._lock:
            targets = list(self._clients)
        if not targets:
            return

        payload = json.dumps({"type": type_, "data": data}, ensure_ascii=False)
        results = await asyncio.gather(
            *(self._send_one(client, payload) for client in targets), return_exceptions=True
        )
        dead = [client for client, ok in zip(targets, results) if ok is not True]
        if dead:
            async with self._lock:
                for client in dead:
                    self._clients.discard(client)
            log.warning("dropped %d unresponsive ws client(s)", len(dead))

    async def _send_one(self, client: WebSocket, payload: str) -> bool:
        try:
            await asyncio.wait_for(client.send_text(payload), timeout=self.send_timeout)
            return True
        except Exception:
            return False
