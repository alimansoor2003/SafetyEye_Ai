"""Queue consumer pool (spec §7, M3).

Workers own all LLM traffic. The detection loop enqueues a report_id and moves on; nothing in the
vision path ever waits on the network.

Retry policy across restarts: a transient failure leaves the row 'pending', so the startup
re-queue picks it up again. `max_attempts` bounds that — an incident that has failed transiently
many times is marked 'failed' rather than retried forever at every boot.
"""
from __future__ import annotations

import asyncio
import logging
import time

from agent.client import AgentRejected, AgentUnavailable, HSEAgent, QuotaExhausted

log = logging.getLogger(__name__)

RETRY_DELAY_SECONDS = 30.0
MAX_ATTEMPTS = 6
# How long a paused worker sleeps before re-checking, so a long quota pause does not hold a
# worker hostage past a restart or a config change.
PAUSE_POLL_SECONDS = 10.0


class EnrichmentPool:
    def __init__(self, agent: HSEAgent, db, manager, camera, worker_count: int = 2):
        self.agent = agent
        self.db = db
        self.manager = manager
        self.camera = camera
        self.worker_count = max(1, worker_count)
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._retries: set[asyncio.Task] = set()
        self._queued: set[str] = set()
        self._paused_until = 0.0

    @property
    def depth(self) -> int:
        return self.queue.qsize()

    @property
    def paused_for(self) -> float:
        """Seconds until enrichment resumes; 0 when running. Surfaced on /api/health."""
        return max(0.0, self._paused_until - time.monotonic())

    def _pause(self, seconds: float) -> None:
        until = time.monotonic() + seconds
        if until > self._paused_until:
            self._paused_until = until
            log.error(
                "enrichment paused for %.0fs — API budget exhausted. Incidents keep recording "
                "and stay 'pending'; they will enrich when the budget resets.", seconds,
            )

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._worker(i + 1), name=f"enrich-{i + 1}")
            for i in range(self.worker_count)
        ]
        log.info("enrichment pool started with %d worker(s)", self.worker_count)

    async def stop(self) -> None:
        for task in [*self._tasks, *self._retries]:
            task.cancel()
        for task in [*self._tasks, *self._retries]:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        self._retries.clear()

    def submit(self, report_id: str) -> None:
        """Dedupe: an incident already waiting must not be enqueued twice (spec §7 cooldown)."""
        if report_id in self._queued:
            return
        self._queued.add(report_id)
        self.queue.put_nowait(report_id)

    async def requeue_pending(self) -> int:
        rows = await self.db.pending_incidents()
        for row in rows:
            self.submit(row["report_id"])
        if rows:
            log.info("re-queued %d pending incident(s) from a previous run", len(rows))
        return len(rows)

    async def _worker(self, index: int) -> None:
        while True:
            report_id = await self.queue.get()
            requeue = False
            try:
                wait = self.paused_for
                if wait > 0:
                    await asyncio.sleep(min(wait, PAUSE_POLL_SECONDS))
                    requeue = True
                else:
                    requeue = await self._process(report_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("worker %d crashed handling %s", index, report_id)
            finally:
                self._queued.discard(report_id)
                self.queue.task_done()
                if requeue:
                    self.submit(report_id)

    async def _process(self, report_id: str) -> bool:
        """Returns True when the incident should go straight back on the queue."""
        incident = await self.db.get_incident(report_id)
        if incident is None:
            log.warning("incident %s vanished before enrichment", report_id)
            return False
        if incident["status"] == "enriched":
            return False

        try:
            result = await self.agent.enrich(
                incident, self.camera.zone_label_en, self.camera.zone_label_ar
            )
        except QuotaExhausted as exc:
            # Deliberately not counted as an attempt: nothing is wrong with this incident, and
            # letting a billing state exhaust MAX_ATTEMPTS would discard a valid HSE record.
            self._pause(exc.retry_after)
            await self.db.note_error(report_id, f"awaiting API budget: {exc}")
            return True
        except AgentRejected as exc:
            attempts = await self.db.bump_attempts(report_id)
            log.error("%s permanently rejected after %d attempt(s): %s", report_id, attempts, exc)
            await self.db.mark_failed(report_id, str(exc))
            await self._broadcast_failure(report_id, str(exc))
            return False
        except AgentUnavailable as exc:
            attempts = await self.db.bump_attempts(report_id)
            if attempts >= MAX_ATTEMPTS:
                log.error("%s giving up after %d attempts: %s", report_id, attempts, exc)
                await self.db.mark_failed(report_id, f"unavailable after {attempts} attempts: {exc}")
                await self._broadcast_failure(report_id, str(exc))
                return False
            # Stays 'pending' on purpose: the audit trail must survive a dead network.
            await self.db.note_error(report_id, str(exc))
            log.warning("%s still pending (attempt %d): %s", report_id, attempts, exc)
            self._schedule_retry(report_id)
            return False

        await self.db.bump_attempts(report_id)

        report = result.report
        await self.db.mark_enriched(
            report_id=report_id,
            risk_level=report.risk_level,
            recommended_protocol=report.recommended_protocol,
            summary_en=report.summary_en,
            report_ar=report.report_ar,
            model_used=result.model_used,
        )
        log.info("%s enriched by %s (%s)", report_id, result.model_used, report.risk_level)

        await self.manager.broadcast("incident.enriched", {
            "report_id": report_id,
            "status": "enriched",
            "risk_level": report.risk_level,
            "recommended_protocol": report.recommended_protocol,
            "summary_en": report.summary_en,
            "report_ar": report.report_ar,
            "model_used": result.model_used,
        })
        return False

    def _schedule_retry(self, report_id: str) -> None:
        """Tracked so shutdown cancels it — an untracked task would outlive the pool."""
        task = asyncio.create_task(self._retry_later(report_id))
        self._retries.add(task)
        task.add_done_callback(self._retries.discard)

    async def _retry_later(self, report_id: str) -> None:
        await asyncio.sleep(RETRY_DELAY_SECONDS)
        self.submit(report_id)

    async def _broadcast_failure(self, report_id: str, error: str) -> None:
        await self.manager.broadcast("incident.enriched", {
            "report_id": report_id,
            "status": "failed",
            "last_error": error[:200],
        })
