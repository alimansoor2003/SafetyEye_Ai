from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import ViolationConfig

log = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "IDLE"
    ARMING = "ARMING"
    FIRED = "FIRED"


@dataclass
class ViolationEvent:
    track_id: int
    violation_type: str
    confidence: float
    duration_seconds: float
    detected_at_utc: datetime
    suppressed_since_last: int


@dataclass
class _Entry:
    state: State = State.IDLE
    armed_at: float | None = None
    clean_since: float | None = None
    last_seen_at: float = 0.0
    cooldown_until: float = 0.0
    suppressed: int = 0
    touched_at: float = 0.0
    breaker_logged: bool = False


@dataclass
class Stats:
    tracks_seen: int = 0
    events_emitted: int = 0
    events_suppressed_cooldown: int = 0
    events_suppressed_breaker: int = 0
    arming_discarded_track_lost: int = 0
    _known_tracks: set[int] = field(default_factory=set)


class ViolationStateMachine:
    """Per (track_id, violation_type) timers implementing the IDLE/ARMING/FIRED contract."""

    def __init__(self, cfg: ViolationConfig):
        self.cfg = cfg
        self._entries: dict[tuple[int, str], _Entry] = {}
        self._recent_emits: deque[float] = deque()
        self.stats = Stats()

    def update(
        self,
        now: float,
        active: dict[int, dict[str, float]],
        seen_track_ids: set[int],
    ) -> list[ViolationEvent]:
        """`now` is a monotonic clock. `active` maps track_id -> {violation_type: confidence}."""
        for tid in seen_track_ids:
            if tid not in self.stats._known_tracks:
                self.stats._known_tracks.add(tid)
                self.stats.tracks_seen += 1

        for tid, violations in active.items():
            for vtype in violations:
                self._entries.setdefault((tid, vtype), _Entry(last_seen_at=now))

        events: list[ViolationEvent] = []
        for key, entry in self._entries.items():
            tid, vtype = key
            conf = active.get(tid, {}).get(vtype)
            if tid in seen_track_ids:
                entry.last_seen_at = now
            entry.touched_at = now

            if conf is not None:
                event = self._step_violating(entry, tid, vtype, conf, now)
                if event is not None:
                    events.append(event)
            else:
                self._step_clean(entry, tid, vtype, now, seen=tid in seen_track_ids)

        self._gc(now)
        return events

    def _step_violating(
        self, entry: _Entry, tid: int, vtype: str, conf: float, now: float
    ) -> ViolationEvent | None:
        entry.clean_since = None

        if entry.state is State.IDLE:
            entry.state = State.ARMING
            entry.armed_at = now
            return None

        if entry.armed_at is None:
            entry.armed_at = now

        elapsed = now - entry.armed_at

        if entry.state is State.ARMING and elapsed < self.cfg.persist_seconds:
            return None

        if now < entry.cooldown_until:
            entry.state = State.FIRED
            entry.suppressed += 1
            self.stats.events_suppressed_cooldown += 1
            return None

        if not self._breaker_allows(now):
            entry.state = State.FIRED
            self.stats.events_suppressed_breaker += 1
            if not entry.breaker_logged:
                entry.breaker_logged = True
                log.warning("circuit breaker: >%d incidents/min, suppressing track %d %s",
                            self.cfg.max_incidents_per_minute, tid, vtype)
            return None

        suppressed = entry.suppressed
        entry.state = State.FIRED
        entry.breaker_logged = False
        entry.cooldown_until = now + self.cfg.cooldown_seconds
        entry.suppressed = 0
        self._recent_emits.append(now)
        self.stats.events_emitted += 1

        return ViolationEvent(
            track_id=tid,
            violation_type=vtype,
            confidence=conf,
            duration_seconds=round(elapsed, 2),
            detected_at_utc=datetime.now(timezone.utc),
            suppressed_since_last=suppressed,
        )

    def _step_clean(self, entry: _Entry, tid: int, vtype: str, now: float, seen: bool) -> None:
        if not seen:
            if now - entry.last_seen_at > self.cfg.track_lost_seconds:
                if entry.state is State.ARMING:
                    self.stats.arming_discarded_track_lost += 1
                    log.debug("track %d lost while ARMING on %s — timer discarded", tid, vtype)
                entry.state = State.IDLE
                entry.armed_at = None
                entry.clean_since = None
            return

        if entry.clean_since is None:
            entry.clean_since = now
        elif now - entry.clean_since >= self.cfg.clear_seconds:
            entry.state = State.IDLE
            entry.armed_at = None

    def elapsed(self, now: float, track_id: int, violation_type: str) -> float:
        entry = self._entries.get((track_id, violation_type))
        if entry is None or entry.armed_at is None:
            return 0.0
        return round(now - entry.armed_at, 2)

    def _breaker_allows(self, now: float) -> bool:
        cutoff = now - 60.0
        while self._recent_emits and self._recent_emits[0] < cutoff:
            self._recent_emits.popleft()
        return len(self._recent_emits) < self.cfg.max_incidents_per_minute

    def _gc(self, now: float) -> None:
        horizon = self.cfg.cooldown_seconds + 60.0
        stale = [
            k for k, e in self._entries.items()
            if now - e.touched_at > horizon and now >= e.cooldown_until
        ]
        for k in stale:
            del self._entries[k]
