"""SQLite schema and async access.

Uses aiosqlite rather than SQLAlchemy: the access pattern is a handful of hand-written statements,
and keeping the dependency surface small matters more here than an ORM would buy.

Ordering guarantee from spec §5: the incident row is INSERTed with status='pending' the moment the
violation fires. Enrichment is a later UPDATE. The audit trail never depends on the API.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiosqlite

from config import CameraConfig, Config

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id       TEXT PRIMARY KEY,
    zone_id         TEXT NOT NULL,
    zone_label_en   TEXT NOT NULL,
    zone_label_ar   TEXT NOT NULL,
    source          TEXT NOT NULL,
    last_seen_utc   TEXT
);

CREATE TABLE IF NOT EXISTS incidents (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id         TEXT UNIQUE NOT NULL,
    camera_id         TEXT NOT NULL,
    zone_id           TEXT NOT NULL,
    track_id          INTEGER NOT NULL,
    violations        TEXT NOT NULL,
    confidence        REAL NOT NULL,
    duration_seconds  REAL NOT NULL,
    detected_at_utc   TEXT NOT NULL,
    evidence_path     TEXT,

    status            TEXT NOT NULL DEFAULT 'pending',
    attempts          INTEGER NOT NULL DEFAULT 0,
    last_error        TEXT,

    risk_level        TEXT,
    recommended_protocol TEXT,
    summary_en        TEXT,
    report_ar         TEXT,
    model_used        TEXT,
    enriched_at_utc   TEXT,

    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
);

CREATE TABLE IF NOT EXISTS detections (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id         TEXT NOT NULL,
    ts_utc            TEXT NOT NULL,
    track_id          INTEGER NOT NULL,
    bbox              TEXT NOT NULL,
    conf              REAL NOT NULL,
    violations        TEXT NOT NULL,
    violation_elapsed REAL NOT NULL,
    incident_id       INTEGER,
    FOREIGN KEY (incident_id) REFERENCES incidents(id)
);

CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_time   ON incidents(detected_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_detections_time  ON detections(ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(camera_id, track_id);
"""


class Database:
    def __init__(self, path: Path, display_timezone: str = "Asia/Riyadh"):
        self.path = path
        self.tz = ZoneInfo(display_timezone)
        self._conn: aiosqlite.Connection | None = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("database not connected")
        return self._conn

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        # WAL lets the MJPEG/WS readers proceed while the pipeline writes.
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        log.info("sqlite ready at %s", self.path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def upsert_camera(self, cam: CameraConfig) -> None:
        await self.conn.execute(
            """INSERT INTO cameras (camera_id, zone_id, zone_label_en, zone_label_ar, source)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(camera_id) DO UPDATE SET
                   zone_id=excluded.zone_id,
                   zone_label_en=excluded.zone_label_en,
                   zone_label_ar=excluded.zone_label_ar,
                   source=excluded.source""",
            (cam.camera_id, cam.zone_id, cam.zone_label_en, cam.zone_label_ar, str(cam.source)),
        )
        await self.conn.commit()

    async def mark_camera_seen(self, camera_id: str) -> None:
        await self.conn.execute(
            "UPDATE cameras SET last_seen_utc = ? WHERE camera_id = ?",
            (_now_iso(), camera_id),
        )
        await self.conn.commit()

    async def next_report_id(self, zone_code: str, detected_at: datetime) -> str:
        """Sequence per (day, zone). The UNIQUE constraint on report_id is the real guard."""
        prefix = f"INC-{detected_at.strftime('%Y%m%d')}-{zone_code}-"
        async with self.conn.execute(
            "SELECT report_id FROM incidents WHERE report_id LIKE ? ORDER BY report_id DESC LIMIT 1",
            (f"{prefix}%",),
        ) as cursor:
            row = await cursor.fetchone()

        seq = 0
        if row is not None:
            try:
                seq = int(row["report_id"][len(prefix):])
            except ValueError:
                seq = 0
        return f"{prefix}{seq + 1:04d}"

    async def insert_incident(
        self,
        report_id: str,
        camera_id: str,
        zone_id: str,
        track_id: int,
        violations: list[str],
        confidence: float,
        duration_seconds: float,
        detected_at_utc: datetime,
        evidence_path: str | None,
    ) -> int:
        cursor = await self.conn.execute(
            """INSERT INTO incidents
               (report_id, camera_id, zone_id, track_id, violations, confidence,
                duration_seconds, detected_at_utc, evidence_path, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
            (
                report_id, camera_id, zone_id, track_id, json.dumps(violations),
                confidence, duration_seconds, _iso(detected_at_utc), evidence_path,
            ),
        )
        await self.conn.commit()
        return cursor.lastrowid

    async def insert_detections(self, rows: list[tuple]) -> None:
        if not rows:
            return
        await self.conn.executemany(
            """INSERT INTO detections
               (camera_id, ts_utc, track_id, bbox, conf, violations, violation_elapsed, incident_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        await self.conn.commit()

    async def bump_attempts(self, report_id: str) -> int:
        await self.conn.execute(
            "UPDATE incidents SET attempts = attempts + 1 WHERE report_id = ?", (report_id,)
        )
        await self.conn.commit()
        async with self.conn.execute(
            "SELECT attempts FROM incidents WHERE report_id = ?", (report_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return row["attempts"] if row else 0

    async def mark_enriched(
        self,
        report_id: str,
        risk_level: str,
        recommended_protocol: str,
        summary_en: str,
        report_ar: str,
        model_used: str,
    ) -> None:
        await self.conn.execute(
            """UPDATE incidents SET
                   status = 'enriched', risk_level = ?, recommended_protocol = ?,
                   summary_en = ?, report_ar = ?, model_used = ?, enriched_at_utc = ?,
                   last_error = NULL
               WHERE report_id = ?""",
            (
                risk_level, recommended_protocol, summary_en, report_ar,
                model_used, _now_iso(), report_id,
            ),
        )
        await self.conn.commit()

    async def mark_failed(self, report_id: str, error: str) -> None:
        await self.conn.execute(
            "UPDATE incidents SET status = 'failed', last_error = ? WHERE report_id = ?",
            (error[:500], report_id),
        )
        await self.conn.commit()

    async def note_error(self, report_id: str, error: str) -> None:
        """Record a transient failure without leaving 'pending' — the row stays retryable."""
        await self.conn.execute(
            "UPDATE incidents SET last_error = ? WHERE report_id = ?",
            (error[:500], report_id),
        )
        await self.conn.commit()

    async def list_incidents(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        camera_id: str | None = None,
    ) -> tuple[int, list[dict]]:
        where, params = [], []
        if status:
            where.append("status = ?")
            params.append(status)
        if camera_id:
            where.append("camera_id = ?")
            params.append(camera_id)
        clause = f"WHERE {' AND '.join(where)}" if where else ""

        async with self.conn.execute(
            f"SELECT COUNT(*) AS n FROM incidents {clause}", params
        ) as cursor:
            total = (await cursor.fetchone())["n"]

        async with self.conn.execute(
            f"""SELECT * FROM incidents {clause}
                ORDER BY detected_at_utc DESC, id DESC LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        ) as cursor:
            rows = await cursor.fetchall()

        return total, [self.row_to_dict(r) for r in rows]

    async def get_incident(self, report_id: str) -> dict | None:
        async with self.conn.execute(
            "SELECT * FROM incidents WHERE report_id = ?", (report_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return self.row_to_dict(row) if row else None

    async def recent_incidents(self, limit: int) -> list[dict]:
        _, items = await self.list_incidents(limit=limit)
        return list(reversed(items))

    async def pending_incidents(self) -> list[dict]:
        """Startup re-queue: rows left 'pending' by a previous run (spec §5)."""
        async with self.conn.execute(
            "SELECT * FROM incidents WHERE status = 'pending' ORDER BY detected_at_utc"
        ) as cursor:
            rows = await cursor.fetchall()
        return [self.row_to_dict(r) for r in rows]

    async def count_pending(self) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM incidents WHERE status = 'pending'"
        ) as cursor:
            return (await cursor.fetchone())["n"]

    def row_to_dict(self, row: aiosqlite.Row) -> dict:
        data = dict(row)
        data["violations"] = json.loads(data["violations"])
        evidence = data.pop("evidence_path", None)
        data["evidence_url"] = f"/evidence/{Path(evidence).name}" if evidence else None
        data["detected_at_local"] = self.to_local(data["detected_at_utc"])
        return data

    def to_local(self, iso_utc_value: str) -> str:
        """Store UTC, render Asia/Riyadh (spec §1). Never a naive datetime."""
        parsed = datetime.fromisoformat(iso_utc_value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(self.tz).isoformat()


def _now_iso() -> str:
    return _iso(datetime.now(timezone.utc))


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def open_database(cfg: Config) -> Database:
    from config import REPO_ROOT

    path = Path(cfg.storage.db_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return Database(path, cfg.locale.display_timezone)
