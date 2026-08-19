"""Wire contracts from spec §6, with amendment A2 (violations is an array)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

IncidentStatus = Literal["pending", "enriched", "failed"]
RiskLevel = Literal["Low", "Medium", "High", "Critical"]

# `model_used` is a real column name from spec §5; pydantic's "model_" namespace guard would
# otherwise warn on every import.
ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


def iso_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


class TrackView(BaseModel):
    track_id: int
    bbox: list[int]
    label: str = "Person"
    conf: float
    violations: list[str] = Field(default_factory=list)
    violation_elapsed: float = 0.0


class DetectionFrame(BaseModel):
    camera_id: str
    ts_utc: str
    compliant: bool
    tracks: list[TrackView]


class IncidentCreated(BaseModel):
    report_id: str
    camera_id: str
    zone_id: str
    track_id: int
    violations: list[str]
    confidence: float
    duration_seconds: float
    detected_at_utc: str
    evidence_url: str | None = None
    status: IncidentStatus = "pending"


class IncidentEnriched(BaseModel):
    model_config = ALLOW_MODEL_PREFIX

    report_id: str
    status: IncidentStatus
    risk_level: RiskLevel | None = None
    recommended_protocol: str | None = None
    summary_en: str | None = None
    report_ar: str | None = None
    model_used: str | None = None


class SystemStatus(BaseModel):
    fps: float | None = None
    gpu: str | None = None
    queue_depth: int = 0
    clients: int = 0
    pending_incidents: int = 0
    enrichment_paused_seconds: float = 0.0
    camera_connected: bool = False
    mqtt_connected: bool = False
    esp32_last_seen_utc: str | None = None


class Incident(BaseModel):
    """A full row from the incidents table, as returned by /api/incidents."""

    model_config = ALLOW_MODEL_PREFIX

    id: int
    report_id: str
    camera_id: str
    zone_id: str
    track_id: int
    violations: list[str]
    confidence: float
    duration_seconds: float
    detected_at_utc: str
    detected_at_local: str
    evidence_url: str | None = None
    status: IncidentStatus
    attempts: int = 0
    last_error: str | None = None
    risk_level: RiskLevel | None = None
    recommended_protocol: str | None = None
    summary_en: str | None = None
    report_ar: str | None = None
    model_used: str | None = None
    enriched_at_utc: str | None = None


class IncidentPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[Incident]


class Envelope(BaseModel):
    """Every WebSocket message: {"type": ..., "data": {...}}."""

    type: str
    data: dict[str, Any]

    @classmethod
    def of(cls, type_: str, model: BaseModel) -> "Envelope":
        return cls(type=type_, data=model.model_dump())
