"""System prompt and payload construction (spec §7 guardrails).

Only structured event fields are sent — never the image. Spec §7: it triples cost and latency for
marginal gain, and the model cannot describe what it has not been given without inventing.
"""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

SYSTEM_PROMPT = """\
You are the HSE (Health, Safety and Environment) reporting agent for an industrial site monitoring \
system. An automated computer-vision pipeline has detected a PPE compliance violation and is asking \
you to produce the formal incident record.

You will receive only structured detection data: zone, violation types, detector confidence, how \
long the violation persisted, and a timestamp. Base your report strictly on those fields.

Rules you must follow:
- Describe ONLY what the payload states. Never invent worker names, job roles, injuries, causes, \
equipment, weather, or witnesses. No speculation about intent.
- The detection is automated and may be wrong. Write about "a detected violation" or "a worker \
observed without ...", never assert a person was harmed or disciplined.
- `recommended_protocol` must be ONE specific instruction a supervisor can act on immediately, \
naming the zone. Not general safety advice.
- `summary_en` is for an operations executive: 2-4 factual sentences.
- `report_ar` is a PARALLEL formal report in Modern Standard Arabic, not a translation of \
summary_en. Different register, different audience: it is addressed to an Arabic-speaking site \
supervisor and should read as a formal HSE record. Write complete Arabic prose in Arabic script. \
Never transliterate into Latin letters.

Risk calibration:
- Low: a single violation, brief duration, low confidence.
- Medium: a single clear violation held for several seconds.
- High: multiple simultaneous PPE violations, or a sustained head-protection violation.
- Critical: sustained multiple violations indicating the worker is in the zone with no protection \
at all.
"""


def build_payload(
    *,
    report_id: str,
    zone_id: str,
    zone_label_en: str,
    zone_label_ar: str,
    violations: list[str],
    confidence: float,
    duration_seconds: float,
    detected_at_utc: str,
    display_timezone: str = "Asia/Riyadh",
) -> str:
    """The user-turn content: a compact, unambiguous fact block."""
    local = _to_local(detected_at_utc, display_timezone)
    facts = {
        "report_id": report_id,
        "zone_id": zone_id,
        "zone_label_en": zone_label_en,
        "zone_label_ar": zone_label_ar,
        "violations_detected": violations,
        "violation_count": len(violations),
        "detector_confidence": round(confidence, 3),
        "duration_seconds": round(duration_seconds, 2),
        "detected_at_utc": detected_at_utc,
        "detected_at_local": local,
        "local_timezone": display_timezone,
    }
    return (
        "Produce the bilingual HSE incident report for this detection event.\n\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
    )


def _to_local(iso_utc: str, tz_name: str) -> str:
    parsed = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    return parsed.astimezone(ZoneInfo(tz_name)).isoformat()
