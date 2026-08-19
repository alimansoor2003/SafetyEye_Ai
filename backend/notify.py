"""Email alerting on violation.

Fires on `incident.created`, not on enrichment: the alert must reach a supervisor whether or not
the LLM budget is available, and enrichment can take ~45s under load. The same reasoning as spec
§5's insert-before-enrich ordering — the safety-critical path never depends on the model API.

Credentials come from the environment (SMTP_USERNAME / SMTP_PASSWORD), never from config.yaml,
so nothing secret is ever committed. For Gmail, SMTP_PASSWORD must be a Google App Password;
a normal account password will not authenticate.
"""
from __future__ import annotations

import asyncio
import logging
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

from config import EmailAlertConfig

log = logging.getLogger(__name__)

SEND_TIMEOUT = 20.0


class EmailAlerter:
    def __init__(self, cfg: EmailAlertConfig, display_timezone: str = "Asia/Riyadh"):
        self.cfg = cfg
        self.tz = ZoneInfo(display_timezone)
        self.username = os.getenv("SMTP_USERNAME", "").strip()
        self.password = os.getenv("SMTP_PASSWORD", "").strip()
        self.sender = cfg.sender or self.username
        self.last_error: str | None = None
        self.sent = 0

    @property
    def configured(self) -> bool:
        return bool(
            self.cfg.enabled and self.username and self.password
            and self.sender and self.cfg.recipients
        )

    def why_not(self) -> str:
        if not self.cfg.enabled:
            return "email_alerts.enabled is false in config.yaml"
        missing = [
            name for name, value in (
                ("SMTP_USERNAME", self.username),
                ("SMTP_PASSWORD", self.password),
                ("email_alerts.recipients", self.cfg.recipients),
            ) if not value
        ]
        return f"missing {', '.join(missing)}" if missing else "ready"

    async def send_incident(self, incident: dict, evidence_path: Path | None) -> bool:
        """Never raises: a mail failure must not disturb the detection pipeline."""
        if not self.configured:
            return False
        try:
            message = self._build(incident, evidence_path)
            await asyncio.wait_for(
                asyncio.to_thread(self._deliver, message), timeout=SEND_TIMEOUT
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.error("alert email failed for %s: %s", incident.get("report_id"), self.last_error)
            return False

        self.sent += 1
        self.last_error = None
        log.info(
            "alert email sent for %s to %s",
            incident.get("report_id"), ", ".join(self.cfg.recipients),
        )
        return True

    def _deliver(self, message: EmailMessage) -> None:
        context = ssl.create_default_context()
        with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port, timeout=SEND_TIMEOUT) as server:
            server.starttls(context=context)
            server.login(self.username, self.password)
            server.send_message(message)

    def _build(self, incident: dict, evidence_path: Path | None) -> EmailMessage:
        violations = incident["violations"]
        zone = incident.get("zone_label_en") or incident["zone_id"]
        local = self._local(incident["detected_at_utc"])

        message = EmailMessage()
        message["Subject"] = f"[safetyeye] PPE violation — {zone} — {', '.join(violations)}"
        message["From"] = self.sender
        message["To"] = ", ".join(self.cfg.recipients)
        message.set_content(_body(incident, violations, zone, local))

        if self.cfg.attach_evidence and evidence_path and evidence_path.is_file():
            message.add_attachment(
                evidence_path.read_bytes(),
                maintype="image", subtype="jpeg", filename=evidence_path.name,
            )
        return message

    def _local(self, iso_utc: str) -> str:
        parsed = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(self.tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def _body(incident: dict, violations: list[str], zone: str, local: str) -> str:
    return f"""\
SAFETY VIOLATION DETECTED

Report ID   : {incident['report_id']}
Zone        : {zone} ({incident['zone_id']})
Camera      : {incident['camera_id']}
Violations  : {', '.join(violations)}
Confidence  : {incident['confidence']:.0%}
Held for    : {incident['duration_seconds']:.1f} seconds
Detected at : {local}
Track ID    : {incident['track_id']}

This alert was generated automatically by the safetyeye AI monitoring system when a
person remained non-compliant for longer than the configured threshold. The annotated
evidence image is attached.

The full bilingual HSE report is written to the incident record separately and may not
be complete at the time this alert was sent.
"""
