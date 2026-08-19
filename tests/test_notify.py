"""Email alerting. Builds real messages; never opens a socket."""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.notify import EmailAlerter
from config import EmailAlertConfig

INCIDENT = {
    "report_id": "INC-20260819-ZONE01-0042",
    "camera_id": "CAM-01",
    "zone_id": "ZONE-01-MAIN-ENTRANCE",
    "zone_label_en": "Main Entrance",
    "track_id": 7,
    "violations": ["NO-Hardhat", "NO-Safety Vest"],
    "confidence": 0.93,
    "duration_seconds": 2.4,
    "detected_at_utc": "2026-08-19T09:14:24.700000Z",
}


def alerter(**env) -> EmailAlerter:
    for key in ("SMTP_USERNAME", "SMTP_PASSWORD"):
        os.environ.pop(key, None)
    os.environ.update(env)
    cfg = EmailAlertConfig(enabled=True, recipients=["supervisor@example.com"])
    return EmailAlerter(cfg, "Asia/Riyadh")


def test_not_configured_without_credentials():
    a = alerter()
    assert not a.configured
    assert "SMTP_USERNAME" in a.why_not() and "SMTP_PASSWORD" in a.why_not()


def test_disabled_config_is_not_configured():
    os.environ.update(SMTP_USERNAME="u@example.com", SMTP_PASSWORD="pw")
    a = EmailAlerter(EmailAlertConfig(enabled=False, recipients=["x@example.com"]))
    assert not a.configured
    assert "enabled is false" in a.why_not()


def test_configured_when_complete():
    a = alerter(SMTP_USERNAME="sender@example.com", SMTP_PASSWORD="app-password")
    assert a.configured, a.why_not()
    assert a.why_not() == "ready"
    assert a.sender == "sender@example.com", "blank sender should fall back to SMTP_USERNAME"


def test_send_is_a_noop_when_unconfigured():
    """A missing credential must never raise into the detection pipeline."""
    a = alerter()
    assert asyncio.run(a.send_incident(INCIDENT, None)) is False


def test_message_contents_and_attachment():
    workdir = Path(tempfile.mkdtemp())
    try:
        evidence = workdir / "INC-20260819-ZONE01-0042.jpg"
        evidence.write_bytes(b"\xff\xd8\xff\xe0stub-jpeg-bytes")

        a = alerter(SMTP_USERNAME="sender@example.com", SMTP_PASSWORD="app-password")
        message = a._build(INCIDENT, evidence)

        assert "PPE violation" in message["Subject"]
        assert "Main Entrance" in message["Subject"]
        assert "NO-Hardhat" in message["Subject"]
        assert message["To"] == "supervisor@example.com"

        body = message.get_body(preferencelist=("plain",)).get_content()
        assert "INC-20260819-ZONE01-0042" in body
        assert "NO-Hardhat, NO-Safety Vest" in body
        assert "93%" in body
        assert "2.4 seconds" in body
        # UTC 09:14 renders as 12:14 in Riyadh (spec: store UTC, display Asia/Riyadh).
        assert "12:14:24" in body, body

        attachments = list(message.iter_attachments())
        assert len(attachments) == 1
        assert attachments[0].get_filename() == evidence.name
        assert attachments[0].get_content_type() == "image/jpeg"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_missing_evidence_file_still_sends_body():
    a = alerter(SMTP_USERNAME="sender@example.com", SMTP_PASSWORD="app-password")
    message = a._build(INCIDENT, Path("does-not-exist.jpg"))
    assert not list(message.iter_attachments()), "must not attach a file that is not there"
    assert "INC-20260819-ZONE01-0042" in message.get_content()


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
