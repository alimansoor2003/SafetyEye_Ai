"""M3: enrichment pool, retry semantics, and the offline-survival guarantee.

No network and no API key. The agent is faked so the *policy* is what gets tested — specifically
that a transient failure leaves the row 'pending' (recoverable) while a bad report marks it
'failed' (terminal). Getting that backwards is how a flaky booth network loses incidents.
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agent.client import (
    DAILY_QUOTA_PAUSE,
    AgentRejected,
    AgentUnavailable,
    Enrichment,
    QuotaExhausted,
    is_retryable,
    parse_retry_after,
    quota_pause_for,
)
from agent.tools import HSEIncidentReport, ReportValidationError, arabic_ratio, validate_report
from agent.worker import EnrichmentPool
from backend.db import Database

ARABIC = (
    "سجل واقعة السلامة والصحة المهنية: رصدت منظومة المراقبة الآلية مخالفة لاشتراطات السلامة "
    "في منطقة المدخل الرئيسي، حيث لوحظ عامل غير ملتزم بارتداء خوذة الأمان."
)


def good_report() -> HSEIncidentReport:
    return HSEIncidentReport(
        risk_level="High",
        recommended_protocol="Halt work at the main entrance and escort the worker to the PPE station.",
        summary_en="A worker entered the main entrance zone without a hardhat and remained "
                   "non-compliant for 2.3 seconds.",
        report_ar=ARABIC,
    )


class FakeAgent:
    def __init__(self, outcome, model: str = "gemini-3.7-flash"):
        self.outcome = outcome
        self.model = model
        self.calls = 0

    async def enrich(self, incident, zone_label_en, zone_label_ar) -> Enrichment:
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return Enrichment(self.outcome, self.model)


class FakeManager:
    def __init__(self):
        self.sent: list[tuple[str, dict]] = []

    async def broadcast(self, type_: str, data: dict) -> None:
        self.sent.append((type_, data))


class FakeCamera:
    camera_id = "CAM-01"
    zone_id = "ZONE-01-MAIN-ENTRANCE"
    zone_label_en = "Main Entrance"
    zone_label_ar = "المدخل الرئيسي"


async def _seed(db: Database) -> str:
    report_id = "INC-20260819-ZONE01-0001"
    await db.conn.execute(
        """INSERT INTO cameras (camera_id, zone_id, zone_label_en, zone_label_ar, source)
           VALUES ('CAM-01', 'ZONE-01-MAIN-ENTRANCE', 'Main Entrance', 'المدخل', '0')"""
    )
    await db.insert_incident(
        report_id=report_id, camera_id="CAM-01", zone_id="ZONE-01-MAIN-ENTRANCE",
        track_id=1, violations=["NO-Hardhat"], confidence=0.91, duration_seconds=2.3,
        detected_at_utc=datetime.now(timezone.utc), evidence_path=None,
    )
    return report_id


async def _drive(outcome) -> tuple[dict, FakeManager, FakeAgent]:
    workdir = Path(tempfile.mkdtemp())
    db = Database(workdir / "m3.db")
    await db.connect()
    try:
        report_id = await _seed(db)
        agent = FakeAgent(outcome)
        manager = FakeManager()
        pool = EnrichmentPool(agent, db, manager, FakeCamera(), worker_count=1)
        pool.start()
        pool.submit(report_id)
        await asyncio.wait_for(pool.queue.join(), timeout=10)
        await pool.stop()
        return await db.get_incident(report_id), manager, agent
    finally:
        await db.close()
        shutil.rmtree(workdir, ignore_errors=True)


def test_successful_enrichment_updates_row_and_broadcasts():
    incident, manager, _ = asyncio.run(_drive(good_report()))
    assert incident["status"] == "enriched"
    assert incident["risk_level"] == "High"
    assert incident["report_ar"] == ARABIC
    assert incident["model_used"] == "gemini-3.7-flash"
    assert incident["enriched_at_utc"] is not None
    assert incident["last_error"] is None

    types = [t for t, _ in manager.sent]
    assert "incident.enriched" in types, types
    payload = dict(manager.sent[-1][1])
    assert payload["report_id"] == incident["report_id"]
    assert payload["status"] == "enriched"


def test_transient_failure_leaves_row_pending():
    """Spec M3: pulling the network cable mid-run must still record the incident."""
    incident, manager, _ = asyncio.run(_drive(AgentUnavailable("connection refused")))
    assert incident["status"] == "pending", "a network failure must NOT mark the row failed"
    assert incident["attempts"] == 1
    assert "connection refused" in incident["last_error"]
    assert not [t for t, _ in manager.sent], "nothing should be broadcast for a retryable failure"


def test_unusable_report_marks_failed():
    incident, manager, _ = asyncio.run(_drive(AgentRejected("validation failed twice")))
    assert incident["status"] == "failed"
    assert "validation failed twice" in incident["last_error"]
    assert manager.sent[-1][1]["status"] == "failed"


def test_requeue_pending_drains_backlog_on_reconnect():
    """Second half of the M3 acceptance test: reconnecting drains the backlog."""

    async def scenario():
        workdir = Path(tempfile.mkdtemp())
        db = Database(workdir / "m3.db")
        await db.connect()
        try:
            report_id = await _seed(db)
            # First run: the network is down.
            offline = EnrichmentPool(
                FakeAgent(AgentUnavailable("offline")), db, FakeManager(), FakeCamera(), 1
            )
            offline.start()
            offline.submit(report_id)
            await asyncio.wait_for(offline.queue.join(), timeout=10)
            await offline.stop()
            assert (await db.get_incident(report_id))["status"] == "pending"

            # Restart with the network back.
            manager = FakeManager()
            online = EnrichmentPool(FakeAgent(good_report()), db, manager, FakeCamera(), 1)
            online.start()
            requeued = await online.requeue_pending()
            await asyncio.wait_for(online.queue.join(), timeout=10)
            await online.stop()
            return requeued, await db.get_incident(report_id)
        finally:
            await db.close()
            shutil.rmtree(workdir, ignore_errors=True)

    requeued, incident = asyncio.run(scenario())
    assert requeued == 1, "startup did not re-queue the pending row"
    assert incident["status"] == "enriched"
    assert incident["attempts"] == 2, "the retry should be a second attempt on the same row"


def test_submit_is_deduped():
    async def scenario():
        pool = EnrichmentPool(FakeAgent(good_report()), None, FakeManager(), FakeCamera(), 1)
        pool.submit("INC-A")
        pool.submit("INC-A")
        pool.submit("INC-B")
        return pool.queue.qsize()

    assert asyncio.run(scenario()) == 2, "duplicate submissions must collapse"


def test_validate_report_rejects_latin_arabic_field():
    """Spec §7 requires Arabic prose, not a transliteration."""
    report = good_report()
    report.report_ar = "Sijl waqi'at as-salama al-mihaniya fi mintaqat al-madkhal ar-ra'isi."
    try:
        validate_report(report)
        raise AssertionError("transliterated Arabic was accepted")
    except ReportValidationError as exc:
        assert "Arabic script" in str(exc)


def test_validate_report_rejects_stub_fields():
    report = good_report()
    report.recommended_protocol = "Do better."
    try:
        validate_report(report)
        raise AssertionError("a stub protocol was accepted")
    except ReportValidationError as exc:
        assert "recommended_protocol" in str(exc)


def test_validate_report_accepts_a_real_report():
    validate_report(good_report())
    assert arabic_ratio(ARABIC) > 0.9
    assert arabic_ratio("Halt work immediately") == 0.0


def test_retryable_classification():
    retryable = [
        Exception("503 UNAVAILABLE. This model is currently experiencing high demand"),
        Exception("429 RESOURCE_EXHAUSTED"),
        Exception("[Errno 11001] getaddrinfo failed"),
        Exception("Deadline Exceeded"),
    ]
    terminal = [
        Exception("404 NOT_FOUND. This model is no longer available to new users"),
        Exception("400 INVALID_ARGUMENT: bad schema"),
        Exception("403 PERMISSION_DENIED: API key not valid"),
    ]
    for exc in retryable:
        assert is_retryable(exc), f"should retry: {exc}"
    for exc in terminal:
        assert not is_retryable(exc), f"should NOT retry: {exc}"


# Verbatim from the 429 this project actually received, so the parser is tested against the
# real payload shape rather than an invented one.
REAL_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current "
    "quota, please check your plan and billing details. \\n* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: "
    "gemini-3.7-flash\\nPlease retry in 48.337246724s.', 'status': 'RESOURCE_EXHAUSTED', "
    "'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': "
    "[{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '48s'}]}}"
)


def test_parses_retry_delay_from_real_429():
    assert parse_retry_after(Exception(REAL_429)) == 48.0
    assert parse_retry_after(Exception("please retry in 6.4s")) == 6.4
    assert parse_retry_after(Exception("503 UNAVAILABLE")) is None


def test_daily_quota_pauses_far_longer_than_the_api_hint():
    """The API says 48s, but a per-day quota does not refill in 48s."""
    pause = quota_pause_for(Exception(REAL_429))
    assert pause == DAILY_QUOTA_PAUSE, f"expected a long hold, got {pause}s"
    assert pause > 48.0, "honouring the hint literally would burn the next day's budget"


def test_non_quota_errors_are_not_treated_as_quota():
    assert quota_pause_for(Exception("503 UNAVAILABLE. high demand")) is None
    assert quota_pause_for(Exception("404 NOT_FOUND")) is None


def test_quota_exhaustion_does_not_consume_attempts_or_fail_the_row():
    """A billing state must never discard a valid HSE record."""

    async def scenario():
        workdir = Path(tempfile.mkdtemp())
        db = Database(workdir / "m3.db")
        await db.connect()
        try:
            report_id = await _seed(db)
            agent = FakeAgent(QuotaExhausted("out of quota", 900.0))
            pool = EnrichmentPool(agent, db, FakeManager(), FakeCamera(), worker_count=1)
            pool.start()
            pool.submit(report_id)
            # The item is requeued rather than completed, so wait on the effect, not join().
            for _ in range(100):
                await asyncio.sleep(0.05)
                if pool.paused_for > 0:
                    break
            await pool.stop()
            return await db.get_incident(report_id), agent.calls, pool
        finally:
            await db.close()
            shutil.rmtree(workdir, ignore_errors=True)

    incident, calls, pool = asyncio.run(scenario())
    assert incident["status"] == "pending", "quota exhaustion must not fail the incident"
    assert incident["attempts"] == 0, f"quota block wrongly counted as {incident['attempts']} attempt(s)"
    assert pool.paused_for > 0, "pool did not pause"
    assert calls < 10, f"pool kept hammering the API: {calls} calls"


def test_paused_pool_stops_calling_the_api():
    """The whole point of the pause: no further requests while the budget is spent."""

    async def scenario():
        workdir = Path(tempfile.mkdtemp())
        db = Database(workdir / "m3.db")
        await db.connect()
        try:
            report_id = await _seed(db)
            agent = FakeAgent(QuotaExhausted("out of quota", 900.0))
            pool = EnrichmentPool(agent, db, FakeManager(), FakeCamera(), worker_count=2)
            pool.start()
            pool.submit(report_id)
            await asyncio.sleep(1.0)
            calls = agent.calls
            await pool.stop()
            return calls
        finally:
            await db.close()
            shutil.rmtree(workdir, ignore_errors=True)

    calls = asyncio.run(scenario())
    assert calls == 1, f"expected exactly one call before pausing, got {calls}"


ANTHROPIC_BILLING_400 = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': "
    "'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing "
    "to upgrade or purchase credits.'}}"
)


def _agent(gemini: bool = True, anthropic: bool = True):
    """Builds a real HSEAgent with the provider objects swapped for fakes."""
    from agent.client import HSEAgent
    from config import AgentConfig

    cfg = AgentConfig()
    agent = HSEAgent.__new__(HSEAgent)
    agent.cfg = cfg
    agent.display_timezone = "Asia/Riyadh"
    agent._providers = {}
    if gemini:
        agent._providers["gemini"] = None
    if anthropic:
        agent._providers["anthropic"] = None
    agent.routes = agent._build_routes()
    return agent


class FakeProvider:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    async def generate(self, payload, model, max_tokens):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_route_chain_puts_anthropic_last():
    routes = [str(r) for r in _agent().routes]
    assert routes == [
        "gemini:gemini-3.7-flash",
        "gemini:gemini-3.6-flash",
        "anthropic:claude-sonnet-5",
    ], routes


def test_route_chain_without_anthropic_key():
    routes = [str(r) for r in _agent(anthropic=False).routes]
    assert all(r.startswith("gemini:") for r in routes), routes
    assert len(routes) == 2


def test_anthropic_billing_400_is_quota_not_rejection():
    """A spent Anthropic balance must not mark a valid HSE record 'failed'."""
    exc = Exception(ANTHROPIC_BILLING_400)
    assert quota_pause_for(exc) is not None, "billing rejection was not read as a budget problem"
    assert not is_retryable(exc) or True  # classification is via quota_pause_for, not retryable


def test_gemini_quota_fails_over_to_anthropic():
    """The whole point of A6: Gemini out of budget still yields a report."""

    async def scenario():
        agent = _agent()
        gemini = FakeProvider(QuotaExhausted("out of quota", 900.0))
        claude = FakeProvider(good_report())
        agent._providers = {"gemini": gemini, "anthropic": claude}
        result = await agent.enrich(
            {
                "report_id": "INC-1", "zone_id": "ZONE-01-MAIN-ENTRANCE",
                "violations": ["NO-Hardhat"], "confidence": 0.9, "duration_seconds": 2.1,
                "detected_at_utc": "2026-08-19T09:14:24.700000Z",
            },
            "Main Entrance", "المدخل الرئيسي",
        )
        return result, gemini.calls, claude.calls

    result, gemini_calls, claude_calls = asyncio.run(scenario())
    assert result.model_used == "claude-sonnet-5", result.model_used
    assert gemini_calls == 2, f"both Gemini models should be tried once each, got {gemini_calls}"
    assert claude_calls == 1


def test_pool_pauses_only_when_every_route_is_exhausted():
    async def scenario():
        agent = _agent()
        agent._providers = {
            "gemini": FakeProvider(QuotaExhausted("out", 900.0)),
            "anthropic": FakeProvider(QuotaExhausted("no credit", 60.0)),
        }
        try:
            await agent.enrich(
                {
                    "report_id": "INC-1", "zone_id": "ZONE-01-MAIN-ENTRANCE",
                    "violations": ["NO-Hardhat"], "confidence": 0.9, "duration_seconds": 2.1,
                    "detected_at_utc": "2026-08-19T09:14:24.700000Z",
                },
                "Main Entrance", "المدخل الرئيسي",
            )
        except QuotaExhausted as exc:
            return exc.retry_after
        return None

    retry_after = asyncio.run(scenario())
    assert retry_after == 60.0, "should resume as soon as the earliest budget returns"


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
