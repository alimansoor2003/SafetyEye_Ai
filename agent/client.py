"""Gemini client: structured output, retry/backoff, fallback model.

The distinction that matters for spec M3's acceptance test — "pulling the network cable mid-run
still records incidents, and reconnecting drains the pending backlog" — is between:

  AgentUnavailable : transient (network down, 429, 5xx). The row stays 'pending' and is retried.
  AgentRejected    : the model answered but the answer is unusable after a re-ask. Row -> 'failed'.

Getting that backwards would either lose incidents permanently on a flaky booth network, or retry
a malformed response forever.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass

from agent.prompts import build_payload
from agent.providers import AnthropicProvider, GeminiProvider
from agent.tools import HSEIncidentReport, ReportValidationError, validate_report
from config import AgentConfig

log = logging.getLogger(__name__)

RETRYABLE_MARKERS = (
    "deadline", "unavailable", "timeout", "timed out", "429", "resource_exhausted",
    "rate limit", "500", "502", "503", "504", "internal error", "connection",
    "temporarily", "overloaded", "getaddrinfo", "ssl", "network",
)
QUOTA_MARKERS = (
    "resource_exhausted", "exceeded your current quota", "429",
    # Anthropic signals a spent budget as a 400, not a 429. Without these it would be read as
    # "the model produced an unusable report" and mark a valid HSE record permanently failed.
    "credit balance is too low", "insufficient_quota", "billing_error",
)
DAILY_QUOTA_MARKERS = ("perday", "per day", "requestsperday")

# A daily quota does not refill in the ~50s the API suggests, so honouring that hint literally
# just burns the next day's budget too. Hold for a quarter of an hour instead.
DAILY_QUOTA_PAUSE = 900.0
RATE_LIMIT_PAUSE = 60.0

_RETRY_DELAY_PATTERNS = (
    re.compile(r"retrydelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.I),
    re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.I),
)


class AgentUnavailable(RuntimeError):
    """Transient failure — leave the incident pending and try again later."""


class QuotaExhausted(AgentUnavailable):
    """The API budget is spent. Retrying now cannot succeed and costs more budget.

    Distinct from AgentUnavailable because it must NOT count against an incident's attempt
    budget: nothing is wrong with the incident, and marking it 'failed' would discard a valid
    HSE record because of a billing state.
    """

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


def parse_retry_after(exc: Exception) -> float | None:
    text = str(exc)
    for pattern in _RETRY_DELAY_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def quota_pause_for(exc: Exception) -> float | None:
    """Returns how long to stop calling the API, or None if this is not a quota failure."""
    text = f"{type(exc).__name__} {exc}".lower()
    if not any(marker in text for marker in QUOTA_MARKERS):
        return None
    hint = parse_retry_after(exc)
    if any(marker in text.replace(" ", "") for marker in DAILY_QUOTA_MARKERS):
        return max(hint or 0.0, DAILY_QUOTA_PAUSE)
    return max(hint or 0.0, RATE_LIMIT_PAUSE)


class AgentRejected(RuntimeError):
    """The model responded but could not produce a usable report."""


@dataclass
class Enrichment:
    report: HSEIncidentReport
    model_used: str


@dataclass(frozen=True)
class Route:
    """One (provider, model) attempt in the fallback chain."""

    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in RETRYABLE_MARKERS)


class HSEAgent:
    """Routes an incident through the provider chain until one produces a valid report."""

    def __init__(
        self,
        cfg: AgentConfig,
        gemini_key: str = "",
        anthropic_key: str = "",
        display_timezone: str = "Asia/Riyadh",
    ):
        self.cfg = cfg
        self.display_timezone = display_timezone
        self._providers: dict[str, object] = {}

        if gemini_key:
            self._providers["gemini"] = GeminiProvider(gemini_key)
        if anthropic_key and cfg.claude_model:
            self._providers["anthropic"] = AnthropicProvider(anthropic_key)

        if not self._providers:
            raise ValueError("no provider keys set (GEMINI_API_KEY / ANTHROPIC_API_KEY)")

        self.routes = self._build_routes()
        log.info("agent route chain: %s", " -> ".join(str(r) for r in self.routes))

    def _build_routes(self) -> list[Route]:
        """Gemini first (cheap, fast), Anthropic last (the budget-exhaustion escape hatch)."""
        primary = self.cfg.fallback_model if self.cfg.use_fallback else self.cfg.model
        secondary = self.cfg.model if self.cfg.use_fallback else self.cfg.fallback_model

        routes: list[Route] = []
        if "gemini" in self._providers:
            for model in (primary, secondary):
                route = Route("gemini", model)
                if model and route not in routes:
                    routes.append(route)
        if "anthropic" in self._providers:
            routes.append(Route("anthropic", self.cfg.claude_model))
        return routes

    @property
    def model(self) -> str:
        return self.routes[0].model

    async def enrich(self, incident: dict, zone_label_en: str, zone_label_ar: str) -> Enrichment:
        payload = build_payload(
            report_id=incident["report_id"],
            zone_id=incident["zone_id"],
            zone_label_en=zone_label_en,
            zone_label_ar=zone_label_ar,
            violations=incident["violations"],
            confidence=incident["confidence"],
            duration_seconds=incident["duration_seconds"],
            detected_at_utc=incident["detected_at_utc"],
            display_timezone=self.display_timezone,
        )

        last_error: Exception | None = None
        quota_waits: list[float] = []

        for route in self.routes:
            try:
                report = await self._call_with_retry(payload, route)
                if quota_waits:
                    log.info("%s covered for %d exhausted route(s)", route, len(quota_waits))
                return Enrichment(report, route.model)
            except QuotaExhausted as exc:
                # Budgets are per-model and per-vendor, so the next route may well succeed.
                last_error = exc
                quota_waits.append(exc.retry_after)
                log.warning("%s is out of quota (retry in %.0fs)", route, exc.retry_after)
            except AgentUnavailable as exc:
                last_error = exc
                log.warning("%s unavailable (%s) — trying next route", route, exc)
            except AgentRejected as exc:
                last_error = exc
                log.warning("%s produced an unusable report (%s) — trying next route", route, exc)

        # Only pause the pool when every route is out of budget. If one merely erred, the
        # incident is retryable now rather than in fifteen minutes.
        if quota_waits and len(quota_waits) == len(self.routes):
            raise QuotaExhausted(f"all routes out of quota: {last_error}", min(quota_waits))
        if isinstance(last_error, AgentRejected):
            raise last_error
        raise AgentUnavailable(str(last_error) if last_error else "no model produced a report")

    async def _call_with_retry(self, payload: str, route: Route) -> HSEIncidentReport:
        """Backoff covers transport failures. Validation gets exactly one re-ask (spec §7)."""
        contents = payload
        validation_retried = False

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                provider = self._providers[route.provider]
                report = await provider.generate(contents, route.model, self.cfg.max_tokens)
                validate_report(report)
                return report

            except ReportValidationError as exc:
                if validation_retried:
                    raise AgentRejected(f"validation failed twice: {exc}") from exc
                validation_retried = True
                log.info("re-asking %s after validation failure: %s", route, exc)
                contents = (
                    f"{payload}\n\nYour previous response was rejected: {exc}\n"
                    "Correct that specific problem and emit the report again."
                )

            except (QuotaExhausted, AgentUnavailable, AgentRejected):
                # A provider that has already classified its own failure is authoritative.
                # Re-deriving the category from the message text would only lose information.
                raise

            except Exception as exc:
                # Quota failures abort this model immediately. Burning the remaining attempts
                # on a budget the API has already declined costs requests we cannot spare.
                pause = quota_pause_for(exc)
                if pause is not None:
                    raise QuotaExhausted(str(exc), pause) from exc

                if not is_retryable(exc) or attempt == self.cfg.max_retries:
                    if is_retryable(exc):
                        raise AgentUnavailable(str(exc)) from exc
                    raise AgentRejected(f"{type(exc).__name__}: {exc}") from exc

                delay = self.cfg.backoff_base_seconds ** attempt
                delay += random.uniform(0, delay * 0.25)  # jitter: workers must not sync up
                # The API knows better than our exponent when it tells us how long to wait.
                hint = parse_retry_after(exc)
                if hint is not None:
                    delay = max(delay, hint)
                log.warning(
                    "%s attempt %d/%d failed (%s) — retrying in %.1fs",
                    route, attempt, self.cfg.max_retries, exc, delay,
                )
                await asyncio.sleep(delay)

        raise AgentUnavailable(f"{route} exhausted {self.cfg.max_retries} attempts")
