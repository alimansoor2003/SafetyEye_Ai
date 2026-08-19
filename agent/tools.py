"""Structured output contract (spec §7, amendment A5).

The original spec forced an Anthropic tool schema. Gemini's equivalent is `response_schema`: the
SDK validates the model's output against this Pydantic model and returns a typed instance via
`response.parsed`. The guarantee spec §7 actually cared about — no free-text JSON parsing — holds.

The model is kept free of validators on purpose. `response_schema` is handed to the SDK, and a
validator raising *inside* the SDK surfaces as an opaque parse error rather than something the
worker can act on. The strict checks live in `validate_report`, which the worker calls server-side
exactly as spec §7 requires.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from backend.schemas import RiskLevel

# Arabic block: U+0600–U+06FF, plus the Supplement and Extended-A ranges.
ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF))
MIN_ARABIC_RATIO = 0.30
MIN_FIELD_LENGTH = 20


class ReportValidationError(ValueError):
    """The model produced a structurally valid but unusable report."""


class HSEIncidentReport(BaseModel):
    """The validated bilingual HSE incident report."""

    risk_level: RiskLevel = Field(
        description="Severity of the observed PPE non-compliance."
    )
    recommended_protocol: str = Field(
        description="One specific, immediately actionable on-site instruction for the supervisor."
    )
    summary_en: str = Field(
        description="Executive-level incident description, 2-4 sentences, factual, no speculation."
    )
    report_ar: str = Field(
        description=(
            "Formal Arabic HSE incident report in Modern Standard Arabic, suitable for a site "
            "supervisor. Must be complete Arabic prose in Arabic script, not a transliteration."
        )
    )


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in ARABIC_RANGES))
    return arabic / len(letters)


def validate_report(report: HSEIncidentReport) -> None:
    """Server-side validation. Raises ReportValidationError, which triggers exactly one re-ask."""
    for name in ("recommended_protocol", "summary_en", "report_ar"):
        if len(getattr(report, name).strip()) < MIN_FIELD_LENGTH:
            raise ReportValidationError(f"{name} is too short to be a usable report")

    # Guards spec §7's "complete Arabic prose, not a transliteration". A model answering in
    # English here would otherwise put Latin text into the dashboard's RTL block.
    ratio = arabic_ratio(report.report_ar)
    if ratio < MIN_ARABIC_RATIO:
        raise ReportValidationError(
            f"report_ar is only {ratio:.0%} Arabic script — expected Arabic prose"
        )
