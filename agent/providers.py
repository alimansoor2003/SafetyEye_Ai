"""Model providers behind one interface.

Amendment A6: Gemini is primary, Anthropic is the cross-provider fallback for when Gemini's
budget is spent. Two vendors means an exhausted quota at the booth degrades to a slower report
rather than to no report at all.

Each provider returns a validated `HSEIncidentReport` or raises. Neither parses free-text JSON:
Gemini uses `response_schema`, Anthropic uses forced tool-use exactly as the original spec §7
specified. The Pydantic model is the single source of truth for both.
"""
from __future__ import annotations

import logging
from typing import Protocol

from agent.prompts import SYSTEM_PROMPT
from agent.tools import HSEIncidentReport, ReportValidationError

log = logging.getLogger(__name__)

TOOL_NAME = "emit_hse_incident_report"


class Provider(Protocol):
    name: str

    async def generate(self, payload: str, model: str, max_tokens: int) -> HSEIncidentReport:
        ...


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)

    async def generate(self, payload: str, model: str, max_tokens: int) -> HSEIncidentReport:
        from google.genai import types

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=HSEIncidentReport,
                max_output_tokens=max_tokens,
                temperature=0.4,
                # We use response_schema, not tools. Leaving AFC on logs a warning every call.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        report = response.parsed
        if not isinstance(report, HSEIncidentReport):
            # Truncation by max_output_tokens and safety blocks both land here.
            raise ReportValidationError(
                f"no parseable structured output (finish reason: {_gemini_finish(response)})"
            )
        return report


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str):
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self._tool = {
            "name": TOOL_NAME,
            "description": "Emit the validated bilingual HSE incident report.",
            "input_schema": _tool_schema(),
        }

    async def generate(self, payload: str, model: str, max_tokens: int) -> HSEIncidentReport:
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.4,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": payload}],
            tools=[self._tool],
            # Forces the shape; the model cannot answer in prose.
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
                try:
                    return HSEIncidentReport.model_validate(block.input)
                except Exception as exc:
                    raise ReportValidationError(f"tool input did not validate: {exc}") from exc
        raise ReportValidationError(
            f"no tool_use block returned (stop reason: {response.stop_reason})"
        )


def _tool_schema() -> dict:
    """Anthropic wants a plain JSON Schema object; pydantic's title/$defs noise is not useful."""
    schema = HSEIncidentReport.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


def _gemini_finish(response) -> str:
    try:
        return str(response.candidates[0].finish_reason)
    except Exception:
        return "unknown"
