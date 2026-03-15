"""InferenceBackendAdapter — bridges existing InferenceBackend to ModelRuntime."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeUsage,
    ToolCallTier,
)

logger = logging.getLogger(__name__)


class InferenceBackendAdapter(ModelRuntime):
    """Bridges an existing InferenceBackend to the ModelRuntime protocol.

    Zero changes to existing backend implementations.
    """

    def __init__(
        self,
        backend: object,
        model_name: str,
        capabilities: RuntimeCapabilities | None = None,
    ) -> None:
        self._backend = backend
        self._model_name = model_name
        self._capabilities = capabilities or RuntimeCapabilities()

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return self._capabilities

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        gen_request = self._to_generation_request(request)
        text, metrics = self._backend.generate(gen_request)  # type: ignore[attr-defined]
        usage = RuntimeUsage(
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.total_tokens - metrics.prompt_tokens,
            total_tokens=metrics.total_tokens,
        )

        # Only attempt extraction for TEXT_JSON tier when tools are declared
        if request.tool_definitions and self._capabilities.tool_call_tier == ToolCallTier.TEXT_JSON:
            from octomil.runtime.core.tool_parser import extract_tool_call_with_validation

            declared_names = [td.name for td in request.tool_definitions]
            tool_schemas = _build_tool_schemas(request.tool_definitions)
            result = extract_tool_call_with_validation(text, declared_tools=declared_names, tool_schemas=tool_schemas)

            if result.tool_call is not None:
                logger.debug(
                    "tool_call_parse_succeeded: tool=%s, tier=TEXT_JSON, schema_valid=%s",
                    result.tool_call.name,
                    result.schema_valid,
                )
                _emit_tool_call_telemetry(
                    result.tool_call.name,
                    succeeded=True,
                    tier=ToolCallTier.TEXT_JSON,
                )
                return RuntimeResponse(
                    text="",
                    tool_calls=[result.tool_call],
                    finish_reason="tool_calls",
                    usage=usage,
                    raw_text=text,
                )

            # Log parse failure if text looked like it might be a tool call attempt
            stripped = text.strip() if text else ""
            if stripped.startswith("{"):
                logger.debug(
                    "tool_call_parse_failed: tier=TEXT_JSON, text_preview=%.100s",
                    stripped,
                )
                _emit_tool_call_telemetry(None, succeeded=False, tier=ToolCallTier.TEXT_JSON)

        return RuntimeResponse(
            text=text,
            finish_reason="stop",
            usage=usage,
            raw_text=text if request.tool_definitions else None,
        )

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:  # type: ignore[override]
        gen_request = self._to_generation_request(request)
        async for chunk in self._backend.generate_stream(gen_request):  # type: ignore[attr-defined]
            yield RuntimeChunk(
                text=chunk.text if chunk.text else None,
                finish_reason=chunk.finish_reason,
            )

    def close(self) -> None:
        pass

    def _to_generation_request(self, request: RuntimeRequest) -> object:
        from octomil.serve import GenerationRequest

        return GenerationRequest(
            model=self._model_name,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )


def _build_tool_schemas(tool_definitions: list) -> dict[str, dict]:
    """Build name -> schema dict from RuntimeToolDef list."""
    schemas: dict[str, dict] = {}
    for td in tool_definitions:
        if td.parameters_schema:
            try:
                schemas[td.name] = json.loads(td.parameters_schema)
            except (json.JSONDecodeError, ValueError):
                pass
    return schemas


def _emit_tool_call_telemetry(
    tool_name: str | None,
    *,
    succeeded: bool,
    tier: ToolCallTier,
) -> None:
    """Emit span events for tool call parsing telemetry.

    Best-effort — silently ignores OpenTelemetry import failures.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span.is_recording():
            return

        span.set_attribute("tool.call_tier", tier.value)

        if succeeded:
            span.add_event(
                "tool_call_parse_succeeded",
                attributes={
                    "octomil.tool.name": tool_name or "",
                    "octomil.tool.extraction_strategy": "text_json",
                },
            )
            span.add_event(
                "tool_call_emitted",
                attributes={"octomil.tool.name": tool_name or ""},
            )
        else:
            span.add_event(
                "tool_call_parse_failed",
                attributes={
                    "octomil.tool.extraction_strategy": "text_json",
                },
            )
    except Exception:
        pass  # OTel not available — skip silently
