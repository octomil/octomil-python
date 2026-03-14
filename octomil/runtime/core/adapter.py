"""InferenceBackendAdapter — bridges existing InferenceBackend to ModelRuntime."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeUsage,
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
        self._capabilities = capabilities or RuntimeCapabilities(
            supports_tool_calls=False,
            supports_text_tool_calls=True,
            supports_structured_output=False,
            supports_multimodal_input=False,
            supports_streaming=True,
        )

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

        # Attempt text-based tool call extraction when tools are declared
        if request.tool_definitions:
            from octomil.runtime.core.tool_parser import extract_tool_call_from_text

            declared_names = [td.name for td in request.tool_definitions]
            tool_call = extract_tool_call_from_text(text, declared_tools=declared_names)

            if tool_call is not None:
                logger.debug(
                    "tool_call_parse_succeeded: tool=%s, mode=text_json",
                    tool_call.name,
                )
                _emit_tool_call_telemetry(tool_call.name, succeeded=True)
                return RuntimeResponse(
                    text="",
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    usage=usage,
                )

            # Only log parse failure if text looked like it might contain tool JSON
            stripped = text.strip() if text else ""
            if stripped.startswith("{") or '"tool_call"' in stripped:
                logger.debug(
                    "tool_call_parse_failed: mode=text_json, text_preview=%.100s",
                    stripped,
                )
                _emit_tool_call_telemetry(None, succeeded=False)

        return RuntimeResponse(
            text=text,
            finish_reason="stop",
            usage=usage,
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


def _emit_tool_call_telemetry(tool_name: str | None, *, succeeded: bool) -> None:
    """Emit span events for tool call parsing telemetry.

    Best-effort — silently ignores OpenTelemetry import failures.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span.is_recording():
            return

        span.set_attribute("tool_call_parse_mode", "text_json")

        if succeeded:
            span.add_event(
                "tool_call_parse_succeeded",
                attributes={"octomil.tool.name": tool_name or ""},
            )
            span.add_event(
                "tool_call_emitted",
                attributes={"octomil.tool.name": tool_name or ""},
            )
        else:
            span.add_event("tool_call_parse_failed")
    except Exception:
        pass  # OTel not available — skip silently
