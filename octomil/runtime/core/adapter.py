"""InferenceBackendAdapter — bridges existing InferenceBackend to ModelRuntime."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from octomil._generated.modality import Modality
from octomil.runtime.core.chatml_renderer import render_chatml
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeMessage,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeUsage,
    ToolCallTier,
)

logger = logging.getLogger(__name__)


class InferenceBackendAdapter(ModelRuntime):
    """Bridges an existing InferenceBackend to the ModelRuntime protocol."""

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
        _validate_request(request, self._capabilities)
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
        _validate_request(request, self._capabilities)
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

        if self._capabilities.supports_multimodal_input:
            messages = _build_multimodal_messages(request.messages)
        else:
            # Text-only engine: render to ChatML string
            prompt = render_chatml(request)
            messages = [{"role": "user", "content": prompt}]

        return GenerationRequest(
            model=self._model_name,
            messages=messages,
            max_tokens=request.generation_config.max_tokens,
            temperature=request.generation_config.temperature,
            top_p=request.generation_config.top_p,
        )


def _validate_request(request: RuntimeRequest, capabilities: RuntimeCapabilities) -> None:
    """Validate RuntimeRequest against engine capabilities.

    Raises ValueError for capability mismatches.
    """
    for msg in request.messages:
        media_count = 0
        has_non_text = False
        prev_type: Modality | None = None
        interleaved = False

        for part in msg.parts:
            if part.type != Modality.TEXT:
                if part.type not in capabilities.input_modalities:
                    raise ValueError(
                        f"Engine does not support {part.type.value} input. "
                        f"Supported modalities: {', '.join(m.value for m in capabilities.input_modalities)}"
                    )
                media_count += 1
                if has_non_text and prev_type == Modality.TEXT:
                    interleaved = True
                has_non_text = True
            else:
                if has_non_text:
                    interleaved = True
            prev_type = part.type

        if media_count > 0 and capabilities.max_media_parts_per_message is not None:
            if media_count > capabilities.max_media_parts_per_message:
                raise ValueError(
                    f"Message contains {media_count} media parts but engine supports "
                    f"at most {capabilities.max_media_parts_per_message}"
                )

        if interleaved and not capabilities.supports_interleaved_content:
            raise ValueError(
                "Message contains interleaved text and media parts but engine does not support interleaved content"
            )

    # Check historical media
    if not capabilities.supports_historical_media:
        user_messages = [m for m in request.messages if m.role.value == "user"]
        if len(user_messages) > 1:
            for um in user_messages[:-1]:
                for part in um.parts:
                    if part.type != Modality.TEXT:
                        raise ValueError(
                            "Earlier user messages contain media but engine does not "
                            "support historical media (supportsHistoricalMedia=false)"
                        )


def _build_multimodal_messages(messages: list[RuntimeMessage]) -> list[dict]:
    """Build multimodal message dicts for engines that support structured input."""
    result: list[dict] = []
    for msg in messages:
        content: list[dict] = []
        for part in msg.parts:
            if part.type == Modality.TEXT:
                content.append({"type": "text", "text": part.text})
            elif part.type == Modality.IMAGE:
                content.append(
                    {
                        "type": "image",
                        "data": part.data,
                        "media_type": part.media_type,
                    }
                )
            elif part.type == Modality.AUDIO:
                content.append(
                    {
                        "type": "audio",
                        "data": part.data,
                        "media_type": part.media_type,
                    }
                )
            elif part.type == Modality.VIDEO:
                content.append(
                    {
                        "type": "video",
                        "data": part.data,
                        "media_type": part.media_type,
                    }
                )
        result.append({"role": msg.role.value, "content": content})
    return result


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
