"""CloudModelRuntime — OpenAI-compatible cloud inference as a ModelRuntime."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

from octomil.runtime.core.cloud_client import CloudClient
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeToolCall,
    RuntimeToolCallDelta,
    RuntimeUsage,
    ToolCallTier,
)

logger = logging.getLogger(__name__)


def _messages_to_openai(request: RuntimeRequest) -> list[dict[str, Any]]:
    """Convert RuntimeRequest messages to OpenAI chat messages format.

    Handles native tool calling: assistant messages with tool_calls get the
    OpenAI ``tool_calls`` field, and tool-result messages include
    ``tool_call_id``.
    """
    from octomil._generated.message_role import MessageRole
    from octomil._generated.modality import Modality

    messages: list[dict[str, Any]] = []
    for msg in request.messages:
        text_parts: list[str] = []
        for part in msg.parts:
            if part.type == Modality.TEXT:
                text_parts.append(part.text or "")
            else:
                text_parts.append(f"[{part.type.value}]")
        content = "".join(text_parts)

        # Assistant message with native tool calls
        if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
            openai_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            messages.append(openai_msg)
            continue

        # Tool result message with tool_call_id
        if msg.role == MessageRole.TOOL and msg.tool_call_id:
            messages.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": msg.tool_call_id,
                }
            )
            continue

        messages.append({"role": msg.role.value, "content": content})
    return messages


def _tools_to_openai(request: RuntimeRequest) -> Optional[list[dict[str, Any]]]:
    """Convert RuntimeToolDef list to OpenAI tools format."""
    if not request.tool_definitions:
        return None
    tools: list[dict[str, Any]] = []
    for td in request.tool_definitions:
        func: dict[str, Any] = {
            "name": td.name,
            "description": td.description,
        }
        if td.parameters_schema:
            try:
                func["parameters"] = json.loads(td.parameters_schema)
            except (json.JSONDecodeError, ValueError):
                pass
        tools.append({"type": "function", "function": func})
    return tools


class CloudModelRuntime(ModelRuntime):
    """ModelRuntime that delegates to an OpenAI-compatible cloud endpoint.

    Uses CloudClient for HTTP transport, SSE parsing, and retry logic.
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = CloudClient(base_url, api_key, model)
        self._model = model

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            tool_call_tier=ToolCallTier.NATIVE,
            supports_structured_output=True,
            supports_streaming=True,
        )

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        messages = _messages_to_openai(request)
        tools = _tools_to_openai(request)
        gc = request.generation_config

        result = await self._client.chat(
            messages,
            max_tokens=gc.max_tokens,
            temperature=gc.temperature,
            top_p=gc.top_p,
            tools=tools,
        )

        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content") or ""
        finish_reason = choice.get("finish_reason", "stop")

        # Parse tool calls from response
        tool_calls: Optional[list[RuntimeToolCall]] = None
        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                fn = tc.get("function", {})
                tool_calls.append(
                    RuntimeToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        arguments=fn.get("arguments", ""),
                    )
                )

        # Parse usage
        usage: Optional[RuntimeUsage] = None
        if result.get("usage"):
            u = result["usage"]
            usage = RuntimeUsage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )

        return RuntimeResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        messages = _messages_to_openai(request)
        tools = _tools_to_openai(request)
        gc = request.generation_config

        async for chunk in self._client.chat_stream(
            messages,
            max_tokens=gc.max_tokens,
            temperature=gc.temperature,
            top_p=gc.top_p,
            tools=tools,
        ):
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            text = delta.get("content")

            # Parse streaming tool call deltas
            tc_delta: Optional[RuntimeToolCallDelta] = None
            if delta.get("tool_calls"):
                tc = delta["tool_calls"][0]
                fn = tc.get("function", {})
                tc_delta = RuntimeToolCallDelta(
                    index=tc.get("index", 0),
                    id=tc.get("id"),
                    name=fn.get("name"),
                    arguments_delta=fn.get("arguments"),
                )

            # Parse usage from final chunk
            usage: Optional[RuntimeUsage] = None
            if chunk.get("usage"):
                u = chunk["usage"]
                usage = RuntimeUsage(
                    prompt_tokens=u.get("prompt_tokens", 0),
                    completion_tokens=u.get("completion_tokens", 0),
                    total_tokens=u.get("total_tokens", 0),
                )

            yield RuntimeChunk(
                text=text,
                tool_call_delta=tc_delta,
                finish_reason=finish_reason,
                usage=usage,
            )

    def close(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._client.close())
        except RuntimeError:
            asyncio.run(self._client.close())
