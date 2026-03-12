"""OctomilResponses — developer-facing Response API (Layer 2).

**Tier: Core Contract (MUST)**
"""

from __future__ import annotations

import uuid
from typing import AsyncIterator, Callable, Optional

from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.registry import ModelRuntimeRegistry
from octomil.runtime.core.types import (
    RuntimeRequest,
    RuntimeToolDef,
    RuntimeUsage,
)
from octomil.runtime.core.types import (
    RuntimeResponse as _RuntimeResponse,
)

from .prompt_formatter import PromptFormatter
from .types import (
    AssistantInput,
    DoneEvent,
    InputItem,
    JsonSchemaFormat,
    OutputItem,
    Response,
    ResponseFormat,
    ResponseRequest,
    ResponseStreamEvent,
    ResponseToolCall,
    ResponseUsage,
    TextContent,
    TextDeltaEvent,
    TextOutput,
    ToolCallDeltaEvent,
    ToolCallOutput,
    system_input,
    text_input,
)


class OctomilResponses:
    """Developer-facing Response API (Layer 2).

    Provides create() and stream() methods that resolve a ModelRuntime,
    format the prompt, and return structured responses.
    """

    def __init__(
        self,
        runtime_resolver: Optional[Callable[[str], Optional[ModelRuntime]]] = None,
    ) -> None:
        self._runtime_resolver = runtime_resolver
        self._response_cache: dict[str, Response] = {}

    async def create(self, request: ResponseRequest) -> Response:
        runtime = self._resolve_runtime(request.model)
        effective_request = self._apply_previous_response(request)
        runtime_request = self._build_runtime_request(effective_request)
        runtime_response = await runtime.run(runtime_request)
        response = self._build_response(request.model, runtime_response)
        self._response_cache[response.id] = response
        return response

    async def stream(self, request: ResponseRequest) -> AsyncIterator[ResponseStreamEvent]:
        runtime = self._resolve_runtime(request.model)
        runtime_request = self._build_runtime_request(request)
        response_id = _generate_id()
        text_parts: list[str] = []
        tool_call_buffers: dict[int, _ToolCallBuffer] = {}
        last_usage: Optional[RuntimeUsage] = None

        async for chunk in runtime.stream(runtime_request):
            if chunk.text is not None:
                text_parts.append(chunk.text)
                yield TextDeltaEvent(delta=chunk.text)

            if chunk.tool_call_delta is not None:
                delta = chunk.tool_call_delta
                buffer = tool_call_buffers.setdefault(delta.index, _ToolCallBuffer())
                if delta.id is not None:
                    buffer.id = delta.id
                if delta.name is not None:
                    buffer.name = delta.name
                if delta.arguments_delta is not None:
                    buffer.arguments += delta.arguments_delta

                yield ToolCallDeltaEvent(
                    index=delta.index,
                    id=delta.id,
                    name=delta.name,
                    arguments_delta=delta.arguments_delta,
                )

            if chunk.usage is not None:
                last_usage = chunk.usage

        output: list[OutputItem] = []
        full_text = "".join(text_parts)
        if full_text:
            output.append(TextOutput(text=full_text))
        for idx in sorted(tool_call_buffers):
            buf = tool_call_buffers[idx]
            output.append(
                ToolCallOutput(
                    tool_call=ResponseToolCall(
                        id=buf.id or _generate_id(),
                        name=buf.name or "",
                        arguments=buf.arguments,
                    )
                )
            )

        finish_reason = "tool_calls" if tool_call_buffers else "stop"
        usage = (
            ResponseUsage(
                prompt_tokens=last_usage.prompt_tokens,
                completion_tokens=last_usage.completion_tokens,
                total_tokens=last_usage.total_tokens,
            )
            if last_usage
            else None
        )

        yield DoneEvent(
            response=Response(
                id=response_id,
                model=request.model,
                output=output,
                finish_reason=finish_reason,
                usage=usage,
            )
        )

    def _resolve_runtime(self, model: str) -> ModelRuntime:
        if self._runtime_resolver is not None:
            runtime = self._runtime_resolver(model)
            if runtime is not None:
                return runtime
        runtime = ModelRuntimeRegistry.shared().resolve(model)
        if runtime is not None:
            return runtime
        raise RuntimeError(f"No ModelRuntime registered for model: {model}")

    def _apply_previous_response(self, request: ResponseRequest) -> ResponseRequest:
        """Prepend previous response output as assistant context when previous_response_id is set."""
        if not request.previous_response_id:
            return request
        prev = self._response_cache.get(request.previous_response_id)
        if prev is None:
            return request

        # Build assistant input from previous response output
        assistant_text = "".join(item.text for item in prev.output if isinstance(item, TextOutput))
        assistant_item = AssistantInput(content=[TextContent(text=assistant_text)] if assistant_text else None)

        # Normalize current input
        input_items: list[InputItem]
        if isinstance(request.input, str):
            input_items = [text_input(request.input)]
        else:
            input_items = list(request.input)

        return ResponseRequest(
            model=request.model,
            input=[assistant_item] + input_items,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
            stream=request.stream,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            metadata=request.metadata,
            instructions=request.instructions,
        )

    def _build_runtime_request(self, request: ResponseRequest) -> RuntimeRequest:
        # Normalize string input
        input_items: list[InputItem]
        if isinstance(request.input, str):
            input_items = [text_input(request.input)]
        else:
            input_items = list(request.input)

        # Prepend instructions as system input
        if request.instructions:
            input_items = [system_input(request.instructions)] + input_items

        prompt = PromptFormatter.format(
            input_items,
            request.tools if request.tools else None,
            request.tool_choice,
        )
        tool_defs: Optional[list[RuntimeToolDef]] = None
        if request.tools:
            tool_defs = [
                RuntimeToolDef(
                    name=t.get("function", t).get("name", ""),
                    description=t.get("function", t).get("description", ""),
                    parameters_schema=str(t.get("function", t).get("parameters"))
                    if t.get("function", t).get("parameters")
                    else None,
                )
                for t in request.tools
            ]

        json_schema: Optional[str] = None
        if isinstance(request.response_format, JsonSchemaFormat):
            json_schema = request.response_format.schema
        elif request.response_format == ResponseFormat.JSON_OBJECT:
            json_schema = "{}"

        return RuntimeRequest(
            prompt=prompt,
            max_tokens=request.max_output_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 1.0,
            stop=request.stop,
            tool_definitions=tool_defs,
            json_schema=json_schema,
        )

    def _build_response(self, model: str, runtime_response: _RuntimeResponse) -> Response:
        output: list[OutputItem] = []

        if runtime_response.text:
            output.append(TextOutput(text=runtime_response.text))

        if runtime_response.tool_calls:
            for call in runtime_response.tool_calls:
                output.append(
                    ToolCallOutput(
                        tool_call=ResponseToolCall(
                            id=call.id,
                            name=call.name,
                            arguments=call.arguments,
                        )
                    )
                )

        finish_reason = "tool_calls" if runtime_response.tool_calls else runtime_response.finish_reason

        usage = (
            ResponseUsage(
                prompt_tokens=runtime_response.usage.prompt_tokens,
                completion_tokens=runtime_response.usage.completion_tokens,
                total_tokens=runtime_response.usage.total_tokens,
            )
            if runtime_response.usage
            else None
        )

        return Response(
            id=_generate_id(),
            model=model,
            output=output,
            finish_reason=finish_reason,
            usage=usage,
        )


def _generate_id() -> str:
    return f"resp_{uuid.uuid4().hex[:16]}"


class _ToolCallBuffer:
    def __init__(self) -> None:
        self.id: Optional[str] = None
        self.name: Optional[str] = None
        self.arguments: str = ""
