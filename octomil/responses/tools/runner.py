"""ToolRunner — convenience loop: model -> tool calls -> execute -> feed back -> repeat."""

from __future__ import annotations

from ..responses import OctomilResponses
from ..types import (
    AssistantInput,
    InputItem,
    Response,
    ResponseRequest,
    ToolCallOutput,
    ToolResultInput,
    text_input,
)
from .executor import ToolExecutor, ToolResult


class ToolRunner:
    """Convenience loop that runs model -> tool calls -> execute -> feed results -> repeat.

    Continues until the model produces a text response (no tool calls) or
    max_iterations is reached.
    """

    def __init__(
        self,
        responses: OctomilResponses,
        executor: ToolExecutor,
        max_iterations: int = 10,
    ) -> None:
        self._responses = responses
        self._executor = executor
        self._max_iterations = max_iterations

    async def run(self, request: ResponseRequest) -> Response:
        current_input: list[InputItem] = (
            [text_input(request.input)] if isinstance(request.input, str) else list(request.input)
        )
        iteration = 0

        while iteration < self._max_iterations:
            current_request = ResponseRequest(
                model=request.model,
                input=current_input,
                tools=request.tools,
                tool_choice=request.tool_choice,
                response_format=request.response_format,
                max_output_tokens=request.max_output_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                metadata=request.metadata,
            )
            response = await self._responses.create(current_request)

            tool_calls = [item.tool_call for item in response.output if isinstance(item, ToolCallOutput)]

            if not tool_calls:
                return response

            # Add assistant message with tool calls
            current_input.append(AssistantInput(tool_calls=tool_calls))

            # Execute each tool call and add results
            for call in tool_calls:
                try:
                    result = await self._executor.execute(call)
                except Exception as e:
                    result = ToolResult(
                        tool_call_id=call.id,
                        content=f"Error: {e}",
                        is_error=True,
                    )
                current_input.append(
                    ToolResultInput(
                        tool_call_id=result.tool_call_id,
                        content=result.content,
                    )
                )

            iteration += 1

        # Max iterations reached — final call without tools
        final_request = ResponseRequest(
            model=request.model,
            input=current_input,
            tools=[],
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            metadata=request.metadata,
        )
        return await self._responses.create(final_request)
