"""PromptFormatter — converts InputItems + tools to a ChatML prompt string."""

from __future__ import annotations

from typing import Any

from .types import (
    AssistantInput,
    AudioContent,
    FileContent,
    ImageContent,
    InputItem,
    SpecificToolChoice,
    SystemInput,
    TextContent,
    ToolChoice,
    ToolResultInput,
    UserInput,
)


class PromptFormatter:
    """Formats InputItems and tools into a ChatML-style prompt string.

    Reuses the same template format as the existing chat API.
    """

    @staticmethod
    def format(
        input: list[InputItem],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice | SpecificToolChoice = ToolChoice.AUTO,
    ) -> str:
        sb: list[str] = []

        # Add tool definitions as system prompt if present
        if tools and not (isinstance(tool_choice, ToolChoice) and tool_choice == ToolChoice.NONE):
            sb.append("<|system|>\nYou have access to the following tools:\n\n")
            for tool in tools:
                fn = tool.get("function", tool)
                sb.append(f"Function: {fn.get('name', '')}\n")
                sb.append(f"Description: {fn.get('description', '')}\n")
                params = fn.get("parameters")
                if params:
                    sb.append(f"Parameters: {params}\n")
                sb.append("\n")
            sb.append(
                'To use a tool, respond with JSON: {"tool_call": {"name": "function_name", "arguments": {...}}}\n'
            )

            if isinstance(tool_choice, ToolChoice) and tool_choice == ToolChoice.REQUIRED:
                sb.append("You MUST use one of the available tools.\n")
            elif isinstance(tool_choice, SpecificToolChoice):
                sb.append(f"You MUST use the tool: {tool_choice.name}\n")
            sb.append("\n")

        # Format each input item
        for item in input:
            if isinstance(item, SystemInput):
                sb.append(f"<|system|>\n{item.content}\n")
            elif isinstance(item, UserInput):
                sb.append("<|user|>\n")
                for part in item.content:
                    if isinstance(part, TextContent):
                        sb.append(part.text)
                    elif isinstance(part, ImageContent):
                        sb.append("[image]")
                    elif isinstance(part, AudioContent):
                        sb.append("[audio]")
                    elif isinstance(part, FileContent):
                        sb.append(f"[file: {part.filename or 'attachment'}]")
                sb.append("\n")
            elif isinstance(item, AssistantInput):
                sb.append("<|assistant|>\n")
                if item.content:
                    for part in item.content:
                        if isinstance(part, TextContent):
                            sb.append(part.text)
                if item.tool_calls:
                    for call in item.tool_calls:
                        sb.append(f'{{"tool_call": {{"name": "{call.name}", "arguments": {call.arguments}}}}}')
                sb.append("\n")
            elif isinstance(item, ToolResultInput):
                sb.append(f"<|tool|>\n{item.content}\n")

        sb.append("<|assistant|>\n")
        return "".join(sb)
