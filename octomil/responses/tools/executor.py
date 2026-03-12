"""ToolExecutor — host-provided interface for executing tool calls."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from ..types import ResponseToolCall


class ToolExecutor(abc.ABC):
    """Host-provided interface for executing tool calls (Layer 3).

    The SDK invokes tools but does NOT execute them — that's the host app's job.
    """

    @abc.abstractmethod
    async def execute(self, call: ResponseToolCall) -> ToolResult: ...


@dataclass
class ToolResult:
    """The result of executing a tool call."""

    tool_call_id: str
    content: str
    is_error: bool = False
