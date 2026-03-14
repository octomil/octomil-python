"""RemoteToolExecutor — executes tool calls via the Octomil server tool API."""

from __future__ import annotations

import json

import httpx

from ..types import ResponseToolCall
from .executor import ToolExecutor, ToolResult


class RemoteToolExecutor(ToolExecutor):
    """Executes tool calls by calling the Octomil server's tool session API.

    Each call is a POST to ``/api/v1/agents/sessions/{session_id}/execute``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        session_id: str,
        auth_token: str,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session_id = session_id
        self._auth_token = auth_token

    async def execute(self, call: ResponseToolCall) -> ToolResult:
        url = f"{self._base_url}/api/v1/agents/sessions/{self._session_id}/execute"
        headers = {"Authorization": f"Bearer {self._auth_token}"}
        payload = {
            "tool_name": call.name,
            "arguments": json.loads(call.arguments),
            "tool_call_id": call.id,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(
                    tool_call_id=data["tool_call_id"],
                    content=data["content"],
                    is_error=data.get("is_error", False),
                )
        except httpx.HTTPStatusError as e:
            return ToolResult(
                tool_call_id=call.id,
                content=f"HTTP {e.response.status_code}: {e.response.text}",
                is_error=True,
            )
        except (httpx.RequestError, json.JSONDecodeError, KeyError) as e:
            return ToolResult(
                tool_call_id=call.id,
                content=f"Remote tool execution failed: {e}",
                is_error=True,
            )
