"""AgentSession — high-level agent session for client-side LLM + server tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from ..responses.responses import OctomilResponses
from ..responses.tools.remote_executor import RemoteToolExecutor
from ..responses.tools.runner import ToolRunner
from ..responses.types import ResponseRequest, SystemInput, TextOutput, text_input


@dataclass
class AgentResult:
    """Result of a completed agent session."""

    session_id: str
    summary: str
    confidence: float | None = None
    evidence: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


class AgentSession:
    """Run an agent loop: model runs on-device, server provides tools.

    Usage::

        session = AgentSession(base_url="https://api.octomil.com", auth_token="...")
        result = await session.run("deployment_advisor", "Deploy phi-mini to iOS staging")
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        *,
        responses: OctomilResponses | None = None,
        max_iterations: int = 10,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._responses = responses or OctomilResponses()
        self._max_iterations = max_iterations

    async def run(
        self,
        agent_type: str,
        query: str,
        context: dict[str, Any] | None = None,
        *,
        model: str = "qwen2.5:7b",
    ) -> AgentResult:
        """Execute a full agent session.

        1. POST /sessions → get session_id, tools, system_prompt
        2. Build ResponseRequest with system_prompt, query, tools
        3. Create RemoteToolExecutor + ToolRunner
        4. runner.run(request) → Response
        5. POST /sessions/{id}/complete with parsed result
        6. Return AgentResult
        """
        headers = {"Authorization": f"Bearer {self._auth_token}"}

        # 1. Create session
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/v1/agents/sessions",
                json={
                    "agent_type": agent_type,
                    "query": query,
                    "context": context or {},
                },
                headers=headers,
            )
            resp.raise_for_status()
            session_data = resp.json()

        session_id = session_data["session_id"]
        tools = session_data["tools"]
        system_prompt = session_data["system_prompt"]

        # 2. Build request
        request = ResponseRequest(
            model=model,
            input=[
                SystemInput(content=system_prompt),
                text_input(query),
            ],
            tools=tools,
        )

        # 3. Create executor + runner
        executor = RemoteToolExecutor(
            base_url=self._base_url,
            session_id=session_id,
            auth_token=self._auth_token,
        )
        runner = ToolRunner(
            self._responses,
            executor,
            max_iterations=self._max_iterations,
        )

        # 4. Run agent loop
        response = await runner.run(request)

        # 5. Extract result text
        texts = [item.text for item in response.output if isinstance(item, TextOutput)]
        summary = "\n".join(texts) if texts else "No response from agent."

        # Parse structured output if present
        parsed = _parse_result(summary)

        # 6. Complete session
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{self._base_url}/api/v1/agents/sessions/{session_id}/complete",
                    json={
                        "summary": parsed["summary"],
                        "confidence": parsed["confidence"],
                        "evidence": parsed["evidence"],
                        "next_steps": parsed["next_steps"],
                    },
                    headers=headers,
                )
        except httpx.HTTPError:
            pass  # Best-effort completion; session result is already available

        return AgentResult(
            session_id=session_id,
            summary=parsed["summary"],
            confidence=parsed["confidence"],
            evidence=parsed["evidence"],
            next_steps=parsed["next_steps"],
        )


def _parse_result(text: str) -> dict[str, Any]:
    """Extract structured fields from agent output text."""
    result: dict[str, Any] = {
        "summary": text[:500] if text else "No response",
        "confidence": None,
        "evidence": [],
        "next_steps": [],
    }

    lines = text.strip().split("\n")
    current_section: str | None = None
    current_items: list[str] = []

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("## summary"):
            _flush(result, current_section, current_items)
            current_section = "summary"
            current_items = []
        elif lower.startswith("## detail") or lower.startswith("## evidence"):
            _flush(result, current_section, current_items)
            current_section = "evidence"
            current_items = []
        elif lower.startswith("## risk"):
            _flush(result, current_section, current_items)
            current_section = "risk"
            current_items = []
        elif lower.startswith("## recommended") or lower.startswith("## next"):
            _flush(result, current_section, current_items)
            current_section = "next_steps"
            current_items = []
        elif stripped:
            current_items.append(stripped)

    _flush(result, current_section, current_items)
    return result


def _flush(result: dict[str, Any], section: str | None, items: list[str]) -> None:
    if not section or not items:
        return

    if section == "summary":
        result["summary"] = " ".join(items)
    elif section == "evidence":
        result["evidence"] = [item.lstrip("- *") for item in items if item.startswith(("-", "*"))] or items
    elif section == "risk":
        for item in items:
            if "confidence" in item.lower() and ":" in item:
                try:
                    val = float(item.split(":", 1)[1].strip().rstrip("%"))
                    result["confidence"] = val / 100.0 if val > 1.0 else val
                except (ValueError, IndexError):
                    pass
    elif section == "next_steps":
        result["next_steps"] = [item.lstrip("- *0123456789. ") for item in items if item.strip()]
