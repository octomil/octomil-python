"""Workflow data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


@dataclass
class InferenceStep:
    model: str
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None


@dataclass
class ToolRoundStep:
    tools: list[dict[str, Any]]
    model: str
    max_iterations: int = 5


@dataclass
class TransformStep:
    name: str
    transform: Callable[[str], Awaitable[str]] | Callable[[str], str]


WorkflowStep = InferenceStep | ToolRoundStep | TransformStep


@dataclass
class Workflow:
    name: str
    steps: list[WorkflowStep] = field(default_factory=list)


@dataclass
class WorkflowResult:
    outputs: list[Any] = field(default_factory=list)  # list[Response]
    total_latency_ms: float = 0.0
