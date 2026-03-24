"""Domain types for the chat conversation subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatThread:
    """A conversation thread containing messages."""

    id: str
    model: str
    created_at: str
    updated_at: str
    title: str | None = None
    binding_key: str | None = None
    storage_mode: str | None = None  # full, metadata_only, redacted, encrypted
    retention_policy: str | None = None  # persistent, ttl
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreadMessage:
    """A single message within a chat thread."""

    id: str
    thread_id: str
    role: str
    created_at: str
    content: str | None = None
    content_parts: list[dict[str, Any]] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    parent_message_id: str | None = None
    status: str | None = None  # pending, completed, failed, cancelled, expired
    model_ref: str | None = None
    storage_mode: str | None = None
    metrics: dict[str, Any] | None = None


@dataclass
class ToolCall:
    """Domain-level tool call entity."""

    id: str
    name: str
    message_id: str | None = None
    thread_id: str | None = None
    arguments: str | None = None
    arguments_ref: str | None = None
    status: str | None = None  # requested, started, succeeded, failed, expired
    started_at: str | None = None
    ended_at: str | None = None
    latency_ms: int | None = None
    error_code: str | None = None


@dataclass
class ToolResult:
    """Result of a tool call execution."""

    id: str
    tool_call_id: str
    message_id: str | None = None
    output: str | None = None
    output_ref: str | None = None
    status: str | None = None  # requested, started, succeeded, failed, expired
    size_bytes: int | None = None
    is_final: bool = True
