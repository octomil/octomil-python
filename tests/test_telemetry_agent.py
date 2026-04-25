"""Tests for the SDK telemetry agent (PR 7 MVP)."""

from __future__ import annotations

import asyncio

import pytest

from octomil.runtime.telemetry_agent import (
    TelemetryAgent,
    TelemetryAgentConfig,
)


class _RecordingSender:
    def __init__(self) -> None:
        self.batches: list[list[dict]] = []
        self.fail_next: int = 0

    async def __call__(self, events: list[dict]) -> None:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("simulated upload failure")
        self.batches.append(list(events))


@pytest.mark.asyncio
async def test_queue_and_flush_roundtrip():
    sender = _RecordingSender()
    agent = TelemetryAgent(sender)
    agent.record({"event": "runtime.route.completed", "request_id": "r1"})
    agent.record({"event": "runtime.execution.completed", "request_id": "r2"})

    assert agent.queue_size == 2
    await agent.flush()

    assert agent.queue_size == 0
    assert len(sender.batches) == 1
    assert {e["event"] for e in sender.batches[0]} == {
        "runtime.route.completed",
        "runtime.execution.completed",
    }


@pytest.mark.asyncio
async def test_disabled_agent_is_noop(monkeypatch):
    sender = _RecordingSender()
    agent = TelemetryAgent(sender, config=TelemetryAgentConfig(enabled=False))
    agent.record({"event": "runtime.route.completed"})
    await agent.flush()
    assert sender.batches == []
    assert agent.queue_size == 0


@pytest.mark.asyncio
async def test_env_var_disables_agent(monkeypatch):
    monkeypatch.setenv("OCTOMIL_TELEMETRY_DISABLED", "1")
    sender = _RecordingSender()
    agent = TelemetryAgent(sender)
    assert agent.enabled is False
    agent.record({"event": "runtime.route.completed"})
    await agent.flush()
    assert sender.batches == []


@pytest.mark.asyncio
async def test_forbidden_key_event_dropped():
    sender = _RecordingSender()
    agent = TelemetryAgent(sender)
    agent.record(
        {
            "event": "runtime.route.completed",
            "prompt": "hello — must not leak",
        }
    )
    # Sanitize strips the forbidden key but the event itself still
    # records, so the queue size is 1 and no exception escapes.
    await agent.flush()
    assert len(sender.batches) == 1
    assert "prompt" not in sender.batches[0][0]


@pytest.mark.asyncio
async def test_failed_upload_does_not_raise():
    sender = _RecordingSender()
    sender.fail_next = 1
    agent = TelemetryAgent(sender)
    agent.record({"event": "runtime.route.completed", "request_id": "r1"})
    # Must not raise.
    await agent.flush()
    # Failed batch is re-queued for the next attempt.
    assert agent.queue_size == 1
    await agent.flush()
    assert agent.queue_size == 0
    assert len(sender.batches) == 1


@pytest.mark.asyncio
async def test_flush_threshold_triggers_send():
    sender = _RecordingSender()
    agent = TelemetryAgent(
        sender,
        config=TelemetryAgentConfig(flush_threshold=2, flush_interval_s=60),
    )
    agent.record({"event": "runtime.route.completed", "request_id": "r1"})
    agent.record({"event": "runtime.route.completed", "request_id": "r2"})
    # Give the auto-flush task a tick.
    for _ in range(20):
        await asyncio.sleep(0)
        if sender.batches:
            break
    assert sender.batches  # auto-flushed at least once


@pytest.mark.asyncio
async def test_no_prompts_in_payload():
    sender = _RecordingSender()
    agent = TelemetryAgent(sender)
    agent.record(
        {
            "event": "runtime.route.completed",
            "metadata": {"engine": "llamacpp", "messages": "leak"},
        }
    )
    await agent.flush()
    assert sender.batches
    body = sender.batches[0][0]
    # Forbidden nested key was scrubbed.
    assert "messages" not in body.get("metadata", {})


@pytest.mark.asyncio
async def test_secret_keys_stripped_case_insensitively():
    """Authorization headers, API keys, tokens, passwords MUST never leave
    the device. Match must be case-insensitive — earlier impl let
    'Authorization' (capital A) slip through."""
    sender = _RecordingSender()
    agent = TelemetryAgent(sender)
    agent.record(
        {
            "event": "runtime.route.completed",
            "Authorization": "Bearer sk-secret",
            "API_KEY": "sk-secret",
            "metadata": {
                "Token": "abc",
                "password": "hunter2",
                "engine": "llamacpp",
            },
        }
    )
    await agent.flush()

    assert sender.batches
    body = sender.batches[0][0]
    for forbidden in [
        "Authorization",
        "authorization",
        "API_KEY",
        "api_key",
        "Token",
        "token",
        "password",
    ]:
        assert forbidden not in body
        assert forbidden not in body.get("metadata", {})
    assert body["metadata"]["engine"] == "llamacpp"


@pytest.mark.asyncio
async def test_secrets_stripped_at_arbitrary_depth():
    sender = _RecordingSender()
    agent = TelemetryAgent(sender)
    agent.record(
        {
            "event": "runtime.route.completed",
            "metadata": {
                "outer": {
                    "inner": {
                        "Authorization": "Bearer sk-secret",
                        "engine": "llamacpp",
                    }
                }
            },
        }
    )
    await agent.flush()
    body = sender.batches[0][0]
    inner = body["metadata"]["outer"]["inner"]
    assert "Authorization" not in inner
    assert "authorization" not in inner
    assert inner["engine"] == "llamacpp"
