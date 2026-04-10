"""Tests for RouterModelRuntime and RoutingPolicy."""

from __future__ import annotations

import pytest

from octomil._generated.message_role import MessageRole
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.policy import RoutingPolicy
from octomil.runtime.core.router import RouterModelRuntime
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeContentPart,
    RuntimeMessage,
    RuntimeRequest,
    RuntimeResponse,
)


def _text_request(text: str = "test") -> RuntimeRequest:
    """Build a minimal text-only RuntimeRequest for tests."""
    return RuntimeRequest(messages=[RuntimeMessage(role=MessageRole.USER, parts=[RuntimeContentPart.text_part(text)])])


class StubRuntime(ModelRuntime):
    def __init__(self, name: str):
        self.name = name

    @property
    def capabilities(self):
        return RuntimeCapabilities()

    async def run(self, request):
        return RuntimeResponse(text=f"from-{self.name}")

    async def stream(self, request):
        yield RuntimeChunk(text=f"from-{self.name}")


@pytest.mark.asyncio
async def test_auto_prefers_local():
    router = RouterModelRuntime(
        local_factory=lambda mid: StubRuntime("local"),
        cloud_factory=lambda mid: StubRuntime("cloud"),
    )
    resp = await router.run(_text_request())
    assert resp.text == "from-local"


@pytest.mark.asyncio
async def test_auto_fallback_to_cloud():
    router = RouterModelRuntime(
        local_factory=lambda mid: None,
        cloud_factory=lambda mid: StubRuntime("cloud"),
    )
    resp = await router.run(_text_request())
    assert resp.text == "from-cloud"


@pytest.mark.asyncio
async def test_local_only_raises_when_no_local():
    router = RouterModelRuntime(
        local_factory=lambda mid: None,
        default_policy=RoutingPolicy.local_only(),
    )
    with pytest.raises(RuntimeError, match="No local runtime"):
        await router.run(_text_request())


@pytest.mark.asyncio
async def test_cloud_only_uses_cloud():
    router = RouterModelRuntime(
        cloud_factory=lambda mid: StubRuntime("cloud"),
        default_policy=RoutingPolicy.cloud_only(),
    )
    resp = await router.run(_text_request())
    assert resp.text == "from-cloud"


@pytest.mark.asyncio
async def test_cloud_only_raises_when_no_cloud():
    router = RouterModelRuntime(
        cloud_factory=lambda mid: None,
        default_policy=RoutingPolicy.cloud_only(),
    )
    with pytest.raises(RuntimeError, match="No cloud runtime"):
        await router.run(_text_request())


@pytest.mark.asyncio
async def test_auto_raises_when_no_runtime():
    router = RouterModelRuntime(
        local_factory=lambda mid: None,
        cloud_factory=lambda mid: None,
    )
    with pytest.raises(RuntimeError, match="No runtime available"):
        await router.run(_text_request())


@pytest.mark.asyncio
async def test_stream_routes_correctly():
    router = RouterModelRuntime(
        local_factory=lambda mid: StubRuntime("local"),
        cloud_factory=lambda mid: StubRuntime("cloud"),
    )
    chunks = []
    async for chunk in router.stream(_text_request()):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].text == "from-local"


def test_from_metadata_parses_auto():
    policy = RoutingPolicy.from_metadata({"routing.policy": "auto", "routing.prefer_local": "true"})
    assert policy is not None
    assert policy.mode == "auto"
    assert policy.prefer_local is True


def test_from_metadata_parses_local_only():
    policy = RoutingPolicy.from_metadata({"routing.policy": "local_only"})
    assert policy is not None
    assert policy.mode == "local_only"


def test_from_metadata_parses_cloud_only():
    policy = RoutingPolicy.from_metadata({"routing.policy": "cloud_only"})
    assert policy is not None
    assert policy.mode == "cloud_only"


def test_from_metadata_parses_max_latency():
    policy = RoutingPolicy.from_metadata(
        {
            "routing.policy": "auto",
            "routing.max_latency_ms": "500",
            "routing.fallback": "none",
        }
    )
    assert policy is not None
    assert policy.max_latency_ms == 500
    assert policy.fallback == "none"


def test_from_metadata_returns_none_for_missing():
    assert RoutingPolicy.from_metadata(None) is None
    assert RoutingPolicy.from_metadata({}) is None


def test_from_metadata_returns_none_for_unknown_mode():
    assert RoutingPolicy.from_metadata({"routing.policy": "unknown_mode"}) is None


# ---------------------------------------------------------------------------
# from_desired_state_entry
# ---------------------------------------------------------------------------


class TestFromDesiredStateEntry:
    """Verify RoutingPolicy.from_desired_state_entry maps serving policy presets."""

    def test_private_local_only(self):
        entry = {"serving_policy": {"routing_mode": "local_only"}}
        policy = RoutingPolicy.from_desired_state_entry(entry)
        assert policy is not None
        assert policy.mode == "local_only"

    def test_cloud_only(self):
        entry = {"serving_policy": {"routing_mode": "cloud_only"}}
        policy = RoutingPolicy.from_desired_state_entry(entry)
        assert policy is not None
        assert policy.mode == "cloud_only"

    def test_local_first_preference(self):
        entry = {
            "serving_policy": {
                "routing_mode": "auto",
                "routing_preference": "local",
                "fallback": {"allow_cloud_fallback": True},
            }
        }
        policy = RoutingPolicy.from_desired_state_entry(entry)
        assert policy is not None
        assert policy.mode == "local_first"
        assert policy.prefer_local is True
        assert policy.fallback == "cloud"

    def test_performance_preference(self):
        entry = {
            "serving_policy": {
                "routing_mode": "auto",
                "routing_preference": "performance",
                "fallback": {"allow_cloud_fallback": True},
            }
        }
        policy = RoutingPolicy.from_desired_state_entry(entry)
        assert policy is not None
        assert policy.mode == "auto"
        assert policy.prefer_local is True
        assert policy.fallback == "cloud"

    def test_quality_preference(self):
        entry = {
            "serving_policy": {
                "routing_mode": "auto",
                "routing_preference": "quality",
                "fallback": {"allow_local_fallback": True},
            }
        }
        policy = RoutingPolicy.from_desired_state_entry(entry)
        assert policy is not None
        assert policy.mode == "auto"
        assert policy.prefer_local is False
        assert policy.fallback == "local"

    def test_cloud_preference(self):
        entry = {"serving_policy": {"routing_mode": "auto", "routing_preference": "cloud"}}
        policy = RoutingPolicy.from_desired_state_entry(entry)
        assert policy is not None
        assert policy.mode == "auto"
        assert policy.prefer_local is False
        assert policy.fallback == "local"

    def test_local_only_disables_fallback(self):
        entry = {"serving_policy": {"routing_mode": "local_only"}}
        policy = RoutingPolicy.from_desired_state_entry(entry)
        assert policy is not None
        assert policy.mode == "local_only"

    def test_no_serving_policy_returns_none(self):
        policy = RoutingPolicy.from_desired_state_entry({})
        assert policy is None

    def test_fallback_disabled(self):
        entry = {
            "serving_policy": {
                "routing_mode": "auto",
                "routing_preference": "local",
                "fallback": {"allow_cloud_fallback": False},
            }
        }
        policy = RoutingPolicy.from_desired_state_entry(entry)
        assert policy is not None
        assert policy.fallback == "none"

    def test_no_fallback_key_defaults_cloud(self):
        entry = {"serving_policy": {"routing_mode": "auto", "routing_preference": "performance"}}
        policy = RoutingPolicy.from_desired_state_entry(entry)
        assert policy is not None
        assert policy.fallback == "cloud"

    def test_cloud_preference_can_disable_local_fallback(self):
        entry = {
            "serving_policy": {
                "routing_mode": "auto",
                "routing_preference": "cloud",
                "fallback": {"allow_local_fallback": False},
            }
        }
        policy = RoutingPolicy.from_desired_state_entry(entry)
        assert policy is not None
        assert policy.fallback == "none"


# ---------------------------------------------------------------------------
# Runtime decision tests: quality vs balanced behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quality_prefers_cloud_when_both_available():
    """Quality preset (prefer_local=False) should pick cloud over local."""
    router = RouterModelRuntime(
        local_factory=lambda mid: StubRuntime("local"),
        cloud_factory=lambda mid: StubRuntime("cloud"),
        default_policy=RoutingPolicy.auto(prefer_local=False),
    )
    resp = await router.run(_text_request())
    assert resp.text == "from-cloud"


@pytest.mark.asyncio
async def test_quality_falls_back_to_local_when_no_cloud():
    """Quality preset falls back to local when cloud is unavailable."""
    router = RouterModelRuntime(
        local_factory=lambda mid: StubRuntime("local"),
        cloud_factory=lambda mid: None,
        default_policy=RoutingPolicy.auto(prefer_local=False),
    )
    resp = await router.run(_text_request())
    assert resp.text == "from-local"


@pytest.mark.asyncio
async def test_cloud_first_respects_disabled_local_fallback():
    router = RouterModelRuntime(
        local_factory=lambda mid: StubRuntime("local"),
        cloud_factory=lambda mid: None,
        default_policy=RoutingPolicy.auto(prefer_local=False, fallback="none"),
    )
    with pytest.raises(RuntimeError, match="No runtime available"):
        await router.run(_text_request())


@pytest.mark.asyncio
async def test_balanced_still_prefers_local():
    """Balanced preset (prefer_local=True) should pick local first."""
    router = RouterModelRuntime(
        local_factory=lambda mid: StubRuntime("local"),
        cloud_factory=lambda mid: StubRuntime("cloud"),
        default_policy=RoutingPolicy.auto(prefer_local=True),
    )
    resp = await router.run(_text_request())
    assert resp.text == "from-local"


@pytest.mark.asyncio
async def test_quality_resolve_locality():
    """resolve_locality with quality preset returns cloud, is_fallback=False."""
    router = RouterModelRuntime(
        local_factory=lambda mid: StubRuntime("local"),
        cloud_factory=lambda mid: StubRuntime("cloud"),
        default_policy=RoutingPolicy.auto(prefer_local=False),
    )
    locality, is_fallback = router.resolve_locality()
    assert locality == "cloud"
    assert is_fallback is False
