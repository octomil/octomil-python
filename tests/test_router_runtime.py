"""Tests for RouterModelRuntime and RoutingPolicy."""

from __future__ import annotations

import pytest

from octomil.responses.runtime.model_runtime import ModelRuntime
from octomil.responses.runtime.policy import RoutingPolicy
from octomil.responses.runtime.router import RouterModelRuntime
from octomil.responses.runtime.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
)


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
    resp = await router.run(RuntimeRequest(prompt="test"))
    assert resp.text == "from-local"


@pytest.mark.asyncio
async def test_auto_fallback_to_cloud():
    router = RouterModelRuntime(
        local_factory=lambda mid: None,
        cloud_factory=lambda mid: StubRuntime("cloud"),
    )
    resp = await router.run(RuntimeRequest(prompt="test"))
    assert resp.text == "from-cloud"


@pytest.mark.asyncio
async def test_local_only_raises_when_no_local():
    router = RouterModelRuntime(
        local_factory=lambda mid: None,
        default_policy=RoutingPolicy.local_only(),
    )
    with pytest.raises(RuntimeError, match="No local runtime"):
        await router.run(RuntimeRequest(prompt="test"))


@pytest.mark.asyncio
async def test_cloud_only_uses_cloud():
    router = RouterModelRuntime(
        cloud_factory=lambda mid: StubRuntime("cloud"),
        default_policy=RoutingPolicy.cloud_only(),
    )
    resp = await router.run(RuntimeRequest(prompt="test"))
    assert resp.text == "from-cloud"


@pytest.mark.asyncio
async def test_cloud_only_raises_when_no_cloud():
    router = RouterModelRuntime(
        cloud_factory=lambda mid: None,
        default_policy=RoutingPolicy.cloud_only(),
    )
    with pytest.raises(RuntimeError, match="No cloud runtime"):
        await router.run(RuntimeRequest(prompt="test"))


@pytest.mark.asyncio
async def test_auto_raises_when_no_runtime():
    router = RouterModelRuntime(
        local_factory=lambda mid: None,
        cloud_factory=lambda mid: None,
    )
    with pytest.raises(RuntimeError, match="No runtime available"):
        await router.run(RuntimeRequest(prompt="test"))


@pytest.mark.asyncio
async def test_stream_routes_correctly():
    router = RouterModelRuntime(
        local_factory=lambda mid: StubRuntime("local"),
        cloud_factory=lambda mid: StubRuntime("cloud"),
    )
    chunks = []
    async for chunk in router.stream(RuntimeRequest(prompt="test")):
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
