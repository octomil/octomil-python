"""Tests for Layer 1: ModelRuntime registry and adapter."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from octomil.runtime.core import (
    ModelRuntime,
    ModelRuntimeRegistry,
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
)


class StubRuntime(ModelRuntime):
    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        return RuntimeResponse(text="stub")

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        yield RuntimeChunk(text="stub")
        return


@pytest.fixture(autouse=True)
def _clean_registry():
    ModelRuntimeRegistry.shared().clear()
    yield
    ModelRuntimeRegistry.shared().clear()


def test_resolve_returns_none_when_empty():
    assert ModelRuntimeRegistry.shared().resolve("any-model") is None


def test_resolve_exact_family():
    ModelRuntimeRegistry.shared().register("phi-4-mini", lambda _: StubRuntime())
    assert ModelRuntimeRegistry.shared().resolve("phi-4-mini") is not None


def test_resolve_prefix_match():
    ModelRuntimeRegistry.shared().register("phi", lambda _: StubRuntime())
    assert ModelRuntimeRegistry.shared().resolve("phi-4-mini") is not None


def test_resolve_prefers_exact_over_prefix():
    used_exact = False

    def exact_factory(_: str) -> StubRuntime:
        nonlocal used_exact
        used_exact = True
        return StubRuntime()

    ModelRuntimeRegistry.shared().register("phi-4-mini", exact_factory)
    ModelRuntimeRegistry.shared().register("phi", lambda _: StubRuntime())
    ModelRuntimeRegistry.shared().resolve("phi-4-mini")
    assert used_exact


def test_resolve_falls_back_to_default():
    ModelRuntimeRegistry.shared().default_factory = lambda _: StubRuntime()
    assert ModelRuntimeRegistry.shared().resolve("unknown") is not None


def test_resolve_returns_none_no_match():
    ModelRuntimeRegistry.shared().register("phi", lambda _: StubRuntime())
    assert ModelRuntimeRegistry.shared().resolve("llama-3") is None


def test_resolve_case_insensitive():
    ModelRuntimeRegistry.shared().register("PHI", lambda _: StubRuntime())
    assert ModelRuntimeRegistry.shared().resolve("phi-4-mini") is not None


def test_clear_removes_all():
    ModelRuntimeRegistry.shared().register("phi", lambda _: StubRuntime())
    ModelRuntimeRegistry.shared().default_factory = lambda _: StubRuntime()
    ModelRuntimeRegistry.shared().clear()
    assert ModelRuntimeRegistry.shared().resolve("phi") is None
