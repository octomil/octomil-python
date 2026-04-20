"""Tests for production-path planner integration in OctomilResponses and CLI.

Verifies that CandidateAttemptRunner is used when a planner plan is available,
and that backward compatibility is preserved when no plan exists.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from octomil.responses.responses import OctomilResponses
from octomil.responses.types import (
    DoneEvent,
    ResponseRequest,
    TextDeltaEvent,
    TextOutput,
    text_input,
)
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.registry import ModelRuntimeRegistry
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeResponse,
)
from octomil.runtime.routing.attempt_runner import CandidateAttemptRunner

# ---------------------------------------------------------------------------
# Mock runtime
# ---------------------------------------------------------------------------


class MockRuntime(ModelRuntime):
    """Simple mock runtime for testing."""

    def __init__(self, text: str = "mock response"):
        self._text = text

    @property
    def capabilities(self):
        return RuntimeCapabilities()

    async def run(self, request, **kwargs):
        return RuntimeResponse(text=self._text)

    async def stream(self, request, **kwargs):
        for word in self._text.split():
            yield RuntimeChunk(text=word + " ")
        yield RuntimeChunk(text="", finish_reason="stop")


class FailingRuntime(ModelRuntime):
    """Runtime that always fails."""

    @property
    def capabilities(self):
        return RuntimeCapabilities()

    async def run(self, request, **kwargs):
        raise RuntimeError("local model load failed")

    async def stream(self, request, **kwargs):
        raise RuntimeError("local stream failed")
        yield  # type: ignore[misc]  # noqa: E501 — make it a generator


class FailAfterTokenRuntime(ModelRuntime):
    """Runtime that fails after emitting the first token."""

    @property
    def capabilities(self):
        return RuntimeCapabilities()

    async def run(self, request, **kwargs):
        raise RuntimeError("should not be called in stream test")

    async def stream(self, request, **kwargs):
        yield RuntimeChunk(text="partial ")
        exc = RuntimeError("stream crashed after first token")
        exc.first_token_emitted = True  # type: ignore[attr-defined]
        raise exc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRuntimeRegistry.shared().clear()
    yield
    ModelRuntimeRegistry.shared().clear()


def _make_selection(
    locality: str = "local",
    engine: str | None = "mlx-lm",
    fallback_allowed: bool = True,
    candidates: list[dict[str, Any]] | None = None,
):
    """Create a mock RuntimeSelection for testing."""
    mock = MagicMock()
    mock.locality = locality
    mock.engine = engine
    mock.artifact = None
    mock.fallback_allowed = fallback_allowed
    mock.reason = "test selection"
    mock.app_resolution = None
    mock.source = "server_plan"

    if candidates is not None:
        mock_candidates = []
        for c in candidates:
            mc = MagicMock()
            mc.locality = c.get("locality", "local")
            mc.priority = c.get("priority", 0)
            mc.confidence = c.get("confidence", 1.0)
            mc.reason = c.get("reason", "")
            mc.engine = c.get("engine")
            mc.artifact = None
            mc.gates = []
            mock_candidates.append(mc)
        mock.candidates = mock_candidates
    else:
        mock.candidates = None

    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateUsesAttemptRunnerWhenPlanAvailable:
    @pytest.mark.asyncio
    async def test_create_uses_attempt_runner_when_plan_available(self) -> None:
        """When the planner returns a selection, create() should use the attempt runner."""
        mock_rt = MockRuntime("planned response")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        selection = _make_selection(locality="local", engine="mlx-lm")

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=selection,
        ):
            responses = OctomilResponses(planner_enabled=True)
            response = await responses.create(ResponseRequest(model="test-model", input=[text_input("Hello")]))

        assert len(response.output) > 0
        assert isinstance(response.output[0], TextOutput)
        assert response.output[0].text == "planned response"
        # Route metadata should be present
        assert response.route is not None


class TestCreateBypassesRunnerWhenNoPlan:
    @pytest.mark.asyncio
    async def test_create_bypasses_runner_when_no_plan(self) -> None:
        """When no plan is cached, create() should fall back to direct runtime call."""
        mock_rt = MockRuntime("direct response")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=None,
        ):
            responses = OctomilResponses(planner_enabled=True)
            response = await responses.create(ResponseRequest(model="test-model", input=[text_input("Hello")]))

        assert len(response.output) > 0
        assert isinstance(response.output[0], TextOutput)
        assert response.output[0].text == "direct response"
        # Route metadata should still be present (from fallback path)
        assert response.route is not None

    @pytest.mark.asyncio
    async def test_create_bypasses_runner_when_planner_disabled(self) -> None:
        """When planner_enabled=False, create() never invokes the planner."""
        mock_rt = MockRuntime("direct response")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
        ) as mock_planner:
            responses = OctomilResponses(planner_enabled=False)
            response = await responses.create(ResponseRequest(model="test-model", input=[text_input("Hello")]))

            mock_planner.assert_not_called()

        assert isinstance(response.output[0], TextOutput)
        assert response.output[0].text == "direct response"


class TestStreamUsesAttemptRunner:
    @pytest.mark.asyncio
    async def test_stream_uses_attempt_runner(self) -> None:
        """When a plan is available, stream() uses the attempt runner."""
        mock_rt = MockRuntime("streamed content here")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        selection = _make_selection(locality="local", engine="mlx-lm")

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=selection,
        ):
            responses = OctomilResponses(planner_enabled=True)
            events = []
            async for event in responses.stream(ResponseRequest(model="test-model", input=[text_input("Hello")])):
                events.append(event)

        # Should have text deltas and a done event
        text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]
        assert len(text_deltas) > 0
        assert len(done_events) == 1
        assert done_events[0].response.route is not None


class TestStreamFallbackBeforeFirstToken:
    @pytest.mark.asyncio
    async def test_stream_fallback_before_first_token(self) -> None:
        """If streaming fails before first token, fall back to next candidate."""
        call_count = {"n": 0}

        class _SwitchingRuntime(ModelRuntime):
            @property
            def capabilities(self):
                return RuntimeCapabilities()

            async def run(self, request, **kwargs):
                return RuntimeResponse(text="unused")

            async def stream(self, request, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("local stream failed before first token")
                # Second call succeeds (cloud fallback)
                yield RuntimeChunk(text="cloud ")
                yield RuntimeChunk(text="response")

        switching_rt = _SwitchingRuntime()
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: switching_rt

        selection = _make_selection(
            fallback_allowed=True,
            candidates=[
                {"locality": "local", "engine": "mlx-lm", "priority": 0, "confidence": 1.0, "reason": "primary"},
                {"locality": "cloud", "priority": 1, "confidence": 0.5, "reason": "fallback"},
            ],
        )

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=selection,
        ):
            responses = OctomilResponses(planner_enabled=True)
            events = []
            async for event in responses.stream(ResponseRequest(model="test-model", input=[text_input("Hello")])):
                events.append(event)

        done_events = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done_events) == 1
        # The response text should be from the cloud fallback
        text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        full_text = "".join(e.delta for e in text_deltas)
        assert "cloud" in full_text
        # Locality should reflect fallback
        assert done_events[0].response.locality is not None


class TestStreamNoFallbackAfterFirstToken:
    @pytest.mark.asyncio
    async def test_stream_no_fallback_after_first_token(self) -> None:
        """If streaming fails AFTER first token, do NOT fall back — raise error."""
        rt = FailAfterTokenRuntime()
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: rt

        selection = _make_selection(
            fallback_allowed=True,
            candidates=[
                {"locality": "local", "engine": "mlx-lm", "priority": 0, "confidence": 1.0, "reason": "primary"},
                {"locality": "cloud", "priority": 1, "confidence": 0.5, "reason": "fallback"},
            ],
        )

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=selection,
        ):
            responses = OctomilResponses(planner_enabled=True)
            with pytest.raises(RuntimeError, match="stream crashed after first token"):
                events = []
                async for event in responses.stream(ResponseRequest(model="test-model", input=[text_input("Hello")])):
                    events.append(event)


class TestResponseIncludesRouteMetadata:
    @pytest.mark.asyncio
    async def test_response_includes_route_metadata(self) -> None:
        """Both create() and stream() should include route metadata on the response."""
        mock_rt = MockRuntime("with metadata")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        selection = _make_selection(locality="local", engine="mlx-lm")

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=selection,
        ):
            responses = OctomilResponses(planner_enabled=True)

            # Test create
            response = await responses.create(ResponseRequest(model="test-model", input=[text_input("Hello")]))
            assert response.route is not None
            assert response.route.execution is not None
            assert response.route.execution.locality in ("local", "cloud")

            # Test stream
            events = []
            async for event in responses.stream(ResponseRequest(model="test-model", input=[text_input("Hello")])):
                events.append(event)

            done_events = [e for e in events if isinstance(e, DoneEvent)]
            assert len(done_events) == 1
            assert done_events[0].response.route is not None


class TestCliRunUsesAttemptRunner:
    """Verify the CLI run command's execution path uses the attempt runner via ExecutionKernel."""

    @pytest.mark.asyncio
    async def test_cli_run_uses_attempt_runner(self) -> None:
        """The CLI run command uses ExecutionKernel which wires CandidateAttemptRunner."""
        # The CLI delegates to ExecutionKernel.create_response, which already
        # uses CandidateAttemptRunner. We verify the kernel integration here.
        from octomil.execution.kernel import ExecutionKernel

        _unused_rt = MockRuntime("kernel response")  # noqa: F841

        with (
            patch.object(ExecutionKernel, "_build_router") as mock_build_router,
            patch(
                "octomil.execution.kernel._resolve_planner_selection",
                return_value=_make_selection(locality="local"),
            ),
        ):
            # Set up the mock router to return our response
            mock_router = AsyncMock()
            mock_router.run = AsyncMock(return_value=RuntimeResponse(text="kernel response"))
            mock_router.resolve_locality = MagicMock(return_value=("on_device", False))
            mock_build_router.return_value = mock_router

            kernel = ExecutionKernel()
            # Patch _resolve to return defaults
            mock_defaults = MagicMock()
            mock_defaults.model = "test-model"
            mock_defaults.policy_preset = "local_first"
            mock_defaults.inline_policy = None
            mock_defaults.cloud_profile = None

            with patch.object(kernel, "_resolve", return_value=mock_defaults):
                result = await kernel.create_response("Hello", model="test-model")

            assert result.output_text == "kernel response"
            # The kernel should have called _build_router (which is invoked
            # through the CandidateAttemptRunner's execute_candidate callback)
            assert mock_build_router.called


class TestAttemptRunnerIsInvokedNotBypassed:
    """Ensures that if CandidateAttemptRunner exists, public paths do NOT bypass it."""

    @pytest.mark.asyncio
    async def test_create_invokes_runner_when_plan_exists(self) -> None:
        """Fails if OctomilResponses.create() bypasses CandidateAttemptRunner when a plan is available."""
        mock_rt = MockRuntime("runner output")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        selection = _make_selection(locality="local", engine="mlx-lm")

        with (
            patch(
                "octomil.responses.responses._try_resolve_planner_selection",
                return_value=selection,
            ),
            patch(
                "octomil.responses.responses.CandidateAttemptRunner.run_with_inference",
                wraps=CandidateAttemptRunner(fallback_allowed=True).run_with_inference,
            ) as mock_runner,
        ):
            responses = OctomilResponses(planner_enabled=True)
            await responses.create(ResponseRequest(model="test-model", input=[text_input("Hello")]))

            # The attempt runner MUST have been called
            assert mock_runner.called
