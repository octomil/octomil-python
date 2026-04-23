"""Tests for production-path planner integration in OctomilResponses and CLI.

Verifies that CandidateAttemptRunner is used when a planner plan is available,
and that backward compatibility is preserved when no plan exists.
"""

from __future__ import annotations

from types import SimpleNamespace
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
    engine: str | None = None,
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

        selection = _make_selection(locality="local")

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

    @pytest.mark.asyncio
    async def test_create_uses_planner_selected_engine_and_resolved_app_model(self) -> None:
        """A selected local candidate must drive the concrete engine/model used."""
        created_for: list[str] = []

        class _Backend:
            def generate(self, request):
                return "engine-bound response", SimpleNamespace(prompt_tokens=1, total_tokens=4)

        class _Engine:
            name = "fake-engine"

            def detect(self):
                return True

            def create_backend(self, model_name: str):
                created_for.append(model_name)
                return _Backend()

        class _Registry:
            def get_engine(self, name: str):
                return _Engine() if name == "fake-engine" else None

        app_resolution = MagicMock()
        app_resolution.selected_model = "resolved-chat-model"
        app_resolution.selected_model_variant_id = "variant-a"
        app_resolution.selected_model_version = "1.0.0"

        selection = _make_selection(
            locality="local",
            engine="fake-engine",
            candidates=[
                {
                    "locality": "local",
                    "engine": "fake-engine",
                    "priority": 0,
                    "confidence": 1.0,
                    "reason": "planner selected fake-engine",
                }
            ],
        )
        selection.app_resolution = app_resolution

        with (
            patch("octomil.responses.responses._try_resolve_planner_selection", return_value=selection),
            patch("octomil.runtime.engines.get_registry", return_value=_Registry()),
        ):
            responses = OctomilResponses(planner_enabled=True)
            response = await responses.create(ResponseRequest(model="@app/demo/chat", input=[text_input("Hello")]))

        assert isinstance(response.output[0], TextOutput)
        assert response.output[0].text == "engine-bound response"
        assert created_for == ["resolved-chat-model"]
        assert response.route is not None
        assert response.route.execution is not None
        assert response.route.execution.engine == "fake-engine"

    @pytest.mark.asyncio
    async def test_create_rejects_ollama_candidate_and_uses_cloud_fallback(self) -> None:
        """Ollama must never be selected as an Octomil runtime engine."""
        cloud_rt = MockRuntime("cloud fallback response")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: cloud_rt

        selection = _make_selection(
            fallback_allowed=True,
            candidates=[
                {
                    "locality": "local",
                    "engine": "ollama",
                    "priority": 0,
                    "confidence": 1.0,
                    "reason": "legacy planner candidate",
                },
                {
                    "locality": "cloud",
                    "engine": "cloud",
                    "priority": 1,
                    "confidence": 0.5,
                    "reason": "policy-gated cloud fallback",
                },
            ],
        )

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=selection,
        ):
            responses = OctomilResponses(planner_enabled=True)
            response = await responses.create(ResponseRequest(model="test-model", input=[text_input("Hello")]))

        assert isinstance(response.output[0], TextOutput)
        assert response.output[0].text == "cloud fallback response"
        assert response.route is not None
        assert response.route.execution is not None
        assert response.route.execution.locality == "cloud"
        assert response.route.fallback.used is True
        assert response.route.attempts[0]["engine"] == "ollama"
        assert response.route.attempts[0]["status"] == "failed"
        assert response.route.attempts[0]["gate_results"][0]["reason_code"] == "engine_not_supported"

    @pytest.mark.asyncio
    async def test_create_fails_closed_when_ollama_is_only_candidate(self) -> None:
        """A non-fallback local-only Ollama plan should fail clearly, not drift to another route."""
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: MockRuntime("should not run")

        selection = _make_selection(
            fallback_allowed=False,
            candidates=[
                {
                    "locality": "local",
                    "engine": "ollama",
                    "priority": 0,
                    "confidence": 1.0,
                    "reason": "legacy planner candidate",
                }
            ],
        )

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=selection,
        ):
            responses = OctomilResponses(planner_enabled=True)
            with pytest.raises(RuntimeError, match="ollama not available: engine_not_supported"):
                await responses.create(ResponseRequest(model="test-model", input=[text_input("Hello")]))


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

    @pytest.mark.asyncio
    async def test_synthetic_cloud_fallback_defers_to_direct_runtime(self) -> None:
        """Offline planner fallback must not override an available runtime."""
        mock_rt = MockRuntime("direct local response")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt
        selection = _make_selection(locality="cloud")
        selection.source = "fallback"
        selection.engine = None
        selection.candidates = []

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=selection,
        ):
            responses = OctomilResponses(planner_enabled=True)
            response = await responses.create(ResponseRequest(model="test-model", input=[text_input("Hello")]))

        assert isinstance(response.output[0], TextOutput)
        assert response.output[0].text == "direct local response"
        assert response.locality == "on_device"


class TestStreamUsesAttemptRunner:
    @pytest.mark.asyncio
    async def test_stream_uses_attempt_runner(self) -> None:
        """When a plan is available, stream() uses the attempt runner."""
        mock_rt = MockRuntime("streamed content here")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        selection = _make_selection(locality="local")

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
        assert done_events[0].response.route.attempts


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
                {"locality": "local", "engine": None, "priority": 0, "confidence": 1.0, "reason": "primary"},
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
        route = done_events[0].response.route
        assert route is not None
        assert route.fallback.used is True
        assert [attempt["status"] for attempt in route.attempts] == ["failed", "selected"]
        assert route.fallback.trigger is not None
        assert route.fallback.trigger["code"] == "inference_error_before_first_token"


class TestStreamNoFallbackAfterFirstToken:
    @pytest.mark.asyncio
    async def test_stream_no_fallback_after_first_token(self) -> None:
        """If streaming fails AFTER first token, do NOT fall back — raise error."""
        rt = FailAfterTokenRuntime()
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: rt

        selection = _make_selection(
            fallback_allowed=True,
            candidates=[
                {"locality": "local", "engine": None, "priority": 0, "confidence": 1.0, "reason": "primary"},
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

        selection = _make_selection(locality="local")

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


class TestRefKindInRouteMetadata:
    """Verify that model reference kinds propagate correctly through the production path."""

    @pytest.mark.asyncio
    async def test_deployment_ref_kind_in_route_metadata(self) -> None:
        """deploy_xxx model ref should produce kind='deployment' in route metadata."""
        mock_rt = MockRuntime("deployed response")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=None,
        ):
            responses = OctomilResponses(planner_enabled=False)
            response = await responses.create(ResponseRequest(model="deploy_abc123", input=[text_input("Hi")]))

        assert response.route is not None
        assert response.route.model.requested.kind == "deployment"
        assert response.route.model.requested.ref == "deploy_abc123"

    @pytest.mark.asyncio
    async def test_experiment_ref_kind_in_route_metadata(self) -> None:
        """exp_xxx/variant model ref should produce kind='experiment' in route metadata."""
        mock_rt = MockRuntime("experiment response")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=None,
        ):
            responses = OctomilResponses(planner_enabled=False)
            response = await responses.create(ResponseRequest(model="exp_v1/variant_a", input=[text_input("Hi")]))

        assert response.route is not None
        assert response.route.model.requested.kind == "experiment"

    @pytest.mark.asyncio
    async def test_app_ref_kind_in_route_metadata(self) -> None:
        """@app/slug/cap model ref should produce kind='app' in route metadata."""
        mock_rt = MockRuntime("app response")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=None,
        ):
            responses = OctomilResponses(planner_enabled=False)
            response = await responses.create(ResponseRequest(model="@app/my-app/chat", input=[text_input("Hi")]))

        assert response.route is not None
        assert response.route.model.requested.kind == "app"
        assert response.route.model.requested.ref == "@app/my-app/chat"

    @pytest.mark.asyncio
    async def test_plain_model_ref_kind_in_route_metadata(self) -> None:
        """Plain model name should produce kind='model' in route metadata."""
        mock_rt = MockRuntime("model response")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=None,
        ):
            responses = OctomilResponses(planner_enabled=False)
            response = await responses.create(ResponseRequest(model="gemma-2b", input=[text_input("Hi")]))

        assert response.route is not None
        assert response.route.model.requested.kind == "model"
        assert response.route.model.requested.ref == "gemma-2b"

    @pytest.mark.asyncio
    async def test_capability_ref_kind_in_route_metadata(self) -> None:
        """@capability/xxx model ref should produce kind='capability' in route metadata."""
        mock_rt = MockRuntime("cap response")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=None,
        ):
            responses = OctomilResponses(planner_enabled=False)
            response = await responses.create(ResponseRequest(model="@capability/embeddings", input=[text_input("Hi")]))

        assert response.route is not None
        assert response.route.model.requested.kind == "capability"

    @pytest.mark.asyncio
    async def test_route_event_never_contains_prompt_or_output(self) -> None:
        """Route metadata must not leak prompt or output content."""
        mock_rt = MockRuntime("SECRET_OUTPUT")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        with patch(
            "octomil.responses.responses._try_resolve_planner_selection",
            return_value=None,
        ):
            responses = OctomilResponses(planner_enabled=False)
            response = await responses.create(ResponseRequest(model="test-model", input=[text_input("SECRET_PROMPT")]))

        route_str = str(response.route)
        assert "SECRET_PROMPT" not in route_str
        assert "SECRET_OUTPUT" not in route_str


class TestAttemptRunnerIsInvokedNotBypassed:
    """Ensures that if CandidateAttemptRunner exists, public paths do NOT bypass it."""

    @pytest.mark.asyncio
    async def test_create_invokes_runner_when_plan_exists(self) -> None:
        """Fails if OctomilResponses.create() bypasses CandidateAttemptRunner when a plan is available."""
        mock_rt = MockRuntime("runner output")
        ModelRuntimeRegistry.shared().default_factory = lambda model_id: mock_rt

        selection = _make_selection(locality="local")

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
