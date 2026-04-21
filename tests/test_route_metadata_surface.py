"""Tests for public RouteMetadata surface on SDK response objects.

Verifies that:
- Response, EmbeddingResult, and TranscriptionResult expose a .route property
- RouteMetadata and sub-types are importable from octomil and octomil.types
- The nested structure matches the contract shape
- The field is additive and defaults to None (backward compatibility)
"""

from __future__ import annotations

import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock

from octomil.execution.kernel import (
    ArtifactCache,
    FallbackInfo,
    PlannerInfo,
    RouteArtifact,
    RouteExecution,
    RouteMetadata,
    RouteModel,
    RouteModelRequested,
    RouteModelResolved,
    RouteReason,
)
from octomil.responses.types import Response, ResponseUsage, TextOutput

# ---------------------------------------------------------------------------
# Import path tests
# ---------------------------------------------------------------------------


class TestImportPaths:
    """RouteMetadata and sub-types must be importable from octomil and octomil.types."""

    def test_import_from_octomil_top_level(self):
        import octomil

        assert hasattr(octomil, "RouteMetadata")
        assert octomil.RouteMetadata is RouteMetadata

    def test_import_from_octomil_types(self):
        from octomil.types import (
            ArtifactCache as AC,
        )
        from octomil.types import (
            FallbackInfo as FI,
        )
        from octomil.types import (
            PlannerInfo as PI,
        )
        from octomil.types import (
            RouteArtifact as RA,
        )
        from octomil.types import (
            RouteExecution as RE,
        )
        from octomil.types import (
            RouteMetadata as RM,
        )
        from octomil.types import (
            RouteModel as RMo,
        )
        from octomil.types import (
            RouteModelRequested as RMR,
        )
        from octomil.types import (
            RouteModelResolved as RMRe,
        )
        from octomil.types import (
            RouteReason as RR,
        )

        assert RM is RouteMetadata
        assert RE is RouteExecution
        assert RMo is RouteModel
        assert RMR is RouteModelRequested
        assert RMRe is RouteModelResolved
        assert RA is RouteArtifact
        assert AC is ArtifactCache
        assert PI is PlannerInfo
        assert FI is FallbackInfo
        assert RR is RouteReason

    def test_all_subtypes_in_octomil_top_level(self):
        import octomil

        for name in [
            "RouteMetadata",
            "RouteExecution",
            "RouteModel",
            "RouteModelRequested",
            "RouteModelResolved",
            "RouteArtifact",
            "ArtifactCache",
            "PlannerInfo",
            "FallbackInfo",
            "RouteReason",
        ]:
            assert hasattr(octomil, name), f"octomil.{name} missing"
            assert name in octomil.__all__, f"{name} not in octomil.__all__"


# ---------------------------------------------------------------------------
# Response.route field
# ---------------------------------------------------------------------------


def _make_route() -> RouteMetadata:
    """Build a fully-populated RouteMetadata for testing."""
    return RouteMetadata(
        status="selected",
        execution=RouteExecution(locality="local", mode="sdk_runtime", engine="mlx-lm"),
        model=RouteModel(
            requested=RouteModelRequested(ref="phi-4-mini", kind="model", capability="chat"),
            resolved=RouteModelResolved(id="model_123", slug="phi-4-mini", version_id="v1"),
        ),
        artifact=RouteArtifact(
            id="art_1",
            version="1.0",
            format="mlx",
            digest="sha256:abc",
            cache=ArtifactCache(status="hit", managed_by="octomil"),
        ),
        planner=PlannerInfo(source="server"),
        fallback=FallbackInfo(used=False),
        reason=RouteReason(code="planner", message="planner selected local runtime"),
    )


class TestResponseRouteField:
    """Response dataclass exposes .route as Optional[RouteMetadata]."""

    def test_route_defaults_to_none(self):
        resp = Response(
            id="resp_1",
            model="phi-4-mini",
            output=[TextOutput(text="hello")],
            finish_reason="stop",
        )
        assert resp.route is None

    def test_route_can_be_set(self):
        route = _make_route()
        resp = Response(
            id="resp_2",
            model="phi-4-mini",
            output=[TextOutput(text="hello")],
            finish_reason="stop",
            route=route,
        )
        assert resp.route is route
        assert resp.route.status == "selected"
        assert resp.route.execution is not None
        assert resp.route.execution.locality == "local"
        assert resp.route.execution.engine == "mlx-lm"
        assert resp.route.model.requested.ref == "phi-4-mini"
        assert resp.route.planner.source == "server"
        assert resp.route.fallback.used is False
        assert resp.route.reason.code == "planner"

    def test_route_field_exists_in_dataclass_fields(self):
        field_names = {f.name for f in dataclasses.fields(Response)}
        assert "route" in field_names

    def test_backward_compat_existing_fields_intact(self):
        """Adding .route must not remove or change existing fields."""
        resp = Response(
            id="resp_3",
            model="test",
            output=[TextOutput(text="hi")],
            finish_reason="stop",
            usage=ResponseUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            locality="on_device",
        )
        assert resp.id == "resp_3"
        assert resp.model == "test"
        assert resp.output_text == "hi"
        assert resp.finish_reason == "stop"
        assert resp.usage is not None
        assert resp.usage.total_tokens == 15
        assert resp.locality == "on_device"
        assert resp.route is None


# ---------------------------------------------------------------------------
# EmbeddingResult.route field
# ---------------------------------------------------------------------------


class TestEmbeddingResultRouteField:
    def test_route_defaults_to_none(self):
        from octomil.embeddings import EmbeddingResult, EmbeddingUsage

        result = EmbeddingResult(
            embeddings=[[0.1, 0.2]],
            model="nomic-embed",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
        )
        assert result.route is None

    def test_route_can_be_set(self):
        from octomil.embeddings import EmbeddingResult, EmbeddingUsage

        route = RouteMetadata(
            execution=RouteExecution(locality="local", mode="sdk_runtime", engine="ort"),
            planner=PlannerInfo(source="cache"),
        )
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2]],
            model="nomic-embed",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
            route=route,
        )
        assert result.route is route
        assert result.route.execution is not None
        assert result.route.execution.engine == "ort"

    def test_route_field_exists_in_dataclass_fields(self):
        from octomil.embeddings import EmbeddingResult

        field_names = {f.name for f in dataclasses.fields(EmbeddingResult)}
        assert "route" in field_names


# ---------------------------------------------------------------------------
# TranscriptionResult.route field
# ---------------------------------------------------------------------------


class TestTranscriptionResultRouteField:
    def test_route_defaults_to_none(self):
        from octomil.audio.types import TranscriptionResult

        result = TranscriptionResult(text="hello world")
        assert result.route is None

    def test_route_can_be_set(self):
        from octomil.audio.types import TranscriptionResult

        route = RouteMetadata(
            execution=RouteExecution(locality="local", mode="sdk_runtime", engine="whisper.cpp"),
            planner=PlannerInfo(source="offline"),
        )
        result = TranscriptionResult(text="hello world", route=route)
        assert result.route is route
        assert result.route.execution is not None
        assert result.route.execution.engine == "whisper.cpp"

    def test_route_field_exists_in_dataclass_fields(self):
        from octomil.audio.types import TranscriptionResult

        field_names = {f.name for f in dataclasses.fields(TranscriptionResult)}
        assert "route" in field_names


# ---------------------------------------------------------------------------
# RouteMetadata nested structure contract
# ---------------------------------------------------------------------------


class TestRouteMetadataContract:
    """RouteMetadata shape matches the octomil-contracts spec."""

    def test_all_expected_fields(self):
        expected = {"status", "execution", "model", "artifact", "planner", "fallback", "reason"}
        actual = {f.name for f in dataclasses.fields(RouteMetadata)}
        assert expected == actual

    def test_route_execution_fields(self):
        expected = {"locality", "mode", "engine"}
        actual = {f.name for f in dataclasses.fields(RouteExecution)}
        assert expected == actual

    def test_route_model_fields(self):
        expected = {"requested", "resolved"}
        actual = {f.name for f in dataclasses.fields(RouteModel)}
        assert expected == actual

    def test_route_model_requested_fields(self):
        expected = {"ref", "kind", "capability"}
        actual = {f.name for f in dataclasses.fields(RouteModelRequested)}
        assert expected == actual

    def test_route_model_resolved_fields(self):
        expected = {"id", "slug", "version_id", "variant_id"}
        actual = {f.name for f in dataclasses.fields(RouteModelResolved)}
        assert expected == actual

    def test_route_artifact_fields(self):
        expected = {"id", "version", "format", "digest", "cache"}
        actual = {f.name for f in dataclasses.fields(RouteArtifact)}
        assert expected == actual

    def test_artifact_cache_fields(self):
        expected = {"status", "managed_by"}
        actual = {f.name for f in dataclasses.fields(ArtifactCache)}
        assert expected == actual

    def test_planner_info_fields(self):
        expected = {"source"}
        actual = {f.name for f in dataclasses.fields(PlannerInfo)}
        assert expected == actual

    def test_fallback_info_fields(self):
        expected = {"used"}
        actual = {f.name for f in dataclasses.fields(FallbackInfo)}
        assert expected == actual

    def test_route_reason_fields(self):
        expected = {"code", "message"}
        actual = {f.name for f in dataclasses.fields(RouteReason)}
        assert expected == actual

    def test_defaults_produce_valid_instance(self):
        """RouteMetadata() with all defaults must be a valid, inspectable object."""
        rm = RouteMetadata()
        assert rm.status == "selected"
        assert rm.execution is None
        assert rm.model.requested.ref == ""
        assert rm.artifact is None
        assert rm.planner.source == "offline"
        assert rm.fallback.used is False
        assert rm.reason.code == ""


# ---------------------------------------------------------------------------
# Facade integration -- FacadeResponses threading
# ---------------------------------------------------------------------------


class TestFacadeResponsesRouteThreading:
    """FacadeResponses.create() must pass through .route from OctomilResponses."""

    def test_route_preserved_through_facade(self):
        from octomil.facade import FacadeResponses

        route = _make_route()
        fake_response = Response(
            id="resp_facade",
            model="phi-4-mini",
            output=[TextOutput(text="hello from facade")],
            finish_reason="stop",
            route=route,
        )
        mock_responses = MagicMock()
        mock_responses.create = AsyncMock(return_value=fake_response)

        facade = FacadeResponses(mock_responses)
        result = asyncio.run(facade.create(model="phi-4-mini", input="hi"))

        assert result.route is route
        assert result.route.execution is not None
        assert result.route.execution.locality == "local"
        assert result.route.planner.source == "server"

    def test_route_none_through_facade(self):
        from octomil.facade import FacadeResponses

        fake_response = Response(
            id="resp_no_route",
            model="phi-4-mini",
            output=[TextOutput(text="cloud passthrough")],
            finish_reason="stop",
        )
        mock_responses = MagicMock()
        mock_responses.create = AsyncMock(return_value=fake_response)

        facade = FacadeResponses(mock_responses)
        result = asyncio.run(facade.create(model="phi-4-mini", input="hi"))

        assert result.route is None


# ---------------------------------------------------------------------------
# Real OctomilResponses integration tests (not mocks)
# ---------------------------------------------------------------------------


class _StubRuntime:
    """Minimal ModelRuntime that returns a canned response."""

    def __init__(self, text: str = "Hello from stub") -> None:
        self._text = text

    @property
    def capabilities(self):
        from octomil.runtime.core.types import RuntimeCapabilities

        return RuntimeCapabilities()

    async def run(self, request):
        from octomil.runtime.core.types import RuntimeResponse, RuntimeUsage

        return RuntimeResponse(
            text=self._text,
            usage=RuntimeUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )

    async def stream(self, request):
        from octomil.runtime.core.types import RuntimeChunk, RuntimeUsage

        yield RuntimeChunk(
            text=self._text,
            usage=RuntimeUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )


class TestOctomilResponsesRoutePopulation:
    """Verify OctomilResponses.create() populates response.route with RouteMetadata."""

    def test_route_is_populated_on_create(self):
        """OctomilResponses.create() must set response.route to a RouteMetadata instance."""
        from octomil.responses import OctomilResponses
        from octomil.responses.types import ResponseRequest, text_input

        runtime = _StubRuntime()
        responses = OctomilResponses(runtime_resolver=lambda _: runtime)
        request = ResponseRequest(model="test-model", input=[text_input("Hi")])
        response = asyncio.run(responses.create(request))

        assert response.route is not None
        assert isinstance(response.route, RouteMetadata)

    def test_route_execution_is_local_sdk_runtime(self):
        """A plain runtime (not cloud) defaults to locality='local', mode='sdk_runtime'."""
        from octomil.responses import OctomilResponses
        from octomil.responses.types import ResponseRequest, text_input

        runtime = _StubRuntime()
        responses = OctomilResponses(runtime_resolver=lambda _: runtime)
        request = ResponseRequest(model="gemma3-1b", input=[text_input("Hello")])
        response = asyncio.run(responses.create(request))

        route = response.route
        assert route is not None
        assert route.execution is not None
        assert route.execution.locality == "local"
        assert route.execution.mode == "sdk_runtime"

    def test_route_model_requested_ref_matches_input(self):
        """route.model.requested.ref matches the model string passed to create()."""
        from octomil.responses import OctomilResponses
        from octomil.responses.types import ResponseRequest, text_input

        runtime = _StubRuntime()
        responses = OctomilResponses(runtime_resolver=lambda _: runtime)
        request = ResponseRequest(model="phi-4-mini", input=[text_input("Test")])
        response = asyncio.run(responses.create(request))

        route = response.route
        assert route is not None
        assert route.model.requested.ref == "phi-4-mini"
        assert route.model.requested.kind == "model"
        assert route.model.requested.capability == "chat"

    def test_route_planner_source_offline_without_planner(self):
        """Without a planner, planner.source should be 'offline'."""
        from octomil.responses import OctomilResponses
        from octomil.responses.types import ResponseRequest, text_input

        runtime = _StubRuntime()
        responses = OctomilResponses(runtime_resolver=lambda _: runtime)
        request = ResponseRequest(model="test-model", input=[text_input("Hi")])
        response = asyncio.run(responses.create(request))

        assert response.route is not None
        assert response.route.planner.source == "offline"

    def test_route_fallback_not_used(self):
        """Direct runtime (no fallback) should have fallback.used=False."""
        from octomil.responses import OctomilResponses
        from octomil.responses.types import ResponseRequest, text_input

        runtime = _StubRuntime()
        responses = OctomilResponses(runtime_resolver=lambda _: runtime)
        request = ResponseRequest(model="test-model", input=[text_input("Hi")])
        response = asyncio.run(responses.create(request))

        assert response.route is not None
        assert response.route.fallback.used is False

    def test_route_status_is_selected(self):
        """Default status should be 'selected'."""
        from octomil.responses import OctomilResponses
        from octomil.responses.types import ResponseRequest, text_input

        runtime = _StubRuntime()
        responses = OctomilResponses(runtime_resolver=lambda _: runtime)
        request = ResponseRequest(model="test-model", input=[text_input("Hi")])
        response = asyncio.run(responses.create(request))

        assert response.route is not None
        assert response.route.status == "selected"

    def test_stream_done_event_has_route(self):
        """The DoneEvent from stream() must carry RouteMetadata."""
        from octomil.responses import OctomilResponses
        from octomil.responses.types import DoneEvent, ResponseRequest, text_input

        runtime = _StubRuntime()
        responses = OctomilResponses(runtime_resolver=lambda _: runtime)
        request = ResponseRequest(model="test-model", input=[text_input("Hi")])

        async def _collect():
            events = []
            async for event in responses.stream(request):
                events.append(event)
            return events

        events = asyncio.run(_collect())
        done_events = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done_events) == 1

        route = done_events[0].response.route
        assert route is not None
        assert isinstance(route, RouteMetadata)
        assert route.execution is not None
        assert route.execution.locality == "local"
        assert route.model.requested.ref == "test-model"
