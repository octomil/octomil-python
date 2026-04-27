"""Tests for SDK facade wiring — chat, capabilities, telemetry, model format/warmup, device_id."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

from octomil.auth import OrgApiKeyAuth
from octomil.capabilities_client import CapabilitiesClient, CapabilityProfile, _classify_device
from octomil.chat_client import ChatChunk, ChatCompletion
from octomil.model import Model, ModelMetadata
from octomil.serve import GenerationRequest, InferenceMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(**overrides: Any):
    """Create a mock-patched OctomilClient."""
    with (
        patch("octomil.python.octomil.control_plane.RolloutsAPI"),
        patch("octomil.python.octomil.registry.ModelRegistry"),
        patch("octomil.python.octomil.api_client._ApiClient"),
    ):
        from octomil.client import OctomilClient

        auth = overrides.pop("auth", None) or OrgApiKeyAuth(
            api_key=overrides.pop("api_key", "test-key"),
            org_id=overrides.pop("org_id", "default"),
            api_base=overrides.pop("api_base", "https://api.octomil.com/api/v1"),
        )
        return OctomilClient(auth=auth, **overrides)


def _make_engine(backend: MagicMock | None = None) -> MagicMock:
    engine = MagicMock()
    engine.name = "echo"
    engine.manages_own_download = False
    engine.create_backend.return_value = backend or MagicMock()
    return engine


def _make_metadata(**overrides: Any) -> ModelMetadata:
    defaults = {
        "model_id": "model-abc",
        "name": "test-model",
        "version": "1.0.0",
    }
    defaults.update(overrides)
    return ModelMetadata(**defaults)


# ===========================================================================
# Unified facade environment config
# ===========================================================================


class TestOctomilFacadeFromEnv:
    def test_uses_server_key_and_org_id(self, monkeypatch):
        from octomil import Octomil

        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "srv_key")
        monkeypatch.setenv("OCTOMIL_API_KEY", "legacy_key")
        monkeypatch.setenv("OCTOMIL_ORG_ID", "org_public_id")

        client = Octomil.from_env()

        assert client._auth.api_key == "srv_key"
        assert client._auth.org_id == "org_public_id"

    def test_accepts_legacy_api_key_when_server_key_absent(self, monkeypatch):
        from octomil import Octomil

        monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
        monkeypatch.setenv("OCTOMIL_API_KEY", "legacy_key")
        monkeypatch.setenv("OCTOMIL_ORG_ID", "org_public_id")

        client = Octomil.from_env()

        assert client._auth.api_key == "legacy_key"
        assert client._auth.org_id == "org_public_id"

    def test_reads_api_base_override(self, monkeypatch):
        from octomil import Octomil

        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "srv_key")
        monkeypatch.setenv("OCTOMIL_ORG_ID", "org_public_id")
        monkeypatch.setenv("OCTOMIL_API_BASE", "http://localhost:8000/api/v1")

        client = Octomil.from_env()

        assert client._auth.api_base == "http://localhost:8000/api/v1"

    def test_requires_server_key_or_legacy_api_key(self, monkeypatch):
        from octomil import Octomil

        monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
        monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
        monkeypatch.setenv("OCTOMIL_ORG_ID", "org_public_id")

        try:
            Octomil.from_env()
        except ValueError as exc:
            assert "OCTOMIL_SERVER_KEY" in str(exc)
            assert "OCTOMIL_API_KEY" in str(exc)
        else:
            raise AssertionError("Expected Octomil.from_env() to require a server key")

    def test_requires_org_id(self, monkeypatch):
        from octomil import Octomil

        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "srv_key")
        monkeypatch.delenv("OCTOMIL_ORG_ID", raising=False)

        try:
            Octomil.from_env()
        except ValueError as exc:
            assert "OCTOMIL_ORG_ID" in str(exc)
        else:
            raise AssertionError("Expected Octomil.from_env() to require an org id")


class TestOctomilFacadeHostedFromEnv:
    def test_constructs_explicit_hosted_client(self, monkeypatch):
        from octomil import Octomil

        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "srv_key")
        monkeypatch.setenv("OCTOMIL_ORG_ID", "org_public_id")
        monkeypatch.setenv("OCTOMIL_API_BASE", "https://api.octomil.com/api/v1")

        client = Octomil.hosted_from_env()

        assert client._auth.api_key == "srv_key"
        assert client._auth.org_id == "org_public_id"
        assert client._auth.api_base == "https://api.octomil.com/api/v1"
        assert client._force_hosted is True
        assert client.planner_enabled is False

    def test_rejects_planner_routing_true(self, monkeypatch):
        from octomil import Octomil

        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "srv_key")
        monkeypatch.setenv("OCTOMIL_ORG_ID", "org_public_id")

        try:
            Octomil.hosted_from_env(planner_routing=True)
        except ValueError as exc:
            assert "cloud-only" in str(exc)
        else:
            raise AssertionError("Expected hosted_from_env() to reject planner_routing=True")

    def test_force_hosted_requires_server_auth(self):
        from octomil import Octomil

        try:
            Octomil(publishable_key="oct_pub_test_abc", _force_hosted=True)
        except ValueError as exc:
            assert "server-side API key" in str(exc)
        else:
            raise AssertionError("Expected _force_hosted to require server-side auth")

    def test_responses_dispatch_through_cloud_runtime(self, monkeypatch):
        from octomil import Octomil
        from octomil.runtime.core.cloud_runtime import CloudModelRuntime
        from octomil.runtime.core.types import RuntimeResponse

        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "srv_key")
        monkeypatch.setenv("OCTOMIL_ORG_ID", "org_public_id")
        monkeypatch.setenv("OCTOMIL_API_BASE", "https://api.octomil.com/api/v1")

        seen: dict[str, str | None] = {}

        async def _fake_run(self, request):
            seen["model"] = request.model
            seen["base_url"] = self._client._base_url
            return RuntimeResponse(text="hosted ok", finish_reason="stop")

        monkeypatch.setattr(CloudModelRuntime, "run", _fake_run)

        async def _run():
            client = Octomil.hosted_from_env()
            await client.initialize()
            return await client.responses.create(model="gpt-test", input="hello")

        response = asyncio.run(_run())

        assert response.output_text == "hosted ok"
        assert response.locality == "cloud"
        assert response.route is not None
        assert response.route.execution is not None
        assert response.route.execution.locality == "cloud"
        assert response.route.execution.mode == "hosted_gateway"
        assert seen == {"model": "gpt-test", "base_url": "https://api.octomil.com/v1"}

    def test_stream_dispatches_through_cloud_runtime(self, monkeypatch):
        from octomil import Octomil
        from octomil.responses.types import DoneEvent, TextDeltaEvent
        from octomil.runtime.core.cloud_runtime import CloudModelRuntime
        from octomil.runtime.core.types import RuntimeChunk

        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "srv_key")
        monkeypatch.setenv("OCTOMIL_ORG_ID", "org_public_id")
        monkeypatch.setenv("OCTOMIL_API_BASE", "https://api.octomil.com/api/v1")

        seen: dict[str, str | None] = {}

        async def _fake_stream(self, request):
            seen["model"] = request.model
            seen["base_url"] = self._client._base_url
            yield RuntimeChunk(text="hosted ")
            yield RuntimeChunk(text="stream")

        monkeypatch.setattr(CloudModelRuntime, "stream", _fake_stream)

        async def _run():
            client = Octomil.hosted_from_env()
            await client.initialize()
            events = []
            async for event in client.responses.stream(model="gpt-stream", input="hello"):
                events.append(event)
            return events

        events = asyncio.run(_run())

        assert isinstance(events[0], TextDeltaEvent)
        assert events[0].delta == "hosted "
        assert isinstance(events[1], TextDeltaEvent)
        assert events[1].delta == "stream"
        assert isinstance(events[2], DoneEvent)
        assert events[2].response.output_text == "hosted stream"
        assert events[2].response.locality == "cloud"
        assert events[2].response.route is not None
        assert events[2].response.route.execution is not None
        assert events[2].response.route.execution.locality == "cloud"
        assert events[2].response.route.execution.mode == "hosted_gateway"
        assert seen == {"model": "gpt-stream", "base_url": "https://api.octomil.com/v1"}


# ===========================================================================
# P0 #1 — ChatClient
# ===========================================================================


class TestChatCompletion:
    def test_fields(self):
        c = ChatCompletion(
            message={"role": "assistant", "content": "hi"},
            latency_ms=42.0,
        )
        assert c.message["content"] == "hi"
        assert c.latency_ms == 42.0
        assert c.usage == {}

    def test_usage_field(self):
        c = ChatCompletion(
            message={"role": "assistant", "content": ""},
            latency_ms=0.0,
            usage={"prompt_tokens": 10},
        )
        assert c.usage["prompt_tokens"] == 10


class TestChatChunk:
    def test_fields(self):
        chunk = ChatChunk(index=0, content="hello", done=False)
        assert chunk.index == 0
        assert chunk.content == "hello"
        assert chunk.done is False
        assert chunk.role == "assistant"


class TestChatClientCreate:
    def test_returns_chat_completion(self):
        from octomil.responses.types import Response, TextOutput

        fake_response = Response(
            id="resp-1",
            model="phi-4-mini",
            output=[TextOutput(text="Hi there")],
            finish_reason="stop",
        )

        async def _mock_create(request):
            return fake_response

        client = _make_client()
        mock_responses = MagicMock()
        mock_responses.create = _mock_create
        client._responses = mock_responses
        result = client.chat.create(
            model="phi-4-mini",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert isinstance(result, ChatCompletion)
        assert result.message["role"] == "assistant"
        assert result.message["content"] == "Hi there"
        assert result.latency_ms > 0

    def test_passes_params(self):
        from octomil.responses.types import Response, TextOutput

        fake_response = Response(
            id="resp-2",
            model="m",
            output=[TextOutput(text="ok")],
            finish_reason="stop",
        )

        async def _mock_create(request):
            return fake_response

        client = _make_client()
        mock_responses = MagicMock()
        mock_responses.create = _mock_create
        client._responses = mock_responses
        result = client.chat.create(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            temperature=0.3,
            max_tokens=100,
            top_p=0.9,
        )
        assert isinstance(result, ChatCompletion)


class TestChatClientStream:
    def test_yields_chat_chunks(self):
        from octomil.responses.types import DoneEvent, Response, TextDeltaEvent, TextOutput

        async def _run():
            async def _fake_stream(request):
                yield TextDeltaEvent(delta="A")
                yield TextDeltaEvent(delta="B")
                yield DoneEvent(
                    response=Response(
                        id="resp-s1",
                        model="phi-4-mini",
                        output=[TextOutput(text="AB")],
                        finish_reason="stop",
                    )
                )

            client = _make_client()
            mock_responses = MagicMock()
            mock_responses.stream = _fake_stream
            client._responses = mock_responses
            chunks = []
            async for chunk in client.chat.stream(
                model="phi-4-mini",
                messages=[{"role": "user", "content": "hi"}],
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == 3
        assert all(isinstance(c, ChatChunk) for c in chunks)
        assert chunks[0].content == "A"
        assert chunks[2].done is True


class TestChatClientPropertyCached:
    def test_same_instance(self):
        client = _make_client()
        assert client.chat is client.chat


# ===========================================================================
# P0 #2 — CapabilitiesClient
# ===========================================================================


class TestDeviceClassification:
    def test_flagship(self):
        assert _classify_device(32_000, True) == "flagship"

    def test_high(self):
        assert _classify_device(8_000, True) == "high"

    def test_mid_with_gpu(self):
        assert _classify_device(4_000, True) == "mid"

    def test_mid_without_gpu(self):
        assert _classify_device(8_000, False) == "mid"

    def test_low(self):
        assert _classify_device(2_000, False) == "low"

    def test_none_memory(self):
        assert _classify_device(None, True) == "mid"


class TestCapabilitiesClientCurrent:
    def test_returns_profile(self):
        client = _make_client()
        with (
            patch("octomil.device_info.get_memory_info", return_value=16_000),
            patch("octomil.device_info.get_storage_info", return_value=100_000),
            patch("octomil.capabilities_client._detect_accelerators", return_value=["metal"]),
        ):
            profile = client.capabilities.current()

        assert isinstance(profile, CapabilityProfile)
        assert profile.memory_mb == 16_000
        assert profile.storage_mb == 100_000
        assert "metal" in profile.accelerators
        assert profile.device_class == "flagship"
        assert isinstance(profile.available_runtimes, list)

    def test_no_gpu(self):
        with (
            patch("octomil.device_info.get_memory_info", return_value=4_000),
            patch("octomil.device_info.get_storage_info", return_value=50_000),
            patch("octomil.capabilities_client._detect_accelerators", return_value=[]),
        ):
            cap = CapabilitiesClient()
            profile = cap.current()

        assert profile.device_class == "mid"
        assert profile.accelerators == []


class TestCapabilitiesPropertyCached:
    def test_same_instance(self):
        client = _make_client()
        assert client.capabilities is client.capabilities


# ===========================================================================
# P0 #3 — TelemetryClient
# ===========================================================================


class TestTelemetryClientTrack:
    def test_track_enqueues_event(self):
        client = _make_client()
        # The reporter is created when api_key is present
        mock_reporter = MagicMock()
        client._reporter = mock_reporter

        client.telemetry.track("user.login", {"platform": "ios"})

        mock_reporter._enqueue.assert_called_once_with(
            name="user.login",
            attributes={"platform": "ios"},
        )

    def test_track_without_reporter_is_noop(self):
        client = _make_client()
        client._reporter = None
        # Should not raise
        client.telemetry.track("event", {"key": "val"})

    def test_track_with_none_attributes(self):
        client = _make_client()
        mock_reporter = MagicMock()
        client._reporter = mock_reporter

        client.telemetry.track("event")

        mock_reporter._enqueue.assert_called_once_with(name="event", attributes={})


class TestTelemetryClientFlush:
    def test_flush_calls_close_and_recreates(self):
        client = _make_client()
        mock_reporter = MagicMock()
        client._reporter = mock_reporter

        with patch("octomil.telemetry.TelemetryReporter") as mock_tr_cls:
            new_reporter = MagicMock()
            mock_tr_cls.return_value = new_reporter
            client.telemetry.flush()

        mock_reporter.close.assert_called_once()
        assert client._reporter is new_reporter

    def test_flush_without_reporter_is_noop(self):
        client = _make_client()
        client._reporter = None
        # Should not raise
        client.telemetry.flush()


class TestTelemetryPropertyCached:
    def test_same_instance(self):
        client = _make_client()
        assert client.telemetry is client.telemetry


# ===========================================================================
# P1 #4 — Model.format
# ===========================================================================


class TestModelFormat:
    def test_format_from_engine(self):
        engine = _make_engine()
        engine.name = "llama.cpp"
        model = Model(metadata=_make_metadata(), engine=engine)
        assert model.format == "gguf"

    def test_format_from_mlx(self):
        engine = _make_engine()
        engine.name = "mlx-lm"
        model = Model(metadata=_make_metadata(), engine=engine)
        assert model.format == "safetensors"

    def test_format_from_ort(self):
        engine = _make_engine()
        engine.name = "ort"
        model = Model(metadata=_make_metadata(), engine=engine)
        assert model.format == "onnx"

    def test_format_from_metadata_overrides_engine(self):
        engine = _make_engine()
        engine.name = "llama.cpp"
        model = Model(
            metadata=_make_metadata(format="custom-format"),
            engine=engine,
        )
        assert model.format == "custom-format"

    def test_format_unknown_engine(self):
        engine = _make_engine()
        engine.name = "some-future-engine"
        model = Model(metadata=_make_metadata(), engine=engine)
        assert model.format == "unknown"


# ===========================================================================
# P1 #5 — Model.warmup()
# ===========================================================================


class TestModelWarmup:
    def test_warmup_runs_minimal_inference(self):
        backend = MagicMock()
        backend.generate.return_value = ("", InferenceMetrics())
        engine = _make_engine(backend)
        model = Model(metadata=_make_metadata(), engine=engine)

        model.warmup()

        backend.generate.assert_called_once()
        req = backend.generate.call_args[0][0]
        assert isinstance(req, GenerationRequest)
        assert req.max_tokens == 1

    def test_warmup_is_idempotent(self):
        backend = MagicMock()
        backend.generate.return_value = ("", InferenceMetrics())
        engine = _make_engine(backend)
        model = Model(metadata=_make_metadata(), engine=engine)

        model.warmup()
        model.warmup()

        assert backend.generate.call_count == 1

    def test_warmup_failure_is_non_fatal(self):
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("engine not ready")
        engine = _make_engine(backend)
        model = Model(metadata=_make_metadata(), engine=engine)

        # Should not raise
        model.warmup()
        assert model._warmed_up is True


# ===========================================================================
# P3 #6 — device_id in constructor
# ===========================================================================


class TestDeviceId:
    def test_explicit_device_id(self):
        client = _make_client(device_id="my-device-123")
        assert client.device_id == "my-device-123"

    def test_device_id_defaults_to_none(self):
        client = _make_client()
        # _device_id is None, property will lazy-derive
        assert client._device_id is None

    def test_device_id_lazy_derivation(self):
        client = _make_client()
        with patch("octomil.device_info.get_stable_device_id", return_value="derived-id"):
            did = client.device_id
        assert did == "derived-id"

    def test_device_id_stored_on_client(self):
        client = _make_client(device_id="dev-42")
        assert client._device_id == "dev-42"
        assert client.device_id == "dev-42"

    def test_device_id_property_caches(self):
        client = _make_client()
        with patch("octomil.device_info.get_stable_device_id", return_value="cached-id"):
            _ = client.device_id
        # Second access should NOT call get_stable_device_id again (cached)
        assert client.device_id == "cached-id"
