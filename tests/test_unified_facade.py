"""Tests for the unified Octomil facade."""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import octomil
from octomil.embeddings import EmbeddingResult, EmbeddingUsage
from octomil.errors import OctomilError
from octomil.facade import FacadeEmbeddings, FacadeResponses, Octomil, OctomilNotInitializedError
from octomil.responses.types import Response, ResponseToolCall, TextOutput, ToolCallOutput


class TestTopLevelFacadeExport:
    def test_top_level_octomil_exports_unified_facade(self):
        from octomil import Octomil as TopLevelOctomil

        assert TopLevelOctomil is Octomil
        assert hasattr(TopLevelOctomil, "from_env")
        assert hasattr(TopLevelOctomil, "hosted_from_env")
        assert inspect.getsourcefile(TopLevelOctomil) == inspect.getsourcefile(Octomil)
        assert octomil.__version__ == importlib.metadata.version("octomil")


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructorPublishableKey:
    def test_valid_test_key(self):
        client = Octomil(publishable_key="oct_pub_test_abc123")
        assert client._initialized is False

    def test_valid_live_key(self):
        client = Octomil(publishable_key="oct_pub_live_xyz789")
        assert client._initialized is False

    def test_invalid_prefix_raises(self):
        with pytest.raises(OctomilError):
            Octomil(publishable_key="bad_key_prefix")

    def test_empty_key_raises(self):
        with pytest.raises(OctomilError):
            Octomil(publishable_key="")


class TestConstructorApiKey:
    def test_api_key_with_org_id(self):
        client = Octomil(api_key="edg_test_123", org_id="org_abc")
        assert client._initialized is False

    def test_api_key_without_org_id_raises(self):
        with pytest.raises(ValueError, match="org_id is required"):
            Octomil(api_key="edg_test_123")


class TestConstructorAuth:
    def test_auth_passthrough(self):
        from octomil.auth import OrgApiKeyAuth

        auth = OrgApiKeyAuth(api_key="edg_test", org_id="org_1")
        client = Octomil(auth=auth)
        assert client._auth is auth


class TestConstructorNoArgs:
    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="One of"):
            Octomil()


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_sets_initialized(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        assert client._initialized is False
        asyncio.run(client.initialize())
        assert client._initialized is True

    def test_idempotent(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        asyncio.run(client.initialize())
        # Second call should not raise
        asyncio.run(client.initialize())
        assert client._initialized is True


# ---------------------------------------------------------------------------
# responses property — guards
# ---------------------------------------------------------------------------


class TestResponsesGuard:
    def test_create_before_init_raises(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        with pytest.raises(OctomilNotInitializedError):
            _ = client.responses

    def test_stream_before_init_raises(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        with pytest.raises(OctomilNotInitializedError):
            _ = client.responses


# ---------------------------------------------------------------------------
# Audio facade — PR 2 of unified-tts-speech-routing-implementation-plan.md
# ---------------------------------------------------------------------------


class TestAudioFacadeGuard:
    def test_audio_before_init_raises(self):
        """client.audio must throw OctomilNotInitializedError before
        await client.initialize(). Symmetric with responses/embeddings."""
        client = Octomil(publishable_key="oct_pub_test_abc")
        with pytest.raises(OctomilNotInitializedError):
            _ = client.audio


class TestAudioFacadeShape:
    def test_audio_namespace_after_initialize(self):
        from octomil.audio import FacadeAudio, FacadeSpeech

        client = Octomil(publishable_key="oct_pub_test_abc")
        with patch("octomil.client.OctomilClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            asyncio.run(client.initialize())

        assert isinstance(client.audio, FacadeAudio)
        assert isinstance(client.audio.speech, FacadeSpeech)
        assert callable(client.audio.speech.create)


class TestSynthesizeSpeechRouting:
    """ExecutionKernel.synthesize_speech routing matrix.

    Mocks the kernel's helpers so the test does not require sherpa-onnx,
    a staged model, or a live cloud profile.
    """

    @pytest.mark.asyncio
    async def test_local_only_without_sherpa_raises_local_tts_runtime_unavailable(
        self,
    ):
        from octomil.config.local import (
            ResolvedExecutionDefaults,
        )
        from octomil.errors import OctomilError, OctomilErrorCode
        from octomil.execution.kernel import ExecutionKernel

        kernel = ExecutionKernel.__new__(ExecutionKernel)
        kernel._config_set = MagicMock()

        # Resolve gives local_only with no cloud profile.
        defaults = ResolvedExecutionDefaults(
            model="kokoro-82m",
            policy_preset="local_only",
            inline_policy=None,
            cloud_profile=None,
        )
        with (
            patch.object(kernel, "_resolve", return_value=defaults),
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
            patch("octomil.execution.kernel._resolve_routing_policy") as routing,
            patch.object(kernel, "_has_local_tts_backend", return_value=False),
            patch(
                "octomil.execution.kernel._select_locality_for_capability",
                return_value=("on_device", False),
            ),
        ):
            routing.return_value = MagicMock()
            with pytest.raises(OctomilError) as ei:
                await kernel.synthesize_speech(
                    model="@app/tts-tester/tts",
                    input="hello",
                )
            assert ei.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
            assert "local_tts_runtime_unavailable" in str(ei.value)

    @pytest.mark.asyncio
    async def test_local_only_returns_wav_and_does_not_call_cloud(self):
        from octomil.config.local import ResolvedExecutionDefaults
        from octomil.execution.kernel import ExecutionKernel

        kernel = ExecutionKernel.__new__(ExecutionKernel)
        kernel._config_set = MagicMock()

        fake_backend = MagicMock()
        fake_backend.synthesize.return_value = {
            "audio_bytes": b"\x52\x49\x46\x46\x00",  # WAV magic
            "content_type": "audio/wav",
            "format": "wav",
            "sample_rate": 24000,
            "duration_ms": 750,
            "voice": "af_bella",
            "model": "kokoro-82m",
        }

        defaults = ResolvedExecutionDefaults(
            model="kokoro-82m",
            policy_preset="local_only",
            inline_policy=None,
            cloud_profile=None,
        )
        cloud_spy = MagicMock()
        with (
            patch.object(kernel, "_resolve", return_value=defaults),
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
            patch("octomil.execution.kernel._resolve_routing_policy"),
            patch.object(kernel, "_has_local_tts_backend", return_value=True),
            patch(
                "octomil.execution.kernel._select_locality_for_capability",
                return_value=("on_device", False),
            ),
            patch.object(kernel, "_resolve_local_tts_backend", return_value=fake_backend),
            patch.object(kernel, "_cloud_synthesize_speech", new=cloud_spy),
        ):
            response = await kernel.synthesize_speech(
                model="@app/tts-tester/tts",
                input="hello",
                voice="af_bella",
            )

        # WAV bytes returned, locality is on_device, cloud branch never invoked.
        assert response.audio_bytes.startswith(b"\x52\x49\x46\x46")
        assert response.format == "wav"
        assert response.route.locality == "on_device"
        assert response.provider is None  # contract: local provider=None
        # billed_units / unit_kind null on local — no cloud usage written.
        assert response.billed_units is None
        assert response.unit_kind is None
        cloud_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_local_voice_mismatch_raises_voice_not_supported(self):
        from octomil.config.local import ResolvedExecutionDefaults
        from octomil.errors import OctomilError, OctomilErrorCode
        from octomil.execution.kernel import ExecutionKernel

        kernel = ExecutionKernel.__new__(ExecutionKernel)
        kernel._config_set = MagicMock()

        defaults = ResolvedExecutionDefaults(
            model="kokoro-82m",
            policy_preset="local_only",
            inline_policy=None,
            cloud_profile=None,
        )
        with (
            patch.object(kernel, "_resolve", return_value=defaults),
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
            patch("octomil.execution.kernel._resolve_routing_policy"),
            patch.object(kernel, "_has_local_tts_backend", return_value=True),
            patch(
                "octomil.execution.kernel._select_locality_for_capability",
                return_value=("on_device", False),
            ),
        ):
            with pytest.raises(OctomilError) as ei:
                await kernel.synthesize_speech(
                    model="kokoro-82m",
                    input="hello",
                    voice="alloy",  # OpenAI voice on a local Kokoro model
                )
            assert ei.value.code == OctomilErrorCode.INVALID_INPUT
            assert "voice_not_supported_for_locality" in str(ei.value)

    @pytest.mark.asyncio
    async def test_app_ref_uses_planner_model_for_local_availability(self):
        from octomil.config.local import ResolvedExecutionDefaults
        from octomil.execution.kernel import ExecutionKernel

        kernel = ExecutionKernel.__new__(ExecutionKernel)
        kernel._config_set = MagicMock()

        fake_backend = MagicMock()
        fake_backend.synthesize.return_value = {
            "audio_bytes": b"RIFF\x00",
            "content_type": "audio/wav",
            "format": "wav",
            "sample_rate": 24000,
            "duration_ms": 500,
        }
        defaults = ResolvedExecutionDefaults(
            model="@app/tts-tester/tts",
            policy_preset="local_first",
            inline_policy=None,
            cloud_profile=None,
        )
        selection = SimpleNamespace(
            app_resolution=SimpleNamespace(
                selected_model="kokoro-82m",
                routing_policy="private",
            ),
            resolution=None,
        )
        cloud_spy = AsyncMock()

        with (
            patch.object(kernel, "_resolve", return_value=defaults),
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection),
            patch.object(kernel, "_has_local_tts_backend", return_value=True) as has_local,
            patch.object(kernel, "_resolve_local_tts_backend", return_value=fake_backend),
            patch.object(kernel, "_cloud_synthesize_speech", new=cloud_spy),
        ):
            response = await kernel.synthesize_speech(
                model="@app/tts-tester/tts",
                input="hello",
            )

        has_local.assert_called_once_with("kokoro-82m")
        assert response.model == "kokoro-82m"
        assert response.route.locality == "on_device"
        cloud_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_app_ref_cloud_dispatch_keeps_app_ref_for_policy(self):
        from octomil.config.local import CloudProfile, ResolvedExecutionDefaults
        from octomil.execution.kernel import ExecutionKernel

        kernel = ExecutionKernel.__new__(ExecutionKernel)
        kernel._config_set = MagicMock()

        cloud_profile = CloudProfile(api_key_env="OCTOMIL_SERVER_KEY")
        defaults = ResolvedExecutionDefaults(
            model="@app/tts-tester/tts",
            policy_preset="cloud_first",
            inline_policy=None,
            cloud_profile=cloud_profile,
        )
        selection = SimpleNamespace(
            app_resolution=SimpleNamespace(
                selected_model="tts-1",
                routing_policy="cloud_first",
            ),
            resolution=None,
        )
        cloud_synthesize = AsyncMock(
            return_value={
                "audio_bytes": b"mp3",
                "content_type": "audio/mpeg",
                "format": "mp3",
                "model": "tts-1",
                "provider": "openai",
            }
        )

        with (
            patch.object(kernel, "_resolve", return_value=defaults),
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection),
            patch("octomil.execution.kernel._cloud_available", return_value=True),
            patch.object(kernel, "_has_local_tts_backend", return_value=False),
            patch.object(kernel, "_cloud_synthesize_speech", new=cloud_synthesize),
        ):
            response = await kernel.synthesize_speech(
                model="@app/tts-tester/tts",
                input="hello",
                voice="alloy",
                response_format="mp3",
            )

        assert cloud_synthesize.await_args.args[0] == "@app/tts-tester/tts"
        assert response.model == "tts-1"
        assert response.provider == "openai"
        assert response.route.locality == "cloud"

    @pytest.mark.asyncio
    async def test_local_policy_runtime_unavailable_is_octomil_error(self):
        from octomil.config.local import ResolvedExecutionDefaults
        from octomil.errors import OctomilError, OctomilErrorCode
        from octomil.execution.kernel import ExecutionKernel

        kernel = ExecutionKernel.__new__(ExecutionKernel)
        kernel._config_set = MagicMock()

        defaults = ResolvedExecutionDefaults(
            model="kokoro-82m",
            policy_preset="local_only",
            inline_policy=None,
            cloud_profile=None,
        )
        with (
            patch.object(kernel, "_resolve", return_value=defaults),
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=None),
            patch.object(kernel, "_has_local_tts_backend", return_value=False),
        ):
            with pytest.raises(OctomilError) as ei:
                await kernel.synthesize_speech(model="kokoro-82m", input="hello")

        assert ei.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
        assert "local_tts_runtime_unavailable" in str(ei.value)

    @pytest.mark.asyncio
    async def test_empty_input_raises(self):
        from octomil.errors import OctomilError, OctomilErrorCode
        from octomil.execution.kernel import ExecutionKernel

        kernel = ExecutionKernel.__new__(ExecutionKernel)
        kernel._config_set = MagicMock()

        with pytest.raises(OctomilError) as ei:
            await kernel.synthesize_speech(model="kokoro-82m", input="   ")
        assert ei.value.code == OctomilErrorCode.INVALID_INPUT


# ---------------------------------------------------------------------------
# FacadeResponses.create / stream
# ---------------------------------------------------------------------------


class TestFacadeResponsesCreate:
    def test_delegates_to_underlying(self):
        fake_response = Response(
            id="resp_1",
            model="phi-4-mini",
            output=[TextOutput(text="Hello world")],
            finish_reason="stop",
        )
        mock_responses = MagicMock()
        mock_responses.create = AsyncMock(return_value=fake_response)

        facade = FacadeResponses(mock_responses)
        result = asyncio.run(facade.create(model="phi-4-mini", input="hi"))

        assert result is fake_response
        mock_responses.create.assert_called_once()
        request = mock_responses.create.call_args[0][0]
        assert request.model == "phi-4-mini"


class TestFacadeResponsesStream:
    def test_delegates_to_underlying(self):
        from octomil.responses.types import DoneEvent, TextDeltaEvent

        fake_done = DoneEvent(
            response=Response(
                id="resp_s1",
                model="phi-4-mini",
                output=[TextOutput(text="AB")],
                finish_reason="stop",
            )
        )

        async def _fake_stream(request):
            yield TextDeltaEvent(delta="A")
            yield TextDeltaEvent(delta="B")
            yield fake_done

        mock_responses = MagicMock()
        mock_responses.stream = _fake_stream

        facade = FacadeResponses(mock_responses)

        async def _run():
            events = []
            async for event in facade.stream(model="phi-4-mini", input="hi"):
                events.append(event)
            return events

        events = asyncio.run(_run())
        assert len(events) == 3
        assert isinstance(events[0], TextDeltaEvent)
        assert isinstance(events[2], DoneEvent)


# ---------------------------------------------------------------------------
# Response.output_text
# ---------------------------------------------------------------------------


class TestOutputText:
    def test_concatenates_text_items(self):
        resp = Response(
            id="r1",
            model="m",
            output=[TextOutput(text="Hello "), TextOutput(text="world")],
            finish_reason="stop",
        )
        assert resp.output_text == "Hello world"

    def test_empty_when_no_text_items(self):
        resp = Response(
            id="r2",
            model="m",
            output=[ToolCallOutput(tool_call=ResponseToolCall(id="tc1", name="fn", arguments="{}"))],
            finish_reason="tool_calls",
        )
        assert resp.output_text == ""

    def test_empty_output_list(self):
        resp = Response(
            id="r3",
            model="m",
            output=[],
            finish_reason="stop",
        )
        assert resp.output_text == ""

    def test_mixed_output_types(self):
        resp = Response(
            id="r4",
            model="m",
            output=[
                TextOutput(text="part1"),
                ToolCallOutput(tool_call=ResponseToolCall(id="tc1", name="fn", arguments="{}")),
                TextOutput(text="part2"),
            ],
            finish_reason="stop",
        )
        assert resp.output_text == "part1part2"


# ---------------------------------------------------------------------------
# Embeddings namespace
# ---------------------------------------------------------------------------


class TestEmbeddingsNamespace:
    def test_embeddings_namespace_exists(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        asyncio.run(client.initialize())
        assert isinstance(client.embeddings, FacadeEmbeddings)

    def test_embeddings_before_init_raises(self):
        client = Octomil(publishable_key="oct_pub_test_abc")
        with pytest.raises(OctomilNotInitializedError):
            _ = client.embeddings


class TestEmbeddingsCreate:
    def test_embeddings_create_delegates(self):
        fake_result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="nomic-embed-text-v1.5",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
        )
        client = Octomil(publishable_key="oct_pub_test_abc")
        asyncio.run(client.initialize())

        with patch.object(client._client, "embed", return_value=fake_result) as mock_embed:
            result = asyncio.run(
                client.embeddings.create(
                    model="nomic-embed-text-v1.5",
                    input="On-device AI inference at scale",
                )
            )

        assert result is fake_result
        mock_embed.assert_called_once_with(
            "nomic-embed-text-v1.5",
            "On-device AI inference at scale",
            timeout=30.0,
        )

    def test_embeddings_create_batch_input(self):
        fake_result = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="nomic-embed-text-v1.5",
            usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10),
        )
        client = Octomil(publishable_key="oct_pub_test_abc")
        asyncio.run(client.initialize())

        with patch.object(client._client, "embed", return_value=fake_result) as mock_embed:
            result = asyncio.run(
                client.embeddings.create(
                    model="nomic-embed-text-v1.5",
                    input=["hello", "world"],
                )
            )

        assert result is fake_result
        assert len(result.embeddings) == 2
        mock_embed.assert_called_once_with(
            "nomic-embed-text-v1.5",
            ["hello", "world"],
            timeout=30.0,
        )
