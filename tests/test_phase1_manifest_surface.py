"""Tests for Phase 1 manifest-driven runtime surface.

Covers:
- AppManifest / AppModelEntry (manifest types)
- ModelRef (discriminated union)
- ModelCatalogService (bootstrap + resolution)
- ModelReadinessManager (download tracking)
- LocalFileModelRuntime (file-backed runtime)
- OctomilResponses 3-step resolution
- OctomilAudio / AudioTranscriptions
- OctomilText / OctomilPredictor
- OctomilClient wiring (audio, text, catalog)
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from octomil._generated.delivery_mode import DeliveryMode
from octomil._generated.modality import Modality
from octomil._generated.model_capability import ModelCapability
from octomil.manifest.catalog_service import ModelCatalogService
from octomil.manifest.readiness_manager import (
    DownloadStatus,
    DownloadUpdate,
    ModelReadinessManager,
)
from octomil.manifest.types import AppManifest, AppModelEntry
from octomil.model_ref import (
    ModelRefFactory,
    _ModelRefCapability,
    _ModelRefId,
    get_capability,
    get_model_id,
    is_capability_ref,
    is_id_ref,
    model_ref_capability,
    model_ref_id,
)
from octomil.runtime.core import (
    ModelRuntime,
    ModelRuntimeRegistry,
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
    RuntimeUsage,
)

_TEXT = [Modality.TEXT]
_AUDIO = [Modality.AUDIO]

# ---------------------------------------------------------------------------
# Stub runtime for testing
# ---------------------------------------------------------------------------


class StubRuntime(ModelRuntime):
    """Minimal ModelRuntime stub that returns predictable text."""

    def __init__(self, text: str = "stub response") -> None:
        self._text = text

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_streaming=True)

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        return RuntimeResponse(
            text=self._text,
            finish_reason="stop",
            usage=RuntimeUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        yield RuntimeChunk(text=self._text)
        return


@pytest.fixture(autouse=True)
def _clean_registry():
    ModelRuntimeRegistry.shared().clear()
    yield
    ModelRuntimeRegistry.shared().clear()


# ===========================================================================
# ModelRef tests
# ===========================================================================


class TestModelRef:
    def test_id_ref(self) -> None:
        ref = model_ref_id("phi-4-mini")
        assert is_id_ref(ref)
        assert not is_capability_ref(ref)
        assert get_model_id(ref) == "phi-4-mini"
        assert get_capability(ref) is None

    def test_capability_ref(self) -> None:
        ref = model_ref_capability(ModelCapability.CHAT)
        assert is_capability_ref(ref)
        assert not is_id_ref(ref)
        assert get_capability(ref) == ModelCapability.CHAT
        assert get_model_id(ref) is None

    def test_factory_id(self) -> None:
        ref = ModelRefFactory.id("whisper-base")
        assert is_id_ref(ref)
        assert get_model_id(ref) == "whisper-base"

    def test_factory_capability(self) -> None:
        ref = ModelRefFactory.capability(ModelCapability.TRANSCRIPTION)
        assert is_capability_ref(ref)
        assert get_capability(ref) == ModelCapability.TRANSCRIPTION

    def test_equality(self) -> None:
        a = model_ref_id("phi-4-mini")
        b = model_ref_id("phi-4-mini")
        assert a == b

    def test_inequality(self) -> None:
        a = model_ref_id("phi-4-mini")
        b = model_ref_capability(ModelCapability.CHAT)
        assert a != b


# ===========================================================================
# AppManifest / AppModelEntry tests
# ===========================================================================


class TestAppManifest:
    def test_create_manifest(self) -> None:
        entry = AppModelEntry(
            id="phi-4-mini",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.MANAGED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[entry])
        assert len(manifest.models) == 1
        assert manifest.models[0].id == "phi-4-mini"

    def test_entry_for_capability(self) -> None:
        chat = AppModelEntry(
            id="phi-4",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.MANAGED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        asr = AppModelEntry(
            id="whisper-base",
            capability=ModelCapability.TRANSCRIPTION,
            delivery=DeliveryMode.BUNDLED,
            input_modalities=_AUDIO,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[chat, asr])
        assert manifest.entry_for(ModelCapability.CHAT) == chat
        assert manifest.entry_for(ModelCapability.TRANSCRIPTION) == asr
        assert manifest.entry_for(ModelCapability.EMBEDDING) is None

    def test_entry_by_id(self) -> None:
        entry = AppModelEntry(
            id="phi-4",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.CLOUD,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[entry])
        assert manifest.entry_by_id("phi-4") == entry
        assert manifest.entry_by_id("nonexistent") is None

    def test_delivery_mode_enum(self) -> None:
        assert DeliveryMode.BUNDLED.value == "bundled"
        assert DeliveryMode.MANAGED.value == "managed"
        assert DeliveryMode.CLOUD.value == "cloud"


# ===========================================================================
# ModelReadinessManager tests
# ===========================================================================


class TestModelReadinessManager:
    def test_initial_state_not_ready(self) -> None:
        mgr = ModelReadinessManager()
        assert not mgr.is_ready("phi-4")

    def test_mark_ready(self) -> None:
        mgr = ModelReadinessManager()
        mgr.mark_ready("phi-4", Path("/tmp/phi-4.gguf"))
        assert mgr.is_ready("phi-4")
        assert mgr.get_path("phi-4") == Path("/tmp/phi-4.gguf")

    def test_get_path_returns_none_when_not_ready(self) -> None:
        mgr = ModelReadinessManager()
        assert mgr.get_path("nonexistent") is None

    def test_await_ready_already_cached(self) -> None:
        mgr = ModelReadinessManager()
        mgr.mark_ready("phi-4", Path("/models/phi-4.gguf"))
        path = mgr.await_ready("phi-4", timeout=0.1)
        assert path == Path("/models/phi-4.gguf")

    def test_enqueue_ignores_non_managed(self) -> None:
        mgr = ModelReadinessManager()
        entry = AppModelEntry(
            id="phi-4",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.BUNDLED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        mgr.enqueue(entry)
        assert not mgr.is_ready("phi-4")

    def test_callback_on_mark_ready(self) -> None:
        mgr = ModelReadinessManager()
        updates: list[DownloadUpdate] = []
        mgr.add_callback(updates.append)
        mgr.mark_ready("phi-4", Path("/tmp/phi-4.gguf"))
        assert len(updates) == 1
        assert updates[0].model_id == "phi-4"
        assert updates[0].status == DownloadStatus.READY


# ===========================================================================
# ModelCatalogService tests
# ===========================================================================


class TestModelCatalogService:
    def test_bootstrap_cloud(self) -> None:
        entry = AppModelEntry(
            id="gpt-4",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.CLOUD,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[entry])
        stub = StubRuntime("cloud response")

        catalog = ModelCatalogService(
            manifest=manifest,
            cloud_runtime_factory=lambda _: stub,
        )
        catalog.bootstrap()

        assert catalog.runtime_for_capability(ModelCapability.CHAT) is stub

    def test_bootstrap_cloud_no_factory_required_raises(self) -> None:
        entry = AppModelEntry(
            id="gpt-4",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.CLOUD,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            required=True,
        )
        manifest = AppManifest(models=[entry])
        catalog = ModelCatalogService(manifest=manifest)

        with pytest.raises(RuntimeError, match="No cloud runtime factory"):
            catalog.bootstrap()

    def test_bootstrap_cloud_no_factory_optional_skipped(self) -> None:
        entry = AppModelEntry(
            id="gpt-4",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.CLOUD,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            required=False,
        )
        manifest = AppManifest(models=[entry])
        catalog = ModelCatalogService(manifest=manifest)
        catalog.bootstrap()
        assert catalog.runtime_for_capability(ModelCapability.CHAT) is None

    def test_bootstrap_bundled_no_path_raises(self) -> None:
        entry = AppModelEntry(
            id="phi-4",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.BUNDLED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[entry])
        catalog = ModelCatalogService(manifest=manifest)

        with pytest.raises(RuntimeError, match="no bundled_path"):
            catalog.bootstrap()

    def test_runtime_for_ref_by_id(self) -> None:
        entry = AppModelEntry(
            id="gpt-4",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.CLOUD,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[entry])
        stub = StubRuntime()

        registry = ModelRuntimeRegistry()
        catalog = ModelCatalogService(
            manifest=manifest,
            cloud_runtime_factory=lambda _: stub,
            runtime_registry=registry,
        )
        catalog.bootstrap()

        ref = _ModelRefId(model_id="gpt-4")
        result = catalog.runtime_for_ref(ref)
        assert result is stub

    def test_runtime_for_ref_by_capability(self) -> None:
        entry = AppModelEntry(
            id="gpt-4",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.CLOUD,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[entry])
        stub = StubRuntime()

        catalog = ModelCatalogService(
            manifest=manifest,
            cloud_runtime_factory=lambda _: stub,
        )
        catalog.bootstrap()

        ref = _ModelRefCapability(capability=ModelCapability.CHAT)
        result = catalog.runtime_for_ref(ref)
        assert result is stub

    def test_managed_model_queued(self) -> None:
        entry = AppModelEntry(
            id="phi-4-mini",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.MANAGED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[entry])
        catalog = ModelCatalogService(manifest=manifest)
        catalog.bootstrap()

        assert catalog.runtime_for_capability(ModelCapability.CHAT) is None

    def test_on_model_ready_registers_runtime(self) -> None:
        entry = AppModelEntry(
            id="phi-4-mini",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.MANAGED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[entry])
        catalog = ModelCatalogService(manifest=manifest)
        catalog.bootstrap()

        with patch("octomil.runtime.engines.local_file_runtime.LocalFileModelRuntime") as mock_cls:
            mock_runtime = StubRuntime()
            mock_cls.return_value = mock_runtime
            catalog.on_model_ready(entry, Path("/tmp/phi-4-mini.gguf"))

        assert catalog.runtime_for_capability(ModelCapability.CHAT) is mock_runtime


# ===========================================================================
# LocalFileModelRuntime tests
# ===========================================================================


class TestLocalFileModelRuntime:
    def test_model_id_and_path(self) -> None:
        from octomil.runtime.engines.local_file_runtime import LocalFileModelRuntime

        rt = LocalFileModelRuntime(model_id="phi-4", file_path=Path("/tmp/phi-4.gguf"))
        assert rt.model_id == "phi-4"
        assert rt.file_path == Path("/tmp/phi-4.gguf")

    def test_capabilities_default_streaming(self) -> None:
        from octomil.runtime.engines.local_file_runtime import LocalFileModelRuntime

        rt = LocalFileModelRuntime(model_id="phi-4", file_path=Path("/tmp/phi-4.gguf"))
        assert rt.capabilities.supports_streaming is True


# ===========================================================================
# OctomilResponses 3-step resolution tests
# ===========================================================================


class TestResponses3StepResolution:
    def test_step1_catalog_resolves(self) -> None:
        from octomil.responses.responses import OctomilResponses

        stub = StubRuntime("catalog hit")
        mock_catalog = MagicMock()
        mock_catalog.runtime_for_ref.return_value = stub

        responses = OctomilResponses(catalog=mock_catalog)
        runtime = responses._resolve_runtime("phi-4-mini")
        assert runtime is stub

    def test_step2_custom_resolver(self) -> None:
        from octomil.responses.responses import OctomilResponses

        stub = StubRuntime("custom resolver")
        responses = OctomilResponses(runtime_resolver=lambda _: stub)
        runtime = responses._resolve_runtime("phi-4-mini")
        assert runtime is stub

    def test_step3_registry_fallback(self) -> None:
        from octomil.responses.responses import OctomilResponses

        stub = StubRuntime("registry fallback")
        ModelRuntimeRegistry.shared().register("phi-4", lambda _: stub)

        responses = OctomilResponses()
        runtime = responses._resolve_runtime("phi-4-mini")
        assert runtime is stub

    def test_raises_when_no_runtime(self) -> None:
        from octomil.responses.responses import (
            NoRuntimeAvailableError,
            OctomilResponses,
        )

        responses = OctomilResponses()
        # The error type was upgraded to ``NoRuntimeAvailableError`` (a
        # ``RuntimeError`` subclass) so existing ``except RuntimeError``
        # blocks still catch it. The new message replaces the bare
        # ``"No ModelRuntime registered"`` string with a structured,
        # actionable diagnostic — see ``test_responses_api.py`` for the
        # full message-shape regression tests.
        with pytest.raises(NoRuntimeAvailableError, match="No runtime available"):
            responses._resolve_runtime("nonexistent")

    def test_catalog_takes_priority_over_custom(self) -> None:
        from octomil.responses.responses import OctomilResponses

        catalog_stub = StubRuntime("catalog")
        custom_stub = StubRuntime("custom")

        mock_catalog = MagicMock()
        mock_catalog.runtime_for_ref.return_value = catalog_stub

        responses = OctomilResponses(
            runtime_resolver=lambda _: custom_stub,
            catalog=mock_catalog,
        )
        runtime = responses._resolve_runtime("phi-4")
        assert runtime is catalog_stub

    def test_falls_through_catalog_to_custom(self) -> None:
        from octomil.responses.responses import OctomilResponses

        custom_stub = StubRuntime("custom")

        mock_catalog = MagicMock()
        mock_catalog.runtime_for_ref.return_value = None

        responses = OctomilResponses(
            runtime_resolver=lambda _: custom_stub,
            catalog=mock_catalog,
        )
        runtime = responses._resolve_runtime("phi-4")
        assert runtime is custom_stub

    def test_resolve_with_model_ref(self) -> None:
        from octomil.responses.responses import OctomilResponses

        stub = StubRuntime("ref-resolved")
        mock_catalog = MagicMock()
        mock_catalog.runtime_for_ref.return_value = stub

        responses = OctomilResponses(catalog=mock_catalog)
        ref = _ModelRefCapability(capability=ModelCapability.CHAT)
        runtime = responses._resolve_runtime(ref)
        assert runtime is stub


# ===========================================================================
# OctomilAudio / AudioTranscriptions tests
# ===========================================================================


class TestAudioTranscriptions:
    @pytest.mark.asyncio
    async def test_create_returns_transcription(self) -> None:
        from octomil.audio.transcriptions import AudioTranscriptions

        stub = StubRuntime("Hello world")
        transcriptions = AudioTranscriptions(runtime_resolver=lambda _: stub)
        result = await transcriptions.create(audio=b"fake-audio-data")
        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_create_with_language_hint(self) -> None:
        from octomil.audio.transcriptions import AudioTranscriptions

        stub = StubRuntime("Bonjour")
        transcriptions = AudioTranscriptions(runtime_resolver=lambda _: stub)
        result = await transcriptions.create(audio=b"audio", language="fr")
        assert result.text == "Bonjour"
        assert result.language == "fr"

    @pytest.mark.asyncio
    async def test_create_raises_when_no_runtime(self) -> None:
        from octomil.audio.transcriptions import AudioTranscriptions

        transcriptions = AudioTranscriptions(runtime_resolver=lambda _: None)
        with pytest.raises(RuntimeError, match="No runtime"):
            await transcriptions.create(audio=b"audio")


class TestOctomilAudio:
    def test_has_transcriptions_property(self) -> None:
        from octomil.audio import OctomilAudio

        audio = OctomilAudio(runtime_resolver=lambda _: None)
        assert audio.transcriptions is not None


# ===========================================================================
# OctomilText / OctomilPredictor tests
# ===========================================================================


class TestOctomilText:
    @pytest.mark.asyncio
    async def test_predict_returns_suggestions(self) -> None:
        from octomil.text import OctomilText

        stub = StubRuntime("fox\njumps\nover")
        text = OctomilText(runtime_resolver=lambda _: stub)
        result = await text.predict("The quick brown")
        assert result == ["fox", "jumps", "over"]

    @pytest.mark.asyncio
    async def test_predict_max_suggestions(self) -> None:
        from octomil.text import OctomilText

        stub = StubRuntime("fox\njumps\nover\nthe lazy dog")
        text = OctomilText(runtime_resolver=lambda _: stub)
        result = await text.predict("The quick brown", max_suggestions=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_predict_raises_no_runtime(self) -> None:
        from octomil.text import OctomilText

        text = OctomilText(runtime_resolver=lambda _: None)
        with pytest.raises(RuntimeError, match="No runtime"):
            await text.predict("Hello")

    def test_predictor_returns_instance(self) -> None:
        from octomil.text import OctomilText
        from octomil.text.predictor import OctomilPredictor

        stub = StubRuntime()
        text = OctomilText(runtime_resolver=lambda _: stub)
        predictor = text.predictor()
        assert isinstance(predictor, OctomilPredictor)

    def test_predictor_returns_none_no_runtime(self) -> None:
        from octomil.text import OctomilText

        text = OctomilText(runtime_resolver=lambda _: None)
        predictor = text.predictor()
        assert predictor is None


class TestOctomilPredictor:
    @pytest.mark.asyncio
    async def test_predict(self) -> None:
        from octomil.text.predictor import OctomilPredictor

        stub = StubRuntime("completed text\nmore text")
        predictor = OctomilPredictor(runtime=stub, model_id="test")
        result = await predictor.predict("The quick")
        assert len(result) >= 1

    def test_close(self) -> None:
        stub = StubRuntime()
        stub.close = MagicMock()  # type: ignore[method-assign]
        from octomil.text.predictor import OctomilPredictor

        predictor = OctomilPredictor(runtime=stub, model_id="test")
        predictor.close()
        stub.close.assert_called_once()

    def test_context_manager(self) -> None:
        stub = StubRuntime()
        stub.close = MagicMock()  # type: ignore[method-assign]
        from octomil.text.predictor import OctomilPredictor

        predictor = OctomilPredictor(runtime=stub, model_id="test")
        with predictor as p:
            assert p is predictor
        stub.close.assert_called_once()


# ===========================================================================
# OctomilClient wiring tests
# ===========================================================================


class TestClientWiring:
    def _make_client(self):  # type: ignore[no-untyped-def]
        from octomil.auth import OrgApiKeyAuth
        from octomil.client import OctomilClient

        return OctomilClient(
            auth=OrgApiKeyAuth(api_key="test-key", org_id="test-org"),
        )

    def test_audio_property(self) -> None:
        from octomil.audio import OctomilAudio

        client = self._make_client()
        assert isinstance(client.audio, OctomilAudio)

    def test_text_property(self) -> None:
        from octomil.text import OctomilText

        client = self._make_client()
        assert isinstance(client.text, OctomilText)

    def test_catalog_none_before_configure(self) -> None:
        client = self._make_client()
        assert client.catalog is None

    def test_configure_sets_catalog(self) -> None:
        client = self._make_client()
        manifest = AppManifest(models=[])
        client.configure(manifest=manifest)
        assert client.catalog is not None

    def test_responses_passes_catalog(self) -> None:
        client = self._make_client()
        manifest = AppManifest(models=[])
        client.configure(manifest=manifest)
        responses = client.responses
        assert responses._catalog is not None

    def test_resolve_model_ref_via_registry(self) -> None:
        client = self._make_client()
        stub = StubRuntime()
        ModelRuntimeRegistry.shared().register("phi-4", lambda _: stub)
        ref = _ModelRefId(model_id="phi-4")
        result = client._resolve_model_ref(ref)
        assert result is stub

    def test_resolve_model_ref_capability_returns_none_without_catalog(self) -> None:
        client = self._make_client()
        ref = _ModelRefCapability(capability=ModelCapability.CHAT)
        result = client._resolve_model_ref(ref)
        assert result is None
