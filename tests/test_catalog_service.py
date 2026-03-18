"""Tests for octomil.manifest.catalog_service — ModelCatalogService."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import pytest

from octomil._generated.delivery_mode import DeliveryMode
from octomil._generated.modality import Modality
from octomil._generated.model_capability import ModelCapability
from octomil.manifest.catalog_service import ModelCatalogService
from octomil.manifest.readiness_manager import ModelReadinessManager
from octomil.manifest.types import AppManifest, AppModelEntry
from octomil.model_ref import ModelRefFactory
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.registry import ModelRuntimeRegistry
from octomil.runtime.core.types import (
    RuntimeCapabilities,
    RuntimeChunk,
    RuntimeRequest,
    RuntimeResponse,
)

_TEXT = [Modality.TEXT]
_AUDIO = [Modality.AUDIO]


class _FakeRuntime(ModelRuntime):
    """Minimal ModelRuntime for testing."""

    def __init__(self, model_id: str = "fake") -> None:
        self.model_id = model_id

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        raise NotImplementedError

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        raise NotImplementedError
        yield  # pragma: no cover


class TestCatalogServiceBundled:
    def test_bootstrap_bundled_model(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_text("fake model data")

        entry = AppModelEntry(
            id="test-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.BUNDLED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            bundled_path=str(model_file),
        )
        manifest = AppManifest(models=[entry])
        registry = ModelRuntimeRegistry()

        catalog = ModelCatalogService(
            manifest=manifest,
            runtime_registry=registry,
        )
        catalog.bootstrap()

        runtime = catalog.runtime_for_capability(ModelCapability.CHAT)
        assert runtime is not None

    def test_bootstrap_bundled_missing_path_raises(self) -> None:
        entry = AppModelEntry(
            id="test-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.BUNDLED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            bundled_path=None,
        )
        manifest = AppManifest(models=[entry])
        catalog = ModelCatalogService(manifest=manifest, runtime_registry=ModelRuntimeRegistry())

        with pytest.raises(RuntimeError, match="no bundled_path"):
            catalog.bootstrap()

    def test_bootstrap_bundled_file_not_found_raises(self) -> None:
        entry = AppModelEntry(
            id="test-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.BUNDLED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            bundled_path="/nonexistent/model.gguf",
        )
        manifest = AppManifest(models=[entry])
        catalog = ModelCatalogService(manifest=manifest, runtime_registry=ModelRuntimeRegistry())

        with pytest.raises(RuntimeError, match="not found"):
            catalog.bootstrap()

    def test_optional_bundled_skipped_on_error(self) -> None:
        entry = AppModelEntry(
            id="test-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.BUNDLED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            bundled_path=None,
            required=False,
        )
        manifest = AppManifest(models=[entry])
        catalog = ModelCatalogService(manifest=manifest, runtime_registry=ModelRuntimeRegistry())
        catalog.bootstrap()
        assert catalog.runtime_for_capability(ModelCapability.CHAT) is None


class TestCatalogServiceCloud:
    def test_bootstrap_cloud_model(self) -> None:
        fake_runtime = _FakeRuntime("cloud-model")
        entry = AppModelEntry(
            id="cloud-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.CLOUD,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[entry])
        catalog = ModelCatalogService(
            manifest=manifest,
            runtime_registry=ModelRuntimeRegistry(),
            cloud_runtime_factory=lambda _mid: fake_runtime,
        )
        catalog.bootstrap()

        runtime = catalog.runtime_for_capability(ModelCapability.CHAT)
        assert runtime is fake_runtime

    def test_bootstrap_cloud_no_factory_raises(self) -> None:
        entry = AppModelEntry(
            id="cloud-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.CLOUD,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[entry])
        catalog = ModelCatalogService(
            manifest=manifest,
            runtime_registry=ModelRuntimeRegistry(),
        )
        with pytest.raises(RuntimeError, match="No cloud runtime factory"):
            catalog.bootstrap()


class TestCatalogServiceManaged:
    def test_bootstrap_managed_queues_download(self) -> None:
        readiness = ModelReadinessManager()
        entry = AppModelEntry(
            id="managed-model",
            capability=ModelCapability.TRANSCRIPTION,
            delivery=DeliveryMode.MANAGED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            download_url="https://example.com/model.bin",
        )
        manifest = AppManifest(models=[entry])
        catalog = ModelCatalogService(
            manifest=manifest,
            readiness=readiness,
            runtime_registry=ModelRuntimeRegistry(),
        )
        catalog.bootstrap()
        assert catalog.runtime_for_capability(ModelCapability.TRANSCRIPTION) is None

    def test_on_model_ready_registers_runtime(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.bin"
        model_file.write_text("fake")

        entry = AppModelEntry(
            id="managed-model",
            capability=ModelCapability.TRANSCRIPTION,
            delivery=DeliveryMode.MANAGED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        manifest = AppManifest(models=[entry])
        registry = ModelRuntimeRegistry()
        catalog = ModelCatalogService(
            manifest=manifest,
            runtime_registry=registry,
        )
        catalog.on_model_ready(entry, model_file)
        runtime = catalog.runtime_for_capability(ModelCapability.TRANSCRIPTION)
        assert runtime is not None


class TestCatalogServiceResolution:
    def test_runtime_for_ref_by_id(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_text("fake")

        entry = AppModelEntry(
            id="test-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.BUNDLED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            bundled_path=str(model_file),
        )
        manifest = AppManifest(models=[entry])
        registry = ModelRuntimeRegistry()
        catalog = ModelCatalogService(manifest=manifest, runtime_registry=registry)
        catalog.bootstrap()

        ref = ModelRefFactory.id("test-model")
        runtime = catalog.runtime_for_ref(ref)
        assert runtime is not None

    def test_runtime_for_ref_by_capability(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_text("fake")

        entry = AppModelEntry(
            id="test-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.BUNDLED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            bundled_path=str(model_file),
        )
        manifest = AppManifest(models=[entry])
        registry = ModelRuntimeRegistry()
        catalog = ModelCatalogService(manifest=manifest, runtime_registry=registry)
        catalog.bootstrap()

        ref = ModelRefFactory.capability(ModelCapability.CHAT)
        runtime = catalog.runtime_for_ref(ref)
        assert runtime is not None

    def test_runtime_for_ref_not_found(self) -> None:
        manifest = AppManifest(models=[])
        catalog = ModelCatalogService(manifest=manifest, runtime_registry=ModelRuntimeRegistry())
        ref = ModelRefFactory.id("nonexistent")
        assert catalog.runtime_for_ref(ref) is None
