"""ModelCatalogService — bootstraps AppManifest into the runtime registry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from octomil._generated.delivery_mode import DeliveryMode
from octomil._generated.model_capability import ModelCapability
from octomil.manifest.readiness_manager import ModelReadinessManager
from octomil.manifest.types import AppManifest, AppModelEntry
from octomil.model_ref import ModelRef, _ModelRefCapability, _ModelRefId
from octomil.runtime.core.model_runtime import ModelRuntime, RuntimeFactory
from octomil.runtime.core.registry import ModelRuntimeRegistry

logger = logging.getLogger(__name__)


def _constant_factory(runtime: ModelRuntime) -> RuntimeFactory:
    """Create a RuntimeFactory that always returns the given runtime."""

    def factory(_model_id: str) -> Optional[ModelRuntime]:
        return runtime

    return factory


class ModelCatalogService:
    """Bootstraps AppManifest entries into the ModelRuntimeRegistry.

    The catalog service bridges the declarative manifest and the
    runtime system. It does NOT reference any specific engine — engine
    selection is delegated to EngineRegistry and ModelRuntimeRegistry.
    """

    def __init__(
        self,
        manifest: AppManifest,
        readiness: Optional[ModelReadinessManager] = None,
        runtime_registry: Optional[ModelRuntimeRegistry] = None,
        cloud_runtime_factory: Optional[Callable[[str], ModelRuntime]] = None,
    ) -> None:
        self._manifest = manifest
        self._readiness = readiness or ModelReadinessManager()
        self._registry = runtime_registry or ModelRuntimeRegistry.shared()
        self._cloud_runtime_factory = cloud_runtime_factory
        self._capability_runtimes: dict[ModelCapability, ModelRuntime] = {}

    @property
    def manifest(self) -> AppManifest:
        return self._manifest

    def bootstrap(self) -> None:
        """Walk every manifest entry and prepare its runtime.

        - BUNDLED: look up the local file, create a LocalFileModelRuntime.
        - MANAGED: queue a background download via ModelReadinessManager.
        - CLOUD: register a cloud runtime immediately.

        Required entries that cannot be resolved raise RuntimeError.
        """
        for entry in self._manifest.models:
            try:
                self._bootstrap_entry(entry)
            except Exception as exc:
                if entry.required:
                    raise
                logger.warning("Optional model '%s' skipped: %s", entry.id, exc)

    def runtime_for_capability(self, capability: ModelCapability) -> Optional[ModelRuntime]:
        """Resolve a ModelRuntime for a given capability."""
        return self._capability_runtimes.get(capability)

    def runtime_for_ref(self, ref: ModelRef) -> Optional[ModelRuntime]:
        """Resolve a ModelRuntime for a ModelRef."""
        if isinstance(ref, _ModelRefId):
            return self._registry.resolve(ref.model_id)
        if isinstance(ref, _ModelRefCapability):
            return self._capability_runtimes.get(ref.capability)
        return None

    def on_model_ready(self, entry: AppModelEntry, path: Path) -> None:
        """Called when a managed download completes."""
        from octomil.runtime.engines.local_file_runtime import LocalFileModelRuntime

        runtime = LocalFileModelRuntime(model_id=entry.id, file_path=path)
        self._capability_runtimes[entry.capability] = runtime
        self._registry.register(family=entry.id, factory=_constant_factory(runtime))
        logger.info("Managed model '%s' now ready", entry.id)

    def _bootstrap_entry(self, entry: AppModelEntry) -> None:
        if entry.delivery == DeliveryMode.BUNDLED:
            self._bootstrap_bundled(entry)
        elif entry.delivery == DeliveryMode.MANAGED:
            self._bootstrap_managed(entry)
        elif entry.delivery == DeliveryMode.CLOUD:
            self._bootstrap_cloud(entry)

    def _bootstrap_bundled(self, entry: AppModelEntry) -> None:
        from octomil.runtime.engines.local_file_runtime import LocalFileModelRuntime

        if entry.bundled_path is None:
            raise RuntimeError(f"Bundled model '{entry.id}' has no bundled_path")
        path = Path(entry.bundled_path)
        if not path.exists():
            raise RuntimeError(f"Bundled model '{entry.id}' not found at {path}")
        runtime = LocalFileModelRuntime(model_id=entry.id, file_path=path)
        self._capability_runtimes[entry.capability] = runtime
        self._registry.register(family=entry.id, factory=_constant_factory(runtime))
        logger.info("Bundled model '%s' registered for capability '%s'", entry.id, entry.capability.value)

    def _bootstrap_managed(self, entry: AppModelEntry) -> None:
        # Check if already cached
        cached_path = self._readiness.get_path(entry.id)
        if cached_path is not None:
            self.on_model_ready(entry, cached_path)
            logger.info("Managed model '%s' loaded from cache", entry.id)
            return
        # Queue download
        self._readiness.enqueue(entry)
        logger.info("Managed model '%s' queued for download", entry.id)

    def _bootstrap_cloud(self, entry: AppModelEntry) -> None:
        if self._cloud_runtime_factory is None:
            raise RuntimeError(f"No cloud runtime factory for model '{entry.id}'")
        runtime = self._cloud_runtime_factory(entry.id)
        self._capability_runtimes[entry.capability] = runtime
        self._registry.register(family=entry.id, factory=_constant_factory(runtime))
        logger.info("Cloud model '%s' registered for capability '%s'", entry.id, entry.capability.value)
