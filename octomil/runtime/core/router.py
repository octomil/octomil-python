"""RouterModelRuntime — Layer 4 routing between local and cloud runtimes."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from octomil.runtime.core.model_runtime import ModelRuntime, RuntimeFactory
from octomil.runtime.core.policy import RoutingPolicy
from octomil.runtime.core.types import RuntimeCapabilities, RuntimeChunk, RuntimeRequest, RuntimeResponse

# Locality values emitted on telemetry spans.
LOCALITY_ON_DEVICE = "on_device"
LOCALITY_CLOUD = "cloud"


class RouterModelRuntime(ModelRuntime):
    def __init__(
        self,
        local_factory: Optional[RuntimeFactory] = None,
        cloud_factory: Optional[RuntimeFactory] = None,
        default_policy: RoutingPolicy = RoutingPolicy.auto(),
    ) -> None:
        self._local_factory = local_factory
        self._cloud_factory = cloud_factory
        self._default_policy = default_policy

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_streaming=True)

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        runtime, _meta = self._select_runtime_with_locality()
        return await runtime.run(request)

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        runtime, _meta = self._select_runtime_with_locality()
        async for chunk in runtime.stream(request):
            yield chunk

    def resolve_locality(self) -> tuple[str, bool]:
        """Return (locality, is_fallback) without executing inference.

        locality: LOCALITY_ON_DEVICE or LOCALITY_CLOUD
        is_fallback: True when local was unavailable and cloud is used as fallback.
        """
        _, meta = self._select_runtime_with_locality()
        return meta

    def _select_runtime(self) -> ModelRuntime:
        runtime, _meta = self._select_runtime_with_locality()
        return runtime

    def _select_runtime_with_locality(self) -> tuple[ModelRuntime, tuple[str, bool]]:
        """Select the runtime and return (runtime, (locality, is_fallback))."""
        policy = self._default_policy
        if policy.mode == "local_only":
            local = self._local_factory("local") if self._local_factory else None
            if local is None:
                raise RuntimeError("No local runtime available")
            return local, (LOCALITY_ON_DEVICE, False)
        if policy.mode == "cloud_only":
            cloud = self._cloud_factory("cloud") if self._cloud_factory else None
            if cloud is None:
                raise RuntimeError("No cloud runtime available")
            return cloud, (LOCALITY_CLOUD, False)
        # auto mode — prefer local, fall back to cloud
        local = self._local_factory("local") if self._local_factory else None
        if local is not None:
            return local, (LOCALITY_ON_DEVICE, False)
        if policy.fallback == "cloud":
            cloud = self._cloud_factory("cloud") if self._cloud_factory else None
            if cloud is not None:
                return cloud, (LOCALITY_CLOUD, True)
        raise RuntimeError("No runtime available")

    def close(self) -> None:
        pass
