"""RouterModelRuntime — Layer 4 routing between local and cloud runtimes."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from octomil._generated.routing_policy import RoutingPolicy as ContractRoutingPolicy
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

    async def run(self, request: RuntimeRequest, *, policy: Optional[RoutingPolicy] = None) -> RuntimeResponse:
        runtime, _meta = self._select_runtime_with_locality(policy)
        return await runtime.run(request)

    async def stream(
        self, request: RuntimeRequest, *, policy: Optional[RoutingPolicy] = None
    ) -> AsyncIterator[RuntimeChunk]:
        runtime, _meta = self._select_runtime_with_locality(policy)
        async for chunk in runtime.stream(request):
            yield chunk

    def resolve_locality(self, policy: Optional[RoutingPolicy] = None) -> tuple[str, bool]:
        """Return (locality, is_fallback) without executing inference.

        locality: LOCALITY_ON_DEVICE or LOCALITY_CLOUD
        is_fallback: True when local was unavailable and cloud is used as fallback.
        """
        _, meta = self._select_runtime_with_locality(policy)
        return meta

    def _select_runtime(self) -> ModelRuntime:
        runtime, _meta = self._select_runtime_with_locality()
        return runtime

    def _select_runtime_with_locality(
        self, policy_override: Optional[RoutingPolicy] = None
    ) -> tuple[ModelRuntime, tuple[str, bool]]:
        """Select the runtime and return (runtime, (locality, is_fallback))."""
        policy = policy_override or self._default_policy

        if policy.mode == ContractRoutingPolicy.LOCAL_ONLY:
            local = self._local_factory("local") if self._local_factory else None
            if local is None:
                raise RuntimeError("No local runtime available")
            return local, (LOCALITY_ON_DEVICE, False)

        if policy.mode == ContractRoutingPolicy.CLOUD_ONLY:
            cloud = self._cloud_factory("cloud") if self._cloud_factory else None
            if cloud is None:
                raise RuntimeError("No cloud runtime available")
            return cloud, (LOCALITY_CLOUD, False)

        # LOCAL_FIRST always prefers local; AUTO depends on prefer_local.
        if policy.mode == ContractRoutingPolicy.LOCAL_FIRST or policy.prefer_local:
            # Try local first, fall back to cloud
            local = self._local_factory("local") if self._local_factory else None
            if local is not None:
                return local, (LOCALITY_ON_DEVICE, False)
            if policy.fallback == "cloud":
                cloud = self._cloud_factory("cloud") if self._cloud_factory else None
                if cloud is not None:
                    return cloud, (LOCALITY_CLOUD, True)
        else:
            # prefer_local=False (quality preset): try cloud first, fall back to local
            cloud = self._cloud_factory("cloud") if self._cloud_factory else None
            if cloud is not None:
                return cloud, (LOCALITY_CLOUD, False)
            local = self._local_factory("local") if self._local_factory else None
            if local is not None:
                return local, (LOCALITY_ON_DEVICE, True)
        raise RuntimeError("No runtime available")

    def close(self) -> None:
        pass
