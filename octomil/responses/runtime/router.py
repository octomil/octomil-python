"""RouterModelRuntime — Layer 4 routing between local and cloud runtimes."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from .model_runtime import ModelRuntime, RuntimeFactory
from .policy import RoutingPolicy
from .types import RuntimeCapabilities, RuntimeChunk, RuntimeRequest, RuntimeResponse


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
        return RuntimeCapabilities(supports_tool_calls=True, supports_streaming=True)

    async def run(self, request: RuntimeRequest) -> RuntimeResponse:
        runtime = self._select_runtime()
        return await runtime.run(request)

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[RuntimeChunk]:
        runtime = self._select_runtime()
        async for chunk in runtime.stream(request):
            yield chunk

    def _select_runtime(self) -> ModelRuntime:
        policy = self._default_policy
        if policy.mode == "local_only":
            local = self._local_factory("local") if self._local_factory else None
            if local is None:
                raise RuntimeError("No local runtime available")
            return local
        if policy.mode == "cloud_only":
            cloud = self._cloud_factory("cloud") if self._cloud_factory else None
            if cloud is None:
                raise RuntimeError("No cloud runtime available")
            return cloud
        # auto mode
        local = self._local_factory("local") if self._local_factory else None
        if local is not None:
            return local
        if policy.fallback == "cloud":
            cloud = self._cloud_factory("cloud") if self._cloud_factory else None
            if cloud is not None:
                return cloud
        raise RuntimeError("No runtime available")

    def close(self) -> None:
        pass
