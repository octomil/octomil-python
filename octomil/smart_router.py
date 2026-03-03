"""Multi-engine inference router.

Loads multiple inference backends for the same model and dispatches each
request to the most suitable engine based on runtime conditions and request
characteristics.  The router is itself an ``InferenceBackend``, so it
slots into the existing serving infrastructure transparently.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from .serve import GenerationChunk, GenerationRequest, InferenceBackend, InferenceMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterConfig:
    """Tunable routing parameters.

    Routing strategy:
      1. Long generation (>= long_gen_threshold tokens) -> prefer_latency_engine
      2. Concurrent (>= concurrency_threshold inflight) -> prefer_throughput_engine
      3. Single short request -> prefer_latency_engine

    Defaults are fetched from the server via ``DeviceConfigClient``.
    When the server is unreachable, falls back to safe "auto" defaults
    that pick the first available engine.
    """

    long_gen_threshold: int = 256
    concurrency_threshold: int = 2
    prefer_throughput_engine: str = "auto"
    prefer_latency_engine: str = "auto"


def _default_router_config() -> RouterConfig:
    """Build a RouterConfig from server-fetched device config."""
    from octomil.device_config import get_device_config

    sr = get_device_config().smart_router
    return RouterConfig(
        long_gen_threshold=sr.long_gen_threshold,
        concurrency_threshold=sr.concurrency_threshold,
        prefer_throughput_engine=sr.prefer_throughput_engine,
        prefer_latency_engine=sr.prefer_latency_engine,
    )


# ---------------------------------------------------------------------------
# SmartRouter
# ---------------------------------------------------------------------------


class SmartRouter(InferenceBackend):
    """Dispatches inference requests across multiple loaded backends.

    Typical usage::

        router = SmartRouter()
        await router.load("gemma2-2b")  # loads all available engines
        async for chunk in router.generate_stream(request):
            print(chunk.text, end="")
    """

    name = "smart_router"

    def __init__(
        self,
        *,
        config: Optional[RouterConfig] = None,
        cache_size_mb: int = 2048,
        cache_enabled: bool = True,
    ) -> None:
        self._config = config or _default_router_config()
        self._backends: dict[str, InferenceBackend] = {}
        self._model_name: str = ""
        self._cache_size_mb = cache_size_mb
        self._cache_enabled = cache_enabled

        # Concurrency tracking
        self._inflight = 0
        self._inflight_lock = threading.Lock()

        # Telemetry
        self._route_counts: dict[str, int] = {}

    # -- Setup ---------------------------------------------------------------

    def load_model(self, model_name: str) -> None:
        """Load all available engines for *model_name* (SDK canonical name).

        Each engine's own ``load_model`` resolves the canonical name to the
        appropriate format (HF repo for MLX, GGUF repo for llama.cpp, etc.).
        """
        from .engines import get_registry

        self._model_name = model_name
        registry = get_registry()
        detections = registry.detect_all(model_name)

        backend_kwargs: dict[str, Any] = {
            "cache_size_mb": self._cache_size_mb,
            "cache_enabled": self._cache_enabled,
        }

        target_engines = {self._config.prefer_throughput_engine, self._config.prefer_latency_engine}

        for det in detections:
            if not det.available or det.engine.name not in target_engines:
                continue
            try:
                backend = det.engine.create_backend(model_name, **backend_kwargs)
                self._backends[det.engine.name] = backend
                logger.info("Loaded %s backend for %s", det.engine.name, model_name)
            except Exception:
                logger.warning("Failed to load %s for %s", det.engine.name, model_name, exc_info=True)

        if not self._backends:
            raise RuntimeError(f"SmartRouter: no backend could load '{model_name}'. Tried: {', '.join(target_engines)}")

        logger.info(
            "SmartRouter ready: %s (%s)",
            model_name,
            " + ".join(sorted(self._backends)),
        )

    def unload(self) -> None:
        """Release all loaded backends."""
        self._backends.clear()
        self._model_name = ""
        self._inflight = 0

    # -- Routing -------------------------------------------------------------

    def _select(self, request: GenerationRequest) -> tuple[InferenceBackend, str]:
        """Pick the best backend for this request."""
        cfg = self._config
        available = self._backends

        if len(available) == 1:
            name = next(iter(available))
            return available[name], name

        throughput = available.get(cfg.prefer_throughput_engine)
        latency = available.get(cfg.prefer_latency_engine)

        # Long generation — use the engine with stable completion at high token counts
        if request.max_tokens >= cfg.long_gen_threshold and latency:
            return latency, cfg.prefer_latency_engine

        # Concurrent load — use the engine optimised for batch throughput
        with self._inflight_lock:
            concurrent = self._inflight

        if concurrent >= cfg.concurrency_threshold and throughput:
            return throughput, cfg.prefer_throughput_engine

        # Default: lowest latency for single short requests
        if latency:
            return latency, cfg.prefer_latency_engine

        # Final fallback
        name = next(iter(available))
        return available[name], name

    # -- Inference -----------------------------------------------------------

    def generate(self, request: GenerationRequest) -> tuple[str, InferenceMetrics]:
        with self._inflight_lock:
            self._inflight += 1
        try:
            backend, engine_name = self._select(request)
            self._route_counts[engine_name] = self._route_counts.get(engine_name, 0) + 1
            return backend.generate(request)
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationChunk]:
        with self._inflight_lock:
            self._inflight += 1
        try:
            backend, engine_name = self._select(request)
            self._route_counts[engine_name] = self._route_counts.get(engine_name, 0) + 1
            async for chunk in backend.generate_stream(request):
                yield chunk
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []

    # -- Introspection -------------------------------------------------------

    @property
    def engines_loaded(self) -> list[str]:
        """Names of currently loaded backends."""
        return list(self._backends.keys())

    @property
    def route_stats(self) -> dict[str, int]:
        """Per-engine request counts since load."""
        return dict(self._route_counts)
