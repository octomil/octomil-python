"""Runtime planner -- resolves the best engine/locality for a request."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict

from .client import RuntimePlannerClient
from .device_profile import collect_device_runtime_profile
from .schemas import (
    DeviceRuntimeProfile,
    RuntimePlanResponse,
    RuntimeSelection,
)
from .store import RuntimePlannerStore

logger = logging.getLogger(__name__)

_ENGINE_ALIASES = {
    "mlx": "mlx-lm",
    "mlx_lm": "mlx-lm",
    "llamacpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "whisper": "whisper.cpp",
    "whispercpp": "whisper.cpp",
    "whisper_cpp": "whisper.cpp",
}


class RuntimePlanner:
    """Resolves the best engine/locality for a given model + capability.

    Resolution order:
    1. Collect device profile
    2. Check local plan cache
    3. If allow_network and client configured, fetch server plan
    4. Validate server plan against installed runtimes
    5. Check local benchmark cache
    6. If multiple viable engines or benchmark_required, run benchmarks
    7. Persist benchmark locally
    8. Upload telemetry best-effort (not for private policy)
    9. Return RuntimeSelection
    """

    def __init__(
        self,
        *,
        store: RuntimePlannerStore | None = None,
        client: RuntimePlannerClient | None = None,
    ) -> None:
        self._store = store or RuntimePlannerStore()
        self._client = client if client is not None else _client_from_env()

    def resolve(
        self,
        *,
        model: str,
        capability: str,
        routing_policy: str = "local_first",
        allow_network: bool = True,
    ) -> RuntimeSelection:
        """Resolve the best runtime for the given request.

        Parameters
        ----------
        model:
            Model identifier (e.g. "gemma-2b", "llama-8b").
        capability:
            The capability needed (e.g. "text", "embeddings", "audio").
        routing_policy:
            One of "local_first", "cloud_first", "local_only", "cloud_only",
            "private".
        allow_network:
            Whether to contact the server for plan/telemetry.  Set to False
            for fully-offline operation.
        """
        # Step 1: Collect device profile
        device = collect_device_runtime_profile(model_id=model)

        # Step 2: Check local plan cache
        from octomil import __version__

        cache_key = self._store._make_cache_key(
            model=model,
            capability=capability,
            policy=routing_policy,
            sdk_version=__version__,
            platform=device.platform,
            arch=device.arch,
        )

        cached_plan = self._store.get_plan(cache_key)
        if cached_plan is not None:
            logger.debug("Using cached plan for %s/%s", model, capability)
            return self._selection_from_plan_dict(cached_plan, device=device, source="cache")

        # Step 3: Fetch server plan if network is allowed
        server_plan: RuntimePlanResponse | None = None
        is_private = routing_policy == "private"

        if allow_network and self._client is not None and not is_private:
            server_plan = self._client.fetch_plan(
                model=model,
                capability=capability,
                routing_policy=routing_policy,
                device=device,
                allow_cloud_fallback=routing_policy != "local_only",
            )
            if server_plan is not None:
                logger.debug("Received server plan for %s/%s", model, capability)
                # Cache the server plan
                plan_dict = asdict(server_plan)
                self._store.put_plan(
                    cache_key,
                    model=model,
                    capability=capability,
                    policy=routing_policy,
                    plan_json=json.dumps(plan_dict),
                    source="server_plan",
                    ttl_seconds=server_plan.plan_ttl_seconds,
                )

        # Step 4: Validate server plan against installed runtimes
        if server_plan is not None:
            selection = self._resolve_from_server_plan(server_plan, device=device, is_private=is_private)
            if selection is not None:
                return selection

        # Step 5-8: Fall back to local engine selection
        return self._resolve_locally(
            model=model,
            capability=capability,
            routing_policy=routing_policy,
            device=device,
            is_private=is_private,
        )

    def _resolve_from_server_plan(
        self,
        plan: RuntimePlanResponse,
        *,
        device: DeviceRuntimeProfile,
        is_private: bool,
    ) -> RuntimeSelection | None:
        """Try to select a runtime from a server plan."""
        installed_engines = {_canonical_engine_id(r.engine) for r in device.installed_runtimes}

        # Try primary candidates
        for candidate in plan.candidates:
            if candidate.locality == "local":
                engine = _canonical_engine_id(candidate.engine)
                if engine and engine not in installed_engines:
                    continue  # Skip engines we don't have
                return RuntimeSelection(
                    locality="local",
                    engine=engine or None,
                    artifact=candidate.artifact,
                    benchmark_ran=False,
                    source="server_plan",
                    fallback_candidates=plan.fallback_candidates,
                    reason=candidate.reason,
                )
            elif candidate.locality == "cloud":
                return RuntimeSelection(
                    locality="cloud",
                    engine=candidate.engine,
                    artifact=candidate.artifact,
                    benchmark_ran=False,
                    source="server_plan",
                    fallback_candidates=plan.fallback_candidates,
                    reason=candidate.reason,
                )

        # Try fallback candidates
        for candidate in plan.fallback_candidates:
            engine = _canonical_engine_id(candidate.engine)
            if candidate.locality == "local" and engine in installed_engines:
                return RuntimeSelection(
                    locality="local",
                    engine=engine,
                    artifact=candidate.artifact,
                    benchmark_ran=False,
                    source="server_plan",
                    fallback_candidates=[],
                    reason=f"fallback: {candidate.reason}",
                )

        return None

    def _resolve_locally(
        self,
        *,
        model: str,
        capability: str,
        routing_policy: str,
        device: DeviceRuntimeProfile,
        is_private: bool,
    ) -> RuntimeSelection:
        """Resolve using local engine detection and benchmarks."""
        # For cloud_only policy, return cloud selection without local work
        if routing_policy == "cloud_only":
            return RuntimeSelection(
                locality="cloud",
                engine=None,
                source="fallback",
                reason="cloud_only policy — no local engines attempted",
            )

        # Step 5: Check local benchmark cache
        bm_cache_key = _benchmark_cache_key(model=model, capability=capability, policy=routing_policy, device=device)
        cached_bm = self._store.get_benchmark(bm_cache_key)
        if cached_bm is not None:
            return RuntimeSelection(
                locality="local",
                engine=cached_bm.get("engine"),
                benchmark_ran=False,
                source="cache",
                reason=f"cached benchmark: {cached_bm.get('tokens_per_second', 0):.1f} tok/s",
            )

        # Step 6: Run local benchmark selection
        try:
            from octomil.runtime.engines import get_registry

            registry = get_registry()
            engine, benchmark_result = _select_local_engine(registry, model)

            if engine is not None and benchmark_result is not None:
                # Step 7: Persist benchmark locally
                self._store.put_benchmark(
                    bm_cache_key,
                    model=model,
                    capability=capability,
                    engine=engine.name,
                    policy=routing_policy,
                    platform=device.platform,
                    arch=device.arch,
                    chip=device.chip,
                    sdk_version=device.sdk_version,
                    installed_hash=_installed_runtimes_hash(device),
                    tokens_per_second=benchmark_result.tokens_per_second,
                    ttft_ms=benchmark_result.ttft_ms,
                    memory_mb=benchmark_result.memory_mb,
                )

                # Step 8: Upload telemetry best-effort (not for private)
                if not is_private and self._client is not None:
                    try:
                        self._client.upload_benchmark(
                            {
                                "source": "planner",
                                "model": model,
                                "capability": capability,
                                "engine": engine.name,
                                "device": asdict(device),
                                "success": True,
                                "tokens_per_second": benchmark_result.tokens_per_second,
                                "ttft_ms": benchmark_result.ttft_ms,
                                "peak_memory_bytes": int(benchmark_result.memory_mb * 1024 * 1024)
                                if benchmark_result.memory_mb
                                else None,
                                "benchmark_tokens": _benchmark_tokens(),
                                "metadata": {"selection_source": "local_benchmark"},
                            }
                        )
                    except Exception:
                        logger.debug("Telemetry upload failed", exc_info=True)

                return RuntimeSelection(
                    locality="local",
                    engine=engine.name,
                    benchmark_ran=True,
                    source="local_benchmark",
                    reason=f"local benchmark selected {engine.name}",
                )
        except Exception:
            logger.debug("Local engine selection failed", exc_info=True)

        # No local engine available — return fallback
        if routing_policy in ("local_only", "private"):
            return RuntimeSelection(
                locality="local",
                engine=None,
                source="fallback",
                reason="no local engine available",
            )

        return RuntimeSelection(
            locality="cloud",
            engine=None,
            source="fallback",
            reason="no local engine available — falling back to cloud",
        )

    def _selection_from_plan_dict(
        self,
        plan_dict: dict,
        *,
        device: DeviceRuntimeProfile,
        source: str,
    ) -> RuntimeSelection:
        """Convert a cached plan dict back into a RuntimeSelection."""
        candidates = plan_dict.get("candidates", [])
        installed_engines = {_canonical_engine_id(r.engine) for r in device.installed_runtimes}

        for c in candidates:
            if c.get("locality") == "local":
                engine = _canonical_engine_id(c.get("engine"))
                if engine and engine not in installed_engines:
                    continue
                return RuntimeSelection(
                    locality="local",
                    engine=engine or None,
                    source=source,
                    reason=c.get("reason", ""),
                )
            elif c.get("locality") == "cloud":
                return RuntimeSelection(
                    locality="cloud",
                    engine=c.get("engine"),
                    source=source,
                    reason=c.get("reason", ""),
                )

        # Nothing matched from cache — return generic fallback
        return RuntimeSelection(
            locality="local",
            engine=None,
            source="fallback",
            reason="cached plan had no viable candidates",
        )


def _client_from_env() -> RuntimePlannerClient | None:
    api_key = os.environ.get("OCTOMIL_SERVER_KEY") or os.environ.get("OCTOMIL_API_KEY")
    if not api_key:
        return None
    return RuntimePlannerClient(
        base_url=os.environ.get("OCTOMIL_API_BASE") or "https://api.octomil.com",
        api_key=api_key,
    )


def _canonical_engine_id(engine: str | None) -> str:
    if not engine:
        return ""
    cleaned = engine.strip()
    return _ENGINE_ALIASES.get(cleaned.lower(), cleaned)


def _benchmark_tokens() -> int:
    raw = os.environ.get("OCTOMIL_RUNTIME_BENCHMARK_TOKENS")
    if raw is None:
        return 16
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def _installed_runtimes_hash(device: DeviceRuntimeProfile) -> str:
    import hashlib

    runtimes = ",".join(
        sorted(
            f"{runtime.engine}:{runtime.version or ''}" for runtime in device.installed_runtimes if runtime.available
        )
    )
    return hashlib.sha256(runtimes.encode()).hexdigest()[:16]


def _benchmark_cache_key(
    *,
    model: str,
    capability: str,
    policy: str,
    device: DeviceRuntimeProfile,
) -> str:
    return RuntimePlannerStore._make_cache_key(
        model=model,
        capability=capability,
        policy=policy,
        sdk_version=device.sdk_version,
        platform=device.platform,
        arch=device.arch,
        chip=device.chip,
        installed_hash=_installed_runtimes_hash(device),
    )


def _select_local_engine(registry, model: str):
    """Benchmark installed real engines and return the fastest successful one."""
    detections = registry.detect_all(model)
    real_engines = [d.engine for d in detections if d.available and d.engine.name != "echo"]
    if not real_engines:
        return None, None

    ranked = registry.benchmark_all(model, n_tokens=_benchmark_tokens(), engines=real_engines)
    for ranked_engine in ranked:
        if ranked_engine.engine.name == "echo" or not ranked_engine.result.ok:
            continue
        return ranked_engine.engine, ranked_engine.result
    return None, None
