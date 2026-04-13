"""Runtime planner -- resolves the best engine/locality for a request."""

from __future__ import annotations

import json
import logging
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
        self._client = client

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
        installed_engines = {r.engine for r in device.installed_runtimes}

        # Try primary candidates
        for candidate in plan.candidates:
            if candidate.locality == "local":
                if candidate.engine and candidate.engine not in installed_engines:
                    continue  # Skip engines we don't have
                return RuntimeSelection(
                    locality="local",
                    engine=candidate.engine,
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
            if candidate.locality == "local" and candidate.engine in installed_engines:
                return RuntimeSelection(
                    locality="local",
                    engine=candidate.engine,
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
        bm_cache_key = self._store._make_cache_key(
            model=model,
            capability=capability,
            type="benchmark",
        )
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
            from octomil.runtime.core.engine_bridge import _select_real_engine
            from octomil.runtime.engines import get_registry

            registry = get_registry()
            engine, real_engines = _select_real_engine(registry, model)

            if engine is not None:
                # Step 7: Persist benchmark locally
                self._store.put_benchmark(
                    bm_cache_key,
                    model=model,
                    capability=capability,
                    engine=engine.name,
                )

                # Step 8: Upload telemetry best-effort (not for private)
                if not is_private and self._client is not None:
                    try:
                        self._client.upload_benchmark(
                            {
                                "model": model,
                                "capability": capability,
                                "engine": engine.name,
                                "platform": device.platform,
                                "arch": device.arch,
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
        installed_engines = {r.engine for r in device.installed_runtimes}

        for c in candidates:
            if c.get("locality") == "local":
                engine = c.get("engine")
                if engine and engine not in installed_engines:
                    continue
                return RuntimeSelection(
                    locality="local",
                    engine=engine,
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
