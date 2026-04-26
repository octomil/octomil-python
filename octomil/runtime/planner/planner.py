"""Runtime planner -- resolves the best engine/locality for a request."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict

from .app_ref import is_app_ref, parse_app_ref
from .client import RuntimePlannerClient
from .device_profile import collect_device_runtime_profile
from .schemas import (
    AppResolution,
    CandidateGate,
    DeviceRuntimeProfile,
    ModelResolution,
    RuntimeArtifactPlan,
    RuntimeCandidatePlan,
    RuntimePlanResponse,
    RuntimeSelection,
)
from .store import (
    RuntimePlannerStoreProtocol,
    _make_cache_key,
    build_runtime_planner_store,
)

logger = logging.getLogger(__name__)

_ENGINE_ALIASES = {
    "mlx": "mlx-lm",
    "mlx_lm": "mlx-lm",
    "llamacpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "whisper": "whisper.cpp",
    "whispercpp": "whisper.cpp",
    "whisper_cpp": "whisper.cpp",
    "sherpa": "sherpa-onnx",
    "sherpa_onnx": "sherpa-onnx",
    "sherpaonnx": "sherpa-onnx",
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
        store: RuntimePlannerStoreProtocol | None = None,
        client: RuntimePlannerClient | None = None,
    ) -> None:
        self._store: RuntimePlannerStoreProtocol = store or build_runtime_planner_store()
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
            Model identifier (e.g. "gemma-2b", "llama-8b") or an
            ``@app/{slug}/{capability}`` app ref.
        capability:
            The capability needed (e.g. "text", "embeddings", "audio").
        routing_policy:
            One of "local_first", "cloud_first", "local_only", "cloud_only",
            "private".
        allow_network:
            Whether to contact the server for plan/telemetry.  Set to False
            for fully-offline operation.
        """
        # Parse @app/{slug}/{capability} refs
        app_slug: str | None = None
        app_capability: str | None = None
        effective_model = model

        if is_app_ref(model):
            app_slug, app_capability = parse_app_ref(model)
            if app_capability:
                capability = app_capability

        # Step 1: Collect device profile
        device = collect_device_runtime_profile(model_id=effective_model)

        # Step 2: Check local plan cache
        from octomil import __version__

        cache_key = _make_cache_key(
            model=effective_model,
            capability=capability,
            policy=routing_policy,
            sdk_version=__version__,
            platform=device.platform,
            arch=device.arch,
        )

        cached_plan = self._store.get_plan(cache_key)
        is_private = routing_policy == "private"
        prefer_live_app_plan = is_app_ref(model) and allow_network and self._client is not None and not is_private

        if cached_plan is not None and not prefer_live_app_plan:
            logger.debug("Using cached plan for %s/%s", effective_model, capability)
            return self._selection_from_plan_dict(cached_plan, device=device, source="cache")

        # Step 3: Fetch server plan if network is allowed
        server_plan: RuntimePlanResponse | None = None

        if cached_plan is not None and prefer_live_app_plan:
            logger.debug(
                "Bypassing cached app plan for %s/%s to honor live app policy",
                effective_model,
                capability,
            )

        if allow_network and self._client is not None and not is_private:
            server_plan = self._client.fetch_plan(
                model=effective_model,
                capability=capability,
                routing_policy=routing_policy,
                device=device,
                allow_cloud_fallback=routing_policy != "local_only",
                app_slug=app_slug,
            )
            if server_plan is not None:
                logger.debug("Received server plan for %s/%s", effective_model, capability)

                resolved_model = _resolved_model_from_plan(server_plan)
                if resolved_model:
                    effective_model = resolved_model

                # Honour the app-level routing policy when the server resolved
                # through an app reference.
                if server_plan.app_resolution is not None:
                    ar = server_plan.app_resolution
                    if ar.routing_policy:
                        routing_policy = ar.routing_policy
                        is_private = routing_policy == "private"

                # Cache the server plan
                plan_dict = asdict(server_plan)
                self._store.put_plan(
                    cache_key,
                    model=effective_model,
                    capability=capability,
                    policy=routing_policy,
                    plan_json=json.dumps(plan_dict),
                    source="server_plan",
                    ttl_seconds=server_plan.plan_ttl_seconds,
                )
            elif cached_plan is not None:
                logger.debug(
                    "Server plan unavailable for %s/%s, falling back to cached app plan",
                    effective_model,
                    capability,
                )
                return self._selection_from_plan_dict(
                    cached_plan,
                    device=device,
                    source="cache",
                )

        # Step 4: Validate server plan against installed runtimes
        if server_plan is not None:
            selection = self._resolve_from_server_plan(server_plan, device=device, is_private=is_private)
            if selection is not None:
                selection.app_resolution = server_plan.app_resolution
                selection.resolution = server_plan.resolution
                return selection

        # Step 5-8: Fall back to local engine selection
        selection = self._resolve_locally(
            model=effective_model,
            capability=capability,
            routing_policy=routing_policy,
            device=device,
            is_private=is_private,
        )
        if server_plan is not None and server_plan.app_resolution is not None:
            selection.app_resolution = server_plan.app_resolution
        if server_plan is not None and server_plan.resolution is not None:
            selection.resolution = server_plan.resolution
        return selection

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
                    candidates=[*plan.candidates, *plan.fallback_candidates],
                    fallback_candidates=plan.fallback_candidates,
                    fallback_allowed=plan.fallback_allowed,
                    reason=candidate.reason,
                    app_resolution=plan.app_resolution,
                    resolution=plan.resolution,
                )
            elif candidate.locality == "cloud":
                return RuntimeSelection(
                    locality="cloud",
                    engine=candidate.engine,
                    artifact=candidate.artifact,
                    benchmark_ran=False,
                    source="server_plan",
                    candidates=[*plan.candidates, *plan.fallback_candidates],
                    fallback_candidates=plan.fallback_candidates,
                    fallback_allowed=plan.fallback_allowed,
                    reason=candidate.reason,
                    app_resolution=plan.app_resolution,
                    resolution=plan.resolution,
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
                    candidates=[*plan.candidates, *plan.fallback_candidates],
                    fallback_candidates=[],
                    fallback_allowed=plan.fallback_allowed,
                    reason=f"fallback: {candidate.reason}",
                    app_resolution=plan.app_resolution,
                    resolution=plan.resolution,
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
                fallback_allowed=False,
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
                fallback_allowed=routing_policy not in ("local_only", "private"),
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
                    fallback_allowed=routing_policy not in ("local_only", "private"),
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
                fallback_allowed=False,
                reason="no local engine available",
            )

        return RuntimeSelection(
            locality="cloud",
            engine=None,
            source="fallback",
            fallback_allowed=True,
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
        app_resolution = plan_dict_to_app_resolution(plan_dict.get("app_resolution"))
        resolution = plan_dict_to_model_resolution(plan_dict.get("resolution"))

        for c in candidates:
            if c.get("locality") == "local":
                engine = _canonical_engine_id(c.get("engine"))
                if engine and engine not in installed_engines:
                    continue
                return RuntimeSelection(
                    locality="local",
                    engine=engine or None,
                    candidates=plan_dict_to_candidates(candidates),
                    fallback_candidates=plan_dict_to_candidates(plan_dict.get("fallback_candidates", [])),
                    fallback_allowed=plan_dict.get("fallback_allowed", True),
                    source=source,
                    reason=c.get("reason", ""),
                    app_resolution=app_resolution,
                    resolution=resolution,
                )
            elif c.get("locality") == "cloud":
                return RuntimeSelection(
                    locality="cloud",
                    engine=c.get("engine"),
                    candidates=plan_dict_to_candidates(candidates),
                    fallback_candidates=plan_dict_to_candidates(plan_dict.get("fallback_candidates", [])),
                    fallback_allowed=plan_dict.get("fallback_allowed", True),
                    source=source,
                    reason=c.get("reason", ""),
                    app_resolution=app_resolution,
                    resolution=resolution,
                )

        # Nothing matched from cache — return generic fallback
        return RuntimeSelection(
            locality="local",
            engine=None,
            source="fallback",
            fallback_allowed=False,
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


def _resolved_model_from_plan(plan: RuntimePlanResponse) -> str | None:
    """Return the concrete model resolved by the server plan, if any."""
    if plan.app_resolution is not None and plan.app_resolution.selected_model:
        return plan.app_resolution.selected_model
    if plan.resolution is not None and plan.resolution.resolved_model:
        return plan.resolution.resolved_model
    return None


def plan_dict_to_app_resolution(data: dict | None) -> AppResolution | None:
    """Rehydrate a cached app_resolution dict."""
    if not isinstance(data, dict):
        return None

    artifact_candidates: list[RuntimeArtifactPlan] = []
    for artifact_data in data.get("artifact_candidates", []):
        if isinstance(artifact_data, dict):
            artifact_candidates.append(
                RuntimeArtifactPlan(
                    model_id=artifact_data.get("model_id", ""),
                    artifact_id=artifact_data.get("artifact_id"),
                    model_version=artifact_data.get("model_version"),
                    format=artifact_data.get("format"),
                    quantization=artifact_data.get("quantization"),
                    uri=artifact_data.get("uri"),
                    digest=artifact_data.get("digest"),
                    size_bytes=artifact_data.get("size_bytes"),
                    min_ram_bytes=artifact_data.get("min_ram_bytes"),
                )
            )

    return AppResolution(
        app_id=data.get("app_id", ""),
        capability=data.get("capability", ""),
        routing_policy=data.get("routing_policy", ""),
        selected_model=data.get("selected_model", ""),
        app_slug=data.get("app_slug"),
        selected_model_variant_id=data.get("selected_model_variant_id"),
        selected_model_version=data.get("selected_model_version"),
        artifact_candidates=artifact_candidates,
        preferred_engines=data.get("preferred_engines", []),
        fallback_policy=data.get("fallback_policy"),
        plan_ttl_seconds=data.get("plan_ttl_seconds", 604800),
    )


def plan_dict_to_model_resolution(data: dict | None) -> ModelResolution | None:
    """Rehydrate a cached generic model-resolution block."""
    if not isinstance(data, dict):
        return None

    return ModelResolution(
        ref_kind=data.get("ref_kind", ""),
        original_ref=data.get("original_ref", ""),
        resolved_model=data.get("resolved_model", ""),
        deployment_id=data.get("deployment_id"),
        deployment_key=data.get("deployment_key"),
        experiment_id=data.get("experiment_id"),
        variant_id=data.get("variant_id"),
        variant_name=data.get("variant_name"),
        capability=data.get("capability"),
        routing_policy=data.get("routing_policy"),
    )


def plan_dict_to_candidates(candidates: list[dict]) -> list[RuntimeCandidatePlan]:
    """Rehydrate cached candidate dictionaries into typed candidate plans."""
    parsed: list[RuntimeCandidatePlan] = []
    for candidate in candidates:
        artifact_data = candidate.get("artifact")
        artifact = None
        if isinstance(artifact_data, dict):
            artifact = RuntimeArtifactPlan(
                model_id=artifact_data.get("model_id", ""),
                artifact_id=artifact_data.get("artifact_id"),
                model_version=artifact_data.get("model_version"),
                format=artifact_data.get("format"),
                quantization=artifact_data.get("quantization"),
                uri=artifact_data.get("uri"),
                digest=artifact_data.get("digest"),
                size_bytes=artifact_data.get("size_bytes"),
                min_ram_bytes=artifact_data.get("min_ram_bytes"),
            )

        gates = []
        for gate in candidate.get("gates", []):
            if isinstance(gate, dict):
                gates.append(
                    CandidateGate(
                        code=gate.get("code", ""),
                        required=gate.get("required", True),
                        threshold_number=gate.get("threshold_number"),
                        threshold_string=gate.get("threshold_string"),
                        window_seconds=gate.get("window_seconds"),
                        source=gate.get("source", "server"),
                    )
                )

        parsed.append(
            RuntimeCandidatePlan(
                locality=candidate.get("locality", "local"),
                priority=candidate.get("priority", 0),
                confidence=candidate.get("confidence", 0.0),
                reason=candidate.get("reason", ""),
                engine=candidate.get("engine"),
                engine_version_constraint=candidate.get("engine_version_constraint"),
                artifact=artifact,
                benchmark_required=candidate.get("benchmark_required", False),
                gates=gates,
            )
        )
    return parsed


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
    return _make_cache_key(
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
