"""Bridge EngineRegistry into ModelRuntimeRegistry as the default factory."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

from octomil.runtime.core.adapter import InferenceBackendAdapter
from octomil.runtime.core.base import EnginePlugin
from octomil.runtime.core.model_runtime import ModelRuntime, RuntimeFactory
from octomil.runtime.core.types import RuntimeCapabilities, ToolCallTier

_runtime_cache: dict[str, ModelRuntime] = {}
_SELECTION_CACHE_FILENAME = "runtime_engine_selection.json"
_SELECTION_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
_DEFAULT_BENCHMARK_TOKENS = 16

# Models known to follow the tool-call JSON protocol reliably.
# Extend this list as models are evaluated.
_TEXT_JSON_FAMILIES: frozenset[str] = frozenset(
    {
        "gemma",
        "llama-3",
        "qwen2",
        "phi-3",
        "mistral",
    }
)


def _infer_tool_call_tier(model_id: str) -> ToolCallTier:
    """Infer tool-call tier from model family. Defaults to NONE."""
    model_lower = model_id.lower()
    for family in _TEXT_JSON_FAMILIES:
        if family in model_lower:
            return ToolCallTier.TEXT_JSON
    return ToolCallTier.NONE


def _cache_path() -> Path:
    cache_root = Path(os.environ.get("OCTOMIL_CACHE_DIR", Path.home() / ".cache" / "octomil"))
    return cache_root / _SELECTION_CACHE_FILENAME


def _selection_cache_enabled() -> bool:
    return os.environ.get("OCTOMIL_RUNTIME_SELECTION_CACHE", "1").lower() not in {"0", "false", "no"}


def _benchmark_tokens() -> int:
    raw = os.environ.get("OCTOMIL_RUNTIME_BENCHMARK_TOKENS")
    if raw is None:
        return _DEFAULT_BENCHMARK_TOKENS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_BENCHMARK_TOKENS


def _selection_cache_key(model_id: str, engine_names: list[str]) -> str:
    from octomil import __version__

    fingerprint = "|".join(
        [
            model_id,
            platform.system(),
            platform.machine(),
            f"{sys.version_info.major}.{sys.version_info.minor}",
            __version__,
            ",".join(sorted(engine_names)),
        ]
    )
    return fingerprint


def _load_selection_cache() -> dict[str, dict[str, object]]:
    if not _selection_cache_enabled():
        return {}
    try:
        raw = _cache_path().read_text()
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(key): value for key, value in data.items() if isinstance(value, dict)}
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return {}


def _write_selection_cache(cache: dict[str, dict[str, object]]) -> None:
    if not _selection_cache_enabled():
        return
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(path)
    except OSError:
        pass


def _get_cached_engine(model_id: str, real_engines: list[EnginePlugin]) -> Optional[EnginePlugin]:
    engine_by_name = {engine.name: engine for engine in real_engines}
    key = _selection_cache_key(model_id, list(engine_by_name))
    entry = _load_selection_cache().get(key)
    if not entry:
        return None

    selected_at = entry.get("selected_at")
    engine_name = entry.get("engine")
    if not isinstance(selected_at, (int, float)) or not isinstance(engine_name, str):
        return None
    if time.time() - selected_at > _SELECTION_CACHE_TTL_SECONDS:
        return None
    return engine_by_name.get(engine_name)


def _record_cached_engine(
    model_id: str,
    real_engines: list[EnginePlugin],
    engine: EnginePlugin,
    tokens_per_second: float,
) -> None:
    engine_names = [candidate.name for candidate in real_engines]
    key = _selection_cache_key(model_id, engine_names)
    cache = _load_selection_cache()
    cache[key] = {
        "engine": engine.name,
        "selected_at": time.time(),
        "tokens_per_second": tokens_per_second,
    }
    _write_selection_cache(cache)


def _evict_cached_engine(model_id: str, real_engines: list[EnginePlugin]) -> None:
    engine_names = [candidate.name for candidate in real_engines]
    key = _selection_cache_key(model_id, engine_names)
    cache = _load_selection_cache()
    if key in cache:
        del cache[key]
        _write_selection_cache(cache)


def _select_real_engine(registry: Any, model_id: str) -> tuple[Optional[EnginePlugin], list[EnginePlugin]]:
    detections = registry.detect_all(model_id)
    real_engines = [d.engine for d in detections if d.available and d.engine.name != "echo"]
    if not real_engines:
        return None, []

    cached = _get_cached_engine(model_id, real_engines)
    if cached is not None:
        return cached, real_engines

    ranked = registry.benchmark_all(
        model_id,
        n_tokens=_benchmark_tokens(),
        engines=real_engines,
    )
    for ranked_engine in ranked:
        if ranked_engine.engine.name == "echo" or not ranked_engine.result.ok:
            continue
        _record_cached_engine(
            model_id,
            real_engines,
            ranked_engine.engine,
            ranked_engine.result.tokens_per_second,
        )
        return ranked_engine.engine, real_engines

    return None, real_engines


def _reject_echo_only(registry: Any, model_id: str) -> None:
    """Raise if echo is the ONLY available engine.

    The echo engine must never silently serve user-facing requests.  If no
    real engine is installed the caller should get a clear error rather than
    a fake response.
    """
    detections = registry.detect_all(model_id)
    available = [d.engine for d in detections if d.available]
    real = [e for e in available if e.name != "echo"]
    if available and not real:
        raise RuntimeError(
            "No real inference engine is available (only the echo testing stub is installed).\n\n"
            "Install a local runtime for on-device execution:\n"
            "  pip install 'octomil[mlx]'      # Apple Silicon\n"
            "  pip install 'octomil[llama]'    # Cross-platform\n"
            "Or set OCTOMIL_SERVER_KEY to allow hosted cloud fallback."
        )


def engine_registry_factory(model_id: str) -> Optional[ModelRuntime]:
    """RuntimeFactory that benchmark-selects the fastest real local engine."""
    if model_id in _runtime_cache:
        return _runtime_cache[model_id]

    try:
        from octomil.runtime.engines import get_registry

        registry = get_registry()

        # Hard gate: never silently use echo for user-facing inference.
        _reject_echo_only(registry, model_id)

        planner_selection = _select_engine_with_planner(model_id, "responses", "local_first")
        if planner_selection is not None and planner_selection.locality == "local" and planner_selection.engine:
            engine = registry.get_engine(planner_selection.engine)
            if engine is not None and engine.detect() and engine.name != "echo":
                try:
                    backend = engine.create_backend(model_id)
                except (ValueError, RuntimeError, ImportError):
                    backend = None
                if backend is not None:
                    adapter = InferenceBackendAdapter(
                        backend=backend,
                        model_name=model_id,
                        capabilities=RuntimeCapabilities(
                            tool_call_tier=_infer_tool_call_tier(model_id),
                            supports_streaming=True,
                        ),
                    )
                    _runtime_cache[model_id] = adapter
                    return adapter

        engine, real_engines = _select_real_engine(registry, model_id)
        if engine is None:
            return None

        try:
            backend = engine.create_backend(model_id)
        except (ValueError, RuntimeError, ImportError):
            _evict_cached_engine(model_id, real_engines)
            return None

        adapter = InferenceBackendAdapter(
            backend=backend,
            model_name=model_id,
            capabilities=RuntimeCapabilities(
                tool_call_tier=_infer_tool_call_tier(model_id),
                supports_streaming=True,
            ),
        )
        _runtime_cache[model_id] = adapter
        return adapter
    except (ValueError, RuntimeError, ImportError):
        return None


def _select_engine_with_planner(model: str, capability: str, policy: str) -> Any:
    """Try planner-based selection. Returns None if planner is disabled or fails.

    The planner combines server-side plan recommendations with local benchmark
    caching in a SQLite store.  Opt-out via OCTOMIL_RUNTIME_PLANNER_CACHE=0.
    """
    if os.environ.get("OCTOMIL_RUNTIME_PLANNER_CACHE") == "0":
        return None
    try:
        import logging

        from octomil.runtime.planner.planner import RuntimePlanner

        planner = RuntimePlanner()
        return planner.resolve(model=model, capability=capability, routing_policy=policy)
    except Exception:
        logging.getLogger(__name__).debug("Planner selection failed, falling back to legacy", exc_info=True)
        return None


def cloud_runtime_factory(base_url: str, api_key: str, model: str) -> RuntimeFactory:
    """RuntimeFactory that creates CloudModelRuntime for cloud inference."""
    _cached: Optional[ModelRuntime] = None

    def factory(model_id: str) -> Optional[ModelRuntime]:
        nonlocal _cached
        if _cached is None:
            from octomil.runtime.core.cloud_runtime import CloudModelRuntime

            _cached = CloudModelRuntime(base_url, api_key, model)
        return _cached

    return factory
