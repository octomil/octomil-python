"""Backend detection, startup error logging, and cache manager helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..errors import OctomilError, OctomilErrorCode
from .backends.echo import EchoBackend
from .backends.mlx import MLXBackend
from .instrumentation import InstrumentedBackend, unwrap_backend
from .types import InferenceBackend

logger = logging.getLogger(__name__)


def _detect_backend(
    model_name: str,
    *,
    cache_size_mb: int = 2048,
    cache_enabled: bool = True,
    engine_override: Optional[str] = None,
    verbose_emitter: Any = None,
) -> InferenceBackend:
    """Auto-detect engines, benchmark each, and return the fastest backend.

    Uses the engine registry plugin system. Each registered engine is:
    1. Detected (is the library installed? does it support this model?)
    2. Benchmarked (quick 32-token generation to measure tok/s)
    3. Ranked (highest tok/s wins)

    If engine_override is set, skip benchmarking and use that engine directly.
    """
    from ..runtime.engines import get_registry

    registry = get_registry()

    backend_kwargs: dict[str, Any] = {
        "cache_size_mb": cache_size_mb,
        "cache_enabled": cache_enabled,
    }

    def _maybe_wrap(backend: InferenceBackend) -> InferenceBackend:
        if verbose_emitter is not None:
            return InstrumentedBackend(backend, verbose_emitter)
        return backend

    if engine_override:
        engine = registry.get_engine(engine_override)
        if engine is None:
            available = [e.name for e in registry.engines]
            raise OctomilError(
                code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
                message=f"Unknown engine '{engine_override}'. Available: {', '.join(available)}",
            )
        backend: InferenceBackend = engine.create_backend(model_name, **backend_kwargs)
        return _maybe_wrap(backend)

    # Detect all available engines for this model
    detections = registry.detect_all(model_name)
    available_engines = [d.engine for d in detections if d.available]

    for d in detections:
        if d.available:
            logger.info("Engine detected: %s (%s)", d.engine.name, d.info)
        else:
            logger.debug("Engine unavailable: %s", d.engine.name)

    # No real engines -> echo fallback
    real_engines = [e for e in available_engines if e.name != "echo"]
    if not real_engines:
        echo = EchoBackend()
        echo.load_model(model_name)
        return _maybe_wrap(echo)

    # Benchmark real engines and pick fastest
    ranked = registry.benchmark_all(model_name, n_tokens=32, engines=real_engines)
    best = registry.select_best(ranked)
    if best is None:
        echo = EchoBackend()
        echo.load_model(model_name)
        return _maybe_wrap(echo)

    best_backend: InferenceBackend = best.engine.create_backend(model_name, **backend_kwargs)
    return _maybe_wrap(best_backend)


def _log_startup_error(model_name: str, exc: Exception) -> None:
    """Print a human-readable startup error instead of a raw traceback."""
    err_type = type(exc).__name__
    err_msg = str(exc)

    # HuggingFace auth / repo errors
    if "RepositoryNotFoundError" in err_type or "401" in err_msg or "403" in err_msg:
        logger.error(
            "Failed to load model '%s': HuggingFace authentication required.\n"
            "  Fix: Run `huggingface-cli login` or set the HF_TOKEN env var.\n"
            "  Get a token at https://huggingface.co/settings/tokens",
            model_name,
        )
    elif "404" in err_msg or "not found" in err_msg.lower():
        logger.error(
            "Failed to load model '%s': model not found on HuggingFace.\n"
            "  Check the model name and try a full repo ID:\n"
            "    octomil serve REDACTED\n"
            "  List available short names:\n"
            "    octomil serve --help",
            model_name,
        )
    elif "Unknown model" in err_msg:
        logger.error("Failed to load model: %s", err_msg)
    else:
        logger.error(
            "Failed to start server for model '%s': %s\n  If this is a HuggingFace model, try: huggingface-cli login",
            model_name,
            exc,
        )


def _get_cache_manager(backend: InferenceBackend) -> Any:
    """Extract the KVCacheManager from a backend, if available."""
    inner = unwrap_backend(backend)
    if isinstance(inner, MLXBackend):
        return inner.kv_cache
    return None
