"""Planner resolution helpers.

Extracted from kernel.py -- contains planner call wrappers, candidate
conversion, and synthetic fallback detection.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from threading import RLock
from typing import Any, Optional

from octomil._generated.routing_policy import RoutingPolicy as ContractRoutingPolicy
from octomil.config.local import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_TRANSCRIPTION,
    CAPABILITY_TTS,
)
from octomil.runtime.core.policy import RoutingPolicy

logger = logging.getLogger(__name__)


_PLANNER_CAPABILITY_MAP = {
    CAPABILITY_CHAT: "responses",
    CAPABILITY_EMBEDDING: "embeddings",
    CAPABILITY_TRANSCRIPTION: "transcription",
    CAPABILITY_TTS: "tts",
}


_PLANNER_BOOTSTRAP_WARNED = False


# ---------------------------------------------------------------------------
# Per-process planner-selection cache (PE-driven cross-model perf — TTS, ASR,
# embeddings, chat all benefit from skipping the per-call planner round-trip
# when the same (model, capability, policy) tuple has already been resolved).
# ---------------------------------------------------------------------------
#
# The planner is deterministic in (model, capability, policy_preset) within a
# single process, so re-running ``planner.resolve`` on every dispatch is pure
# waste once warmup has already cached the same selection. Eternum's 4.15.1
# warm setup_ms residual (~600 ms after the warmup-cache fix) is dominated by
# this re-resolution; caching brings warm setup_ms into the tens of ms.
#
# The cache is keyed by ``(model, capability, policy_preset)`` and holds the
# *raw* planner selection (or None for negative results — caching the miss is
# what avoids re-paying the failure-path cost on every dispatch when the
# planner is offline). TTL defaults to 30 s so a freshly-deployed planner
# config eventually propagates without an explicit ``release`` call.
#
# Invalidation:
#   - ``release_planner_selection_cache()`` — full clear (called from
#     ``ExecutionKernel.release_warmed_backends`` so the existing public
#     "drop my caches" surface also clears this).
#   - TTL expiry — soft refresh; matches the existing
#     ``OCTOMIL_RUNTIME_PLANNER_CACHE`` env-var posture.
#   - Bypass via ``OCTOMIL_RUNTIME_PLANNER_CACHE=0`` — already supported by
#     the underlying ``_resolve_planner_selection``; cache is also bypassed
#     when this is set so the env switch keeps both layers in sync.
#   - ``OCTOMIL_PLANNER_SELECTION_CACHE_TTL_SECONDS`` — override the default
#     30 s TTL. ``0`` disables caching but still calls the planner.

_PLANNER_SELECTION_CACHE: dict[tuple[str, str, str], tuple[float, Any]] = {}
_PLANNER_SELECTION_CACHE_LOCK = RLock()


def _planner_selection_cache_ttl_seconds() -> float:
    raw = os.environ.get("OCTOMIL_PLANNER_SELECTION_CACHE_TTL_SECONDS")
    if raw is None:
        return 30.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 30.0


def release_planner_selection_cache() -> None:
    """Drop every cached planner selection.

    Idempotent. The kernel's :meth:`release_warmed_backends` calls this so
    a single public "drop my caches" entry point clears both the warmup
    cache and the planner cache.
    """
    with _PLANNER_SELECTION_CACHE_LOCK:
        _PLANNER_SELECTION_CACHE.clear()


def _resolve_planner_selection(
    model: str,
    capability: str,
    policy_preset: str,
) -> Optional[Any]:
    """Try planner-based runtime selection. Returns RuntimeSelection or None.

    Never raises -- planner failure is non-fatal.

    Logging
    ~~~~~~~
    There are two failure modes the SDK should log differently:

      - **Bootstrap / import failures** (planner module won't load,
        cache backend won't initialize, auth misconfig). These are
        actionable: the operator needs to install an extra, pin a
        cache dir, or set ``OCTOMIL_SERVER_KEY``. We log at
        WARNING **once per process** so the message surfaces without
        flooding logs. (Repeated warnings on every call would be
        noise — the first one is enough.)
      - **HTTP planner misses** (network unavailable, 404 for an
        unknown model id, auth 401, transient 5xx). The SDK can
        still fall back to local routing, so these stay at DEBUG to
        keep production logs clean. The HTTP client handles its own
        per-request error reporting.

    The two are distinguished by the exception type / origin: import
    or bootstrap errors fire before ``planner.resolve`` is called.
    """
    if os.environ.get("OCTOMIL_RUNTIME_PLANNER_CACHE") == "0":
        return None

    # Per-process selection cache. Skipping the planner re-resolution
    # is the biggest single cut to warm dispatch ``setup_ms`` for any
    # capability — same selection shape regardless of TTS / ASR /
    # embeddings / chat. TTL=0 disables the cache entirely (still
    # calls the planner; useful for tests / config-change scenarios).
    cache_key = (model, capability, policy_preset)
    ttl = _planner_selection_cache_ttl_seconds()
    if ttl > 0.0:
        with _PLANNER_SELECTION_CACHE_LOCK:
            entry = _PLANNER_SELECTION_CACHE.get(cache_key)
        if entry is not None:
            stored_at, cached = entry
            if time.monotonic() - stored_at <= ttl:
                return cached

    # Bootstrap phase: import the planner module. ImportError /
    # AttributeError / cache-backend errors here are
    # configuration-actionable.
    try:
        from octomil.runtime.planner.planner import RuntimePlanner
    except Exception:
        global _PLANNER_BOOTSTRAP_WARNED
        if not _PLANNER_BOOTSTRAP_WARNED:
            logger.warning(
                "runtime planner unavailable: failed to import or initialize the planner. "
                "Local routing will be used; cloud-only models will fail with "
                "RUNTIME_UNAVAILABLE. Install [planner] extras or check logs.",
                exc_info=True,
            )
            _PLANNER_BOOTSTRAP_WARNED = True
        return None

    try:
        planner_cap = _PLANNER_CAPABILITY_MAP.get(capability, capability)
        planner = RuntimePlanner()
    except Exception:
        if not _PLANNER_BOOTSTRAP_WARNED:
            logger.warning(
                "runtime planner unavailable: planner instance failed to construct. Local routing will be used.",
                exc_info=True,
            )
            _PLANNER_BOOTSTRAP_WARNED = True
        return None

    # Resolve phase: HTTP misses, auth failures, transient 5xx. Stay
    # at DEBUG — the per-request error already surfaces through the
    # client, and routing falls through to local.
    try:
        selection = planner.resolve(
            model=model,
            capability=planner_cap,
            routing_policy=policy_preset,
        )
    except Exception:
        logger.debug("Planner selection failed for %s/%s", capability, model, exc_info=True)
        selection = None

    # Cache the result — including ``None`` (negative caching). When
    # the planner is offline / unreachable the failure-path cost is
    # what hurts every subsequent dispatch; remembering "we tried,
    # got nothing" within the TTL is the whole point.
    if ttl > 0.0:
        with _PLANNER_SELECTION_CACHE_LOCK:
            _PLANNER_SELECTION_CACHE[cache_key] = (time.monotonic(), selection)
    return selection


def _runtime_candidate_to_dict(candidate: Any) -> dict[str, Any]:
    """Convert a RuntimeCandidatePlan-like object into a JSON-safe dict."""
    if isinstance(candidate, dict):
        return candidate
    return asdict(candidate)


def _is_synthetic_cloud_fallback(selection: Any) -> bool:
    """Return true for offline planner "no local engine" cloud fallbacks.

    This is not a binding server-side route. It should defer to the normal
    policy candidate list so injected/test runtimes, registry runtimes, and
    policy-gated cloud fallback are still evaluated by the request path.
    """
    return (
        getattr(selection, "source", None) == "fallback"
        and getattr(selection, "locality", None) == "cloud"
        and not getattr(selection, "engine", None)
        and not getattr(selection, "candidates", None)
    )


def _runtime_model_for_selection(selection: Any, requested_model: str) -> str:
    """Return the concrete model a planner selection resolved to.

    Prefer app-specific resolution when present, then fall back to the generic
    model-resolution block used for aliases, defaults, deployments, and
    experiments. Only fall back to the original caller-provided ref when the
    planner did not resolve a concrete model.
    """
    if selection is None:
        return requested_model

    app_resolution = getattr(selection, "app_resolution", None)
    selected_model = getattr(app_resolution, "selected_model", None)
    if selected_model:
        return str(selected_model)

    resolution = getattr(selection, "resolution", None)
    resolved_model = getattr(resolution, "resolved_model", None)
    if resolved_model:
        return str(resolved_model)

    return requested_model


def _selection_candidate_dicts(selection: Optional[Any], routing_policy: RoutingPolicy) -> list[dict[str, Any]]:
    """Return the ordered candidate list the SDK should attempt for this request."""
    if selection is not None and _is_synthetic_cloud_fallback(selection):
        selection = None

    if selection is not None:
        candidates = getattr(selection, "candidates", None)
        if candidates:
            return [_runtime_candidate_to_dict(candidate) for candidate in candidates]

        candidate: dict[str, Any] = {
            "locality": getattr(selection, "locality", "local"),
            "priority": 0,
            "confidence": 1.0,
            "reason": getattr(selection, "reason", "") or "planner selection",
        }
        engine = getattr(selection, "engine", None)
        artifact = getattr(selection, "artifact", None)
        if engine is not None:
            candidate["engine"] = engine
        if artifact is not None:
            candidate["artifact"] = asdict(artifact)
        return [candidate]

    if routing_policy.mode == ContractRoutingPolicy.LOCAL_ONLY:
        return [{"locality": "local", "priority": 0, "confidence": 0.0, "reason": "local-only policy"}]
    if routing_policy.mode == ContractRoutingPolicy.CLOUD_ONLY:
        return [{"locality": "cloud", "priority": 0, "confidence": 0.0, "reason": "cloud-only policy"}]

    prefer_local = routing_policy.prefer_local or routing_policy.mode == ContractRoutingPolicy.LOCAL_FIRST
    if prefer_local:
        candidates = [{"locality": "local", "priority": 0, "confidence": 0.0, "reason": "offline local-first"}]
        if routing_policy.fallback == "cloud":
            candidates.append({"locality": "cloud", "priority": 1, "confidence": 0.0, "reason": "cloud fallback"})
        return candidates

    candidates = [{"locality": "cloud", "priority": 0, "confidence": 0.0, "reason": "offline cloud-first"}]
    if routing_policy.fallback == "local":
        candidates.append({"locality": "local", "priority": 1, "confidence": 0.0, "reason": "local fallback"})
    return candidates


def _candidate_fallback_allowed(selection: Optional[Any], routing_policy: RoutingPolicy) -> bool:
    if selection is not None and hasattr(selection, "fallback_allowed"):
        return bool(selection.fallback_allowed)
    if routing_policy.mode in (ContractRoutingPolicy.LOCAL_ONLY, ContractRoutingPolicy.CLOUD_ONLY):
        return False
    return routing_policy.fallback != "none"


def _candidate_to_selection(selection: Optional[Any], candidate: dict[str, Any]) -> Any:
    """Build a lightweight planner selection for a single candidate."""
    from octomil.runtime.planner.schemas import RuntimeArtifactPlan, RuntimeSelection

    artifact_data = candidate.get("artifact")
    artifact = None
    if isinstance(artifact_data, dict) and artifact_data.get("model_id"):
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

    return RuntimeSelection(
        locality=candidate.get("locality", "local"),
        engine=candidate.get("engine"),
        artifact=artifact,
        source=getattr(selection, "source", "fallback") if selection is not None else "fallback",
        fallback_allowed=getattr(selection, "fallback_allowed", True) if selection is not None else True,
        reason=candidate.get("reason", getattr(selection, "reason", "") if selection is not None else ""),
        app_resolution=getattr(selection, "app_resolution", None) if selection is not None else None,
        resolution=getattr(selection, "resolution", None) if selection is not None else None,
    )


def _routing_policy_for_candidate(candidate: dict[str, Any]) -> RoutingPolicy:
    if candidate.get("locality") == "cloud":
        return RoutingPolicy.cloud_only()
    return RoutingPolicy.local_only()
