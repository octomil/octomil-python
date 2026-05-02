"""Planner resolution helpers.

Extracted from kernel.py -- contains planner call wrappers, candidate
conversion, and synthetic fallback detection.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
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
# Outer per-process planner-selection cache: REMOVED (reviewer P1).
# ---------------------------------------------------------------------------
#
# An earlier revision of this PR added a TTL'd dict here keyed only on
# ``(model, capability, policy_preset)``. That key shadowed the
# correctness-critical context the planner already keys on internally
# at ``runtime/planner/planner.py::resolve`` —
# ``sdk_version + platform + arch + chip + installed_runtimes_hash +
# api_base + org_id_hash + key_type + app_slug``. The narrow outer
# cache could serve org-A / staging / app-policy plans to org-B /
# prod / later-app-policy calls within the TTL, and bypassed the
# live-app-plan rule the planner enforces for app-ref dispatches.
#
# The planner already covers the "skip the round-trip on warm
# repeats" case via its own ``_store.get_plan(cache_key)`` lookup —
# at the right layer, with the right keys. The outer cache was pure
# complexity for no correctness-safe win. Dropped.
#
# If a future perf cut wants to skip the planner construction or the
# disk lookup entirely, that work belongs INSIDE the planner module
# where the auth/device/app context is already in scope. Don't
# reintroduce an outer cache here.
#
# ``release_planner_selection_cache`` is preserved as a no-op shim
# so existing callers (and the kernel's ``release_warmed_backends``
# cascade) keep importing successfully.


def release_planner_selection_cache() -> None:
    """No-op shim. The outer process cache was removed in the
    P1 fix; the planner's internal store cache is now the only
    correctness-aware caching layer. This entry point stays
    callable for the public ``release_warmed_backends`` cascade.

    Idempotent.
    """
    return None


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
    # client, and routing falls through to local. The planner's own
    # ``_store.get_plan(cache_key)`` already handles repeat calls
    # cheaply (and correctly, since it keys on the full auth/device/
    # app context); no outer caching here.
    try:
        return planner.resolve(
            model=model,
            capability=planner_cap,
            routing_policy=policy_preset,
        )
    except Exception:
        logger.debug("Planner selection failed for %s/%s", capability, model, exc_info=True)
        return None


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
