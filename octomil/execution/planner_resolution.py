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
)
from octomil.runtime.core.policy import RoutingPolicy

logger = logging.getLogger(__name__)


_PLANNER_CAPABILITY_MAP = {
    CAPABILITY_CHAT: "responses",
    CAPABILITY_EMBEDDING: "embeddings",
    CAPABILITY_TRANSCRIPTION: "transcription",
}


def _resolve_planner_selection(
    model: str,
    capability: str,
    policy_preset: str,
) -> Optional[Any]:
    """Try planner-based runtime selection. Returns RuntimeSelection or None.

    Never raises -- planner failure is non-fatal.
    """
    if os.environ.get("OCTOMIL_RUNTIME_PLANNER_CACHE") == "0":
        return None
    try:
        from octomil.runtime.planner.planner import RuntimePlanner

        planner_cap = _PLANNER_CAPABILITY_MAP.get(capability, capability)
        planner = RuntimePlanner()
        return planner.resolve(
            model=model,
            capability=planner_cap,
            routing_policy=policy_preset,
        )
    except Exception:
        logger.debug("Planner selection failed", exc_info=True)
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
    )


def _routing_policy_for_candidate(candidate: dict[str, Any]) -> RoutingPolicy:
    if candidate.get("locality") == "cloud":
        return RoutingPolicy.cloud_only()
    return RoutingPolicy.local_only()
