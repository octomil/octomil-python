"""Route metadata types and builder.

Extracted from kernel.py -- contains all RouteMetadata dataclasses and the
_route_metadata_from_selection builder function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from octomil.runtime.core.router import LOCALITY_ON_DEVICE
from octomil.runtime.planner.schemas import normalize_planner_source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Route metadata types
# ---------------------------------------------------------------------------


@dataclass
class RouteExecution:
    """Execution details within route metadata."""

    locality: str = ""  # "local" | "cloud"
    mode: str = ""  # "sdk_runtime" | "hosted_gateway" | "external_endpoint"
    engine: Optional[str] = None


@dataclass
class RouteModelRequested:
    """The model reference as requested by the caller."""

    ref: str = ""
    kind: str = "unknown"  # "model" | "app" | "deployment" | "alias" | "default" | "unknown"
    capability: Optional[str] = None


@dataclass
class RouteModelResolved:
    """Server-resolved model identifiers."""

    id: Optional[str] = None
    slug: Optional[str] = None
    version_id: Optional[str] = None
    variant_id: Optional[str] = None


@dataclass
class RouteModel:
    """Model information within route metadata."""

    requested: RouteModelRequested = field(default_factory=RouteModelRequested)
    resolved: Optional[RouteModelResolved] = None


@dataclass
class ArtifactCache:
    """Cache status for a model artifact."""

    status: str = "not_applicable"  # "hit" | "miss" | "downloaded" | "not_applicable" | "unavailable"
    managed_by: Optional[str] = None  # "octomil" | "runtime" | "external"


@dataclass
class RouteArtifact:
    """Artifact information within route metadata."""

    id: Optional[str] = None
    version: Optional[str] = None
    format: Optional[str] = None
    digest: Optional[str] = None
    cache: ArtifactCache = field(default_factory=ArtifactCache)


@dataclass
class PlannerInfo:
    """Planner source information."""

    source: str = "offline"  # "server" | "cache" | "offline"


@dataclass
class FallbackInfo:
    """Fallback status information."""

    used: bool = False
    from_attempt: Optional[int] = None
    to_attempt: Optional[int] = None
    trigger: Optional[dict[str, Any]] = None


@dataclass
class RouteReason:
    """Reason for the routing decision."""

    code: str = ""
    message: str = ""


@dataclass
class RouteMetadata:
    """Contract-backed routing metadata from the runtime planner.

    Follows the canonical RouteMetadata shape from octomil-contracts.
    Public locality values are "local" | "cloud" (never "on_device").
    """

    status: str = "selected"  # "selected" | "unavailable" | "failed"
    execution: Optional[RouteExecution] = None
    model: RouteModel = field(default_factory=RouteModel)
    artifact: Optional[RouteArtifact] = None
    planner: PlannerInfo = field(default_factory=PlannerInfo)
    fallback: FallbackInfo = field(default_factory=FallbackInfo)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    reason: RouteReason = field(default_factory=RouteReason)


def _locality_to_public(raw: str) -> str:
    """Map internal locality values to the public contract values.

    Public RouteMetadata uses "local" | "cloud". Internal code may use
    "on_device" which must never appear in public route metadata.
    """
    if raw == LOCALITY_ON_DEVICE or raw == "on_device":
        return "local"
    return raw


def _execution_mode_for_locality(public_locality: str) -> str:
    """Determine execution.mode from the public locality."""
    if public_locality == "local":
        return "sdk_runtime"
    return "hosted_gateway"


def _route_metadata_from_selection(
    selection: Optional[Any],
    locality: str,
    fallback_used: bool,
    *,
    model_name: str = "",
    capability: str = "",
    attempt_loop: Optional[Any] = None,
    status: str = "selected",
) -> RouteMetadata:
    """Build RouteMetadata from a planner RuntimeSelection."""
    from octomil.runtime.routing.model_ref import parse_model_ref

    public_locality = _locality_to_public(locality)

    # Determine model.requested.kind and capability from model_name
    parsed_ref = parse_model_ref(model_name)
    model_kind = parsed_ref.kind
    req_capability = parsed_ref.capability or capability or None

    fallback_info = FallbackInfo(used=fallback_used)
    attempts: list[dict[str, Any]] = []
    if attempt_loop is not None:
        attempts = [attempt.to_dict() for attempt in getattr(attempt_loop, "attempts", [])]
        trigger = getattr(attempt_loop, "fallback_trigger", None)
        fallback_info = FallbackInfo(
            used=bool(getattr(attempt_loop, "fallback_used", False)),
            from_attempt=getattr(attempt_loop, "from_attempt", None),
            to_attempt=getattr(attempt_loop, "to_attempt", None),
            trigger=trigger.to_dict() if trigger is not None else None,
        )

    if selection is None:
        return RouteMetadata(
            status=status,
            execution=RouteExecution(
                locality=public_locality,
                mode=_execution_mode_for_locality(public_locality),
            ),
            model=RouteModel(
                requested=RouteModelRequested(
                    ref=model_name,
                    kind=model_kind,
                    capability=req_capability,
                ),
            ),
            planner=PlannerInfo(source="offline"),
            fallback=fallback_info,
            attempts=attempts,
            reason=RouteReason(code="planner_unavailable", message="planner not available"),
        )

    # Build resolved model info from app_resolution when available
    resolved: Optional[RouteModelResolved] = None
    app_resolution = getattr(selection, "app_resolution", None)
    if app_resolution is not None:
        resolved = RouteModelResolved(
            slug=app_resolution.selected_model,
            variant_id=app_resolution.selected_model_variant_id,
            version_id=app_resolution.selected_model_version,
        )

    selected_attempt = getattr(attempt_loop, "selected_attempt", None) if attempt_loop is not None else None

    # Build artifact info with cache status from ArtifactCache
    route_artifact: Optional[RouteArtifact] = None
    if selection.artifact is not None:
        artifact_digest = selection.artifact.digest
        cache_status_value = "not_applicable"
        try:
            from octomil.runtime.planner.artifact_cache import ArtifactCache as ArtifactCacheManager
            from octomil.runtime.planner.artifact_cache import _warn_if_large_artifact_non_tty

            artifact_cache = ArtifactCacheManager()
            cache_status_value = artifact_cache.cache_status(artifact_digest)

            # Warn in non-TTY environments about large artifacts
            if cache_status_value == "miss":
                _warn_if_large_artifact_non_tty(selection.artifact.size_bytes)
        except Exception:
            pass
        route_artifact = RouteArtifact(
            id=selection.artifact.artifact_id,
            version=selection.artifact.model_version,
            format=selection.artifact.format,
            digest=artifact_digest,
            cache=ArtifactCache(
                status=cache_status_value,
                managed_by="octomil",
            ),
        )
    if selected_attempt is not None:
        if selected_attempt.artifact is None:
            route_artifact = None
        else:
            route_artifact = RouteArtifact(
                id=selected_attempt.artifact.id,
                digest=selected_attempt.artifact.digest,
                cache=ArtifactCache(
                    status=selected_attempt.artifact.cache_status,
                    managed_by=selected_attempt.artifact.managed_by,
                ),
            )

    route_engine = selected_attempt.engine if selected_attempt is not None else selection.engine

    return RouteMetadata(
        status=status,
        execution=RouteExecution(
            locality=public_locality,
            mode=_execution_mode_for_locality(public_locality),
            engine=route_engine,
        ),
        model=RouteModel(
            requested=RouteModelRequested(
                ref=model_name,
                kind=model_kind,
                capability=req_capability,
            ),
            resolved=resolved,
        ),
        artifact=route_artifact,
        planner=PlannerInfo(source=normalize_planner_source(selection.source)),
        fallback=fallback_info,
        attempts=attempts,
        reason=RouteReason(
            code=selection.source or "",
            message=selection.reason or "",
        ),
    )
