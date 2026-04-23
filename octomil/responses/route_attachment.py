"""Route metadata attachment — locality determination and telemetry reporting.

Determines whether inference runs on-device or in the cloud, and reports
fallback events to the telemetry system.
"""

from __future__ import annotations

from typing import Any, Optional

from octomil.runtime.core.adapter import InferenceBackendAdapter
from octomil.runtime.core.cloud_runtime import CloudModelRuntime
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.policy import RoutingPolicy
from octomil.runtime.core.router import (
    LOCALITY_CLOUD,
    LOCALITY_ON_DEVICE,
    RouterModelRuntime,
)

# ---------------------------------------------------------------------------
# Locality determination
# ---------------------------------------------------------------------------


def determine_locality(
    runtime: ModelRuntime,
    model_id: str,
    routing_policy: Optional[RoutingPolicy] = None,
) -> tuple[str, bool]:
    """Return (locality, is_fallback) for a resolved runtime.

    locality: "on_device" | "cloud"
    is_fallback: True when RouterModelRuntime fell back from local to cloud.
    """
    if isinstance(runtime, RouterModelRuntime):
        try:
            return runtime.resolve_locality(routing_policy)
        except RuntimeError:
            return LOCALITY_CLOUD, False
    if isinstance(runtime, CloudModelRuntime):
        return LOCALITY_CLOUD, False
    if isinstance(runtime, InferenceBackendAdapter):
        return LOCALITY_ON_DEVICE, False
    return LOCALITY_ON_DEVICE, False


def _locality_for_candidate(candidate: dict[str, Any]) -> str:
    """Map candidate locality to internal constant."""
    if candidate.get("locality") == "cloud":
        return LOCALITY_CLOUD
    return LOCALITY_ON_DEVICE


# ---------------------------------------------------------------------------
# Telemetry reporting
# ---------------------------------------------------------------------------


def report_fallback_if_needed(
    telemetry: Optional[object],
    model_id: str,
    is_fallback: bool,
) -> None:
    """Report cloud fallback to telemetry if applicable.

    Never raises -- telemetry failures are silently swallowed.
    """
    if not is_fallback or telemetry is None:
        return
    try:
        telemetry.report_fallback_cloud(  # type: ignore[attr-defined]
            model_id=model_id,
            fallback_reason="local_unavailable",
        )
    except Exception:
        pass
