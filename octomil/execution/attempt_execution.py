"""Locality decision logic and policy resolution.

Extracted from kernel.py -- contains locality selection helpers and
policy conversion functions.
"""

from __future__ import annotations

from typing import Optional

from octomil._generated.routing_policy import RoutingPolicy as ContractRoutingPolicy
from octomil.config.local import (
    InlinePolicy,
    ResolvedExecutionDefaults,
)
from octomil.execution.cloud_dispatch import _cloud_api_key
from octomil.runtime.core.policy import RoutingPolicy
from octomil.runtime.core.router import LOCALITY_CLOUD, LOCALITY_ON_DEVICE

# ---------------------------------------------------------------------------
# Policy conversion
# ---------------------------------------------------------------------------


def _resolve_routing_policy(defaults: ResolvedExecutionDefaults) -> RoutingPolicy:
    """Convert resolved config defaults into a runtime RoutingPolicy."""
    preset = defaults.policy_preset
    inline = defaults.inline_policy

    if inline is not None:
        return _inline_to_routing_policy(inline)

    if preset is None or preset == "local_first":
        return RoutingPolicy.local_first()
    if preset in {"private", "local_only"}:
        return RoutingPolicy.local_only()
    if preset == "auto":
        return RoutingPolicy.auto()
    if preset == "cloud_only":
        return RoutingPolicy.cloud_only()
    if preset == "cloud_first":
        return RoutingPolicy(
            mode=ContractRoutingPolicy.AUTO,
            prefer_local=False,
            fallback="local",
        )
    if preset == "performance_first":
        return RoutingPolicy.auto(prefer_local=True)

    return RoutingPolicy.local_first()


def _inline_to_routing_policy(ip: InlinePolicy) -> RoutingPolicy:
    rm = ip.routing_mode
    rp = ip.routing_preference

    if rm == "local_only":
        return RoutingPolicy.local_only()
    if rm == "cloud_only":
        return RoutingPolicy.cloud_only()
    if rp == "local":
        fb = "cloud" if ip.fallback.allow_cloud_fallback else "none"
        return RoutingPolicy.local_first(fallback=fb)
    if rp == "cloud":
        return RoutingPolicy(
            mode=ContractRoutingPolicy.AUTO,
            prefer_local=False,
            fallback="local" if ip.fallback.allow_local_fallback else "none",
        )
    if rp == "performance":
        return RoutingPolicy.auto(prefer_local=True)
    return RoutingPolicy.auto()


def _cloud_available(defaults: ResolvedExecutionDefaults) -> bool:
    return bool(defaults.cloud_profile and _cloud_api_key(defaults.cloud_profile))


def _resolve_localities(
    routing_policy: RoutingPolicy,
    *,
    local_available: bool,
    cloud_available: bool,
) -> tuple[str, Optional[str]]:
    """Return (primary_locality, fallback_locality | None).

    This mirrors ``_select_locality_for_capability`` but returns the
    configured primary and fallback localities for callers that own backend
    lifecycle.  Disabled fallbacks are exact: if the preferred locality is
    unavailable and fallback is ``"none"``, this raises instead of silently
    switching execution locations.
    """
    if routing_policy.mode == ContractRoutingPolicy.LOCAL_ONLY:
        if not local_available:
            raise RuntimeError("Private/local-only policy requires a local backend, but none is available.")
        return LOCALITY_ON_DEVICE, None

    if routing_policy.mode == ContractRoutingPolicy.CLOUD_ONLY:
        if not cloud_available:
            raise RuntimeError("Cloud-only policy requires cloud credentials, but none are configured.")
        return LOCALITY_CLOUD, None

    # AUTO or LOCAL_FIRST -- determine prefer_local from policy
    prefer_local = routing_policy.prefer_local
    if routing_policy.mode == ContractRoutingPolicy.LOCAL_FIRST:
        prefer_local = True

    if prefer_local:
        if local_available:
            fallback = LOCALITY_CLOUD if routing_policy.fallback == "cloud" and cloud_available else None
            return LOCALITY_ON_DEVICE, fallback
        if routing_policy.fallback == "cloud" and cloud_available:
            return LOCALITY_CLOUD, None
        if cloud_available:
            raise RuntimeError("Local chat execution is required by policy, but cloud fallback is disabled.")
        raise RuntimeError("No local or cloud backend available for chat.")

    # cloud-first
    if cloud_available:
        fallback = LOCALITY_ON_DEVICE if routing_policy.fallback == "local" and local_available else None
        return LOCALITY_CLOUD, fallback
    if routing_policy.fallback == "local" and local_available:
        return LOCALITY_ON_DEVICE, None
    if local_available:
        raise RuntimeError("Cloud chat execution is required by policy, but local fallback is disabled.")
    raise RuntimeError("No local or cloud backend available for chat.")


def _select_locality_for_capability(
    routing_policy: RoutingPolicy,
    *,
    local_available: bool,
    cloud_available: bool,
    capability: str,
) -> tuple[str, bool]:
    """Select local/cloud for non-ModelRuntime capabilities using exact policy semantics."""
    if routing_policy.mode == ContractRoutingPolicy.LOCAL_ONLY:
        if local_available:
            return LOCALITY_ON_DEVICE, False
        raise RuntimeError(f"Local {capability} execution is required by policy, but no local runtime is available.")

    if routing_policy.mode == ContractRoutingPolicy.CLOUD_ONLY:
        if cloud_available:
            return LOCALITY_CLOUD, False
        raise RuntimeError(f"Cloud {capability} execution is required by policy, but cloud is not configured.")

    if routing_policy.mode == ContractRoutingPolicy.LOCAL_FIRST or routing_policy.prefer_local:
        if local_available:
            return LOCALITY_ON_DEVICE, False
        if routing_policy.fallback == "cloud" and cloud_available:
            return LOCALITY_CLOUD, True
        raise RuntimeError(f"No local {capability} runtime available and cloud fallback is not configured.")

    if cloud_available:
        return LOCALITY_CLOUD, False
    if routing_policy.fallback == "local" and local_available:
        return LOCALITY_ON_DEVICE, True
    raise RuntimeError(f"No cloud {capability} runtime available and local fallback is not configured.")
