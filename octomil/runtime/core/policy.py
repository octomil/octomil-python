"""Routing policy for Layer 4 runtime selection.

Wraps the contract-level RoutingPolicy enum with additional runtime
hints (latency thresholds, fallback preferences).  The ``mode`` field
is always a ``ContractRoutingPolicy`` enum value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from octomil._generated.routing_policy import RoutingPolicy as ContractRoutingPolicy


@dataclass
class RoutingPolicy:
    """Runtime routing policy binding the contract enum with SDK hints."""

    mode: ContractRoutingPolicy
    prefer_local: bool = True
    max_latency_ms: Optional[int] = None
    fallback: str = "cloud"

    @classmethod
    def auto(
        cls,
        prefer_local: bool = True,
        max_latency_ms: Optional[int] = None,
        fallback: Optional[str] = None,
    ) -> RoutingPolicy:
        fallback_target = fallback if fallback is not None else ("cloud" if prefer_local else "local")
        return cls(
            mode=ContractRoutingPolicy.AUTO,
            prefer_local=prefer_local,
            max_latency_ms=max_latency_ms,
            fallback=fallback_target,
        )

    @classmethod
    def local_only(cls) -> RoutingPolicy:
        return cls(mode=ContractRoutingPolicy.LOCAL_ONLY, fallback="none")

    @classmethod
    def local_first(
        cls,
        max_latency_ms: Optional[int] = None,
        fallback: str = "cloud",
    ) -> RoutingPolicy:
        return cls(
            mode=ContractRoutingPolicy.LOCAL_FIRST,
            prefer_local=True,
            max_latency_ms=max_latency_ms,
            fallback=fallback,
        )

    @classmethod
    def cloud_only(cls) -> RoutingPolicy:
        return cls(mode=ContractRoutingPolicy.CLOUD_ONLY, prefer_local=False, fallback="none")

    @classmethod
    def from_desired_state_entry(cls, entry: dict[str, Any]) -> Optional[RoutingPolicy]:
        """Build a RoutingPolicy from a desired-state model entry.

        Reads the ``serving_policy`` dict emitted by the desired state
        compiler (contract 1.16.0).  The dict contains ``routing_mode``,
        ``routing_preference``, and ``fallback.allow_cloud_fallback``.

        Mapping:
          routing_mode=local_only             → LOCAL_ONLY
          routing_mode=cloud_only             → CLOUD_ONLY
          routing_preference=local            → LOCAL_FIRST
          routing_preference=performance      → AUTO, prefer_local=True
          routing_preference=cloud            → AUTO, prefer_local=False
          routing_preference=quality          → AUTO, prefer_local=False (legacy alias)
          (none of the above)                 → AUTO (default)
        """
        sp = entry.get("serving_policy")
        if not sp:
            return None
        rm = sp.get("routing_mode", "auto")
        rp = sp.get("routing_preference")
        fb = sp.get("fallback") or {}
        cloud_fallback = "cloud" if fb.get("allow_cloud_fallback", True) else "none"
        local_fallback = "local" if fb.get("allow_local_fallback", True) else "none"

        if rm == "local_only":
            return cls.local_only()
        if rm == "cloud_only":
            return cls.cloud_only()
        if rp == "local":
            return cls.local_first(fallback=cloud_fallback)
        if rp == "performance":
            return cls.auto(prefer_local=True, fallback=cloud_fallback)
        if rp in {"quality", "cloud"}:
            return cls.auto(prefer_local=False, fallback=local_fallback)
        return cls.auto(fallback=cloud_fallback)

    @classmethod
    def from_metadata(cls, metadata: Optional[dict[str, str]]) -> Optional[RoutingPolicy]:
        if not metadata or "routing.policy" not in metadata:
            return None
        mode = metadata["routing.policy"]
        if mode == "local_only":
            return cls.local_only()
        if mode == "local_first":
            return cls.local_first(
                max_latency_ms=int(metadata["routing.max_latency_ms"])
                if "routing.max_latency_ms" in metadata
                else None,
                fallback=metadata.get("routing.fallback", "cloud"),
            )
        if mode == "cloud_only":
            return cls.cloud_only()
        if mode == "auto":
            return cls.auto(
                prefer_local=metadata.get("routing.prefer_local") != "false",
                max_latency_ms=int(metadata["routing.max_latency_ms"])
                if "routing.max_latency_ms" in metadata
                else None,
                fallback=metadata.get("routing.fallback", "cloud"),
            )
        return None
