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
        fallback: str = "cloud",
    ) -> RoutingPolicy:
        return cls(
            mode=ContractRoutingPolicy.AUTO,
            prefer_local=prefer_local,
            max_latency_ms=max_latency_ms,
            fallback=fallback,
        )

    @classmethod
    def local_only(cls) -> RoutingPolicy:
        return cls(mode=ContractRoutingPolicy.LOCAL_ONLY)

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
        return cls(mode=ContractRoutingPolicy.CLOUD_ONLY)

    @classmethod
    def from_desired_state_entry(cls, entry: dict[str, Any]) -> Optional[RoutingPolicy]:
        """Build a RoutingPolicy from a desired-state model entry.

        Reads ``routing_policy`` (routing_mode from deployment) and
        ``routing_preference`` / ``cloud_fallback`` fields emitted by
        the desired state compiler.

        Mapping:
          routing_policy=local_only           → LOCAL_ONLY
          routing_policy=cloud_only           → CLOUD_ONLY
          routing_preference=local            → LOCAL_FIRST, fallback=cloud
          routing_preference=balanced         → AUTO, prefer_local=True
          routing_preference=quality          → AUTO, prefer_local=False (cloud-biased)
          (none of the above)                 → AUTO (default)
        """
        rp = entry.get("routing_policy")
        pref = entry.get("routing_preference")
        cf = entry.get("cloud_fallback")
        fallback = "cloud" if (cf and cf.get("enabled", True)) else "none"

        if rp == "local_only":
            return cls.local_only()
        if rp == "cloud_only":
            return cls.cloud_only()

        # routing_preference drives the AUTO/LOCAL_FIRST split
        if pref == "local":
            return cls.local_first(fallback=fallback)
        if pref == "quality":
            return cls.auto(prefer_local=False, fallback=fallback)
        if pref == "balanced":
            return cls.auto(prefer_local=True, fallback=fallback)

        # No routing fields set → caller should use default policy
        if rp is None and pref is None:
            return None

        # Unrecognized routing_policy → auto
        return cls.auto(fallback=fallback)

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
