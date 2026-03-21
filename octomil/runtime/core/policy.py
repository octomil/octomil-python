"""Routing policy for Layer 4 runtime selection.

Wraps the contract-level RoutingPolicy enum with additional runtime
hints (latency thresholds, fallback preferences).  The ``mode`` field
is always a ``ContractRoutingPolicy`` enum value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
