"""Routing policy for Layer 4 runtime selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RoutingPolicy:
    mode: str  # "auto", "local_only", "cloud_only"
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
        return cls(mode="auto", prefer_local=prefer_local, max_latency_ms=max_latency_ms, fallback=fallback)

    @classmethod
    def local_only(cls) -> RoutingPolicy:
        return cls(mode="local_only")

    @classmethod
    def cloud_only(cls) -> RoutingPolicy:
        return cls(mode="cloud_only")

    @classmethod
    def from_metadata(cls, metadata: Optional[dict[str, str]]) -> Optional[RoutingPolicy]:
        if not metadata or "routing.policy" not in metadata:
            return None
        mode = metadata["routing.policy"]
        if mode == "local_only":
            return cls.local_only()
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
