"""Server-fetched device configuration for tuned constants.

Fetches runtime-tuned parameters from ``GET /api/v1/device-config`` and
caches locally.  Falls back to minimal safe defaults when the server is
unreachable, revealing no proprietary tuning data.

Sections:
- ``quant_speed_factors``: quantization speed multipliers
- ``quant_preference_order``: quality-ordered quantization list
- ``early_exit``: per-preset thresholds and min-layer fractions
- ``routing``: score offsets for policy-based query routing
- ``smart_router``: engine preferences and generation thresholds

Pattern follows ``CatalogClient`` / ``_ServerFetcher`` in
``octomil.models.catalog_client``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from octomil.models.catalog_client import _ServerFetcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed config sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarlyExitPresetConfig:
    """Server-provided early exit preset thresholds."""

    threshold: float
    min_layers_fraction: float


@dataclass(frozen=True)
class SmartRouterDefaults:
    """Server-provided smart router tuning defaults."""

    long_gen_threshold: int = 256
    concurrency_threshold: int = 2
    prefer_single_engine: str = "auto"
    prefer_concurrent_engine: str = "auto"
    prefer_long_gen_engine: str = "auto"


@dataclass(frozen=True)
class RoutingOffsets:
    """Server-provided score offsets for policy-based routing."""

    quality_score_offset: float = 0.0
    balanced_score_offset: float = 0.0


@dataclass(frozen=True)
class DeviceConfig:
    """Typed container for all server-fetched tuned constants."""

    quant_speed_factors: dict[str, float] = field(default_factory=lambda: {"Q4_K_M": 1.0})
    quant_preference_order: list[str] = field(default_factory=lambda: ["Q4_K_M"])
    early_exit_presets: dict[str, EarlyExitPresetConfig] = field(
        default_factory=lambda: {
            "quality": EarlyExitPresetConfig(threshold=0.3, min_layers_fraction=0.5),
            "balanced": EarlyExitPresetConfig(threshold=0.3, min_layers_fraction=0.5),
            "fast": EarlyExitPresetConfig(threshold=0.3, min_layers_fraction=0.5),
        }
    )
    routing_offsets: RoutingOffsets = field(default_factory=RoutingOffsets)
    smart_router: SmartRouterDefaults = field(default_factory=SmartRouterDefaults)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceConfig:
        """Parse server JSON into a typed DeviceConfig."""
        # quant_speed_factors
        qsf = data.get("quant_speed_factors", {"Q4_K_M": 1.0})
        if not isinstance(qsf, dict):
            qsf = {"Q4_K_M": 1.0}

        # quant_preference_order
        qpo = data.get("quant_preference_order", ["Q4_K_M"])
        if not isinstance(qpo, list) or len(qpo) == 0:
            qpo = ["Q4_K_M"]

        # early_exit_presets
        ee_raw = data.get("early_exit_presets", {})
        ee_presets: dict[str, EarlyExitPresetConfig] = {}
        for name in ("quality", "balanced", "fast"):
            preset_data = ee_raw.get(name, {})
            ee_presets[name] = EarlyExitPresetConfig(
                threshold=float(preset_data.get("threshold", 0.3)),
                min_layers_fraction=float(preset_data.get("min_layers_fraction", 0.5)),
            )

        # routing_offsets
        ro_raw = data.get("routing_offsets", {})
        routing_offsets = RoutingOffsets(
            quality_score_offset=float(ro_raw.get("quality_score_offset", 0.0)),
            balanced_score_offset=float(ro_raw.get("balanced_score_offset", 0.0)),
        )

        # smart_router
        sr_raw = data.get("smart_router", {})
        smart_router = SmartRouterDefaults(
            long_gen_threshold=int(sr_raw.get("long_gen_threshold", 256)),
            concurrency_threshold=int(sr_raw.get("concurrency_threshold", 2)),
            prefer_single_engine=str(sr_raw.get("prefer_single_engine", "auto")),
            prefer_concurrent_engine=str(sr_raw.get("prefer_concurrent_engine", "auto")),
            prefer_long_gen_engine=str(sr_raw.get("prefer_long_gen_engine", "auto")),
        )

        return cls(
            quant_speed_factors=qsf,
            quant_preference_order=qpo,
            early_exit_presets=ee_presets,
            routing_offsets=routing_offsets,
            smart_router=smart_router,
        )


# ---------------------------------------------------------------------------
# Fallback defaults — reveal nothing proprietary
# ---------------------------------------------------------------------------

_FALLBACK_DATA: dict[str, Any] = {
    "quant_speed_factors": {"Q4_K_M": 1.0},
    "quant_preference_order": ["Q4_K_M"],
    "early_exit_presets": {
        "quality": {"threshold": 0.3, "min_layers_fraction": 0.5},
        "balanced": {"threshold": 0.3, "min_layers_fraction": 0.5},
        "fast": {"threshold": 0.3, "min_layers_fraction": 0.5},
    },
    "routing_offsets": {
        "quality_score_offset": 0.0,
        "balanced_score_offset": 0.0,
    },
    "smart_router": {
        "long_gen_threshold": 256,
        "concurrency_threshold": 2,
        "prefer_single_engine": "auto",
        "prefer_concurrent_engine": "auto",
        "prefer_long_gen_engine": "auto",
    },
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class DeviceConfigClient:
    """Fetches device config from ``GET /api/v1/device-config``.

    Uses the same ``_ServerFetcher`` infrastructure as ``CatalogClient``:
    ETag caching, disk persistence in ``~/.cache/octomil/``, TTL-based
    expiration, and graceful fallback to minimal safe defaults.
    """

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
    ) -> None:
        self._fetcher = _ServerFetcher(
            endpoint="device-config",
            cache_filename="device_config.json",
            default_data=_FALLBACK_DATA,
            api_base=api_base,
            api_key=api_key,
        )
        self._config: Optional[DeviceConfig] = None

    def get_config(self) -> DeviceConfig:
        """Return the current device config, refreshing if expired."""
        raw = self._fetcher.get()

        # If the raw data object identity hasn't changed, reuse parsed config
        if self._config is not None and raw is self._fetcher._cached.data:  # type: ignore[union-attr]
            return self._config

        self._config = DeviceConfig.from_dict(raw)
        return self._config


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_client: Optional[DeviceConfigClient] = None


def get_device_config() -> DeviceConfig:
    """Return the global ``DeviceConfig``, lazily initializing the client."""
    global _client
    if _client is None:
        _client = DeviceConfigClient()
    return _client.get_config()
