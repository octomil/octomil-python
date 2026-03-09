"""Tests for octomil.device_config — server-fetched tuned constants.

Covers:
- DeviceConfig.from_dict parsing with full, partial, and empty data
- Fallback defaults when keys are missing
- DeviceConfigClient with mock server fetcher
- Module-level get_device_config() singleton
- Integration with model_optimizer, early_exit, routing, smart_router
"""

from __future__ import annotations

from typing import Any

from octomil.device_config import (
    _FALLBACK_DATA,
    DeviceConfig,
    DeviceConfigClient,
)

# ---------------------------------------------------------------------------
# DeviceConfig.from_dict
# ---------------------------------------------------------------------------


class TestDeviceConfigFromDict:
    def test_full_server_response(self) -> None:
        """Parse a complete server response into DeviceConfig."""
        data: dict[str, Any] = {
            "quant_speed_factors": {"Q4_K_M": 1.0, "Q8_0": 0.7},
            "quant_preference_order": ["Q8_0", "Q4_K_M"],
            "early_exit_presets": {
                "quality": {"threshold": 0.1, "min_layers_fraction": 0.75},
                "balanced": {"threshold": 0.3, "min_layers_fraction": 0.5},
                "fast": {"threshold": 0.5, "min_layers_fraction": 0.25},
            },
            "routing_offsets": {
                "quality_score_offset": 0.5,
                "balanced_score_offset": 0.25,
                "latency_offset": 0.5,
                "throughput_offset": 0.25,
            },
            "smart_router": {
                "long_gen_threshold": 512,
                "concurrency_threshold": 3,
                "prefer_throughput_engine": "mlx-lm",
                "prefer_latency_engine": "llama.cpp",
            },
        }
        cfg = DeviceConfig.from_dict(data)

        assert cfg.quant_speed_factors == {"Q4_K_M": 1.0, "Q8_0": 0.7}
        assert cfg.quant_preference_order == ["Q8_0", "Q4_K_M"]
        assert cfg.early_exit_presets["quality"].threshold == 0.1
        assert cfg.early_exit_presets["fast"].min_layers_fraction == 0.25
        assert cfg.routing_offsets.quality_score_offset == 0.5
        assert cfg.routing_offsets.balanced_score_offset == 0.25
        assert cfg.routing_offsets.latency_offset == 0.5
        assert cfg.routing_offsets.throughput_offset == 0.25
        assert cfg.smart_router.long_gen_threshold == 512
        assert cfg.smart_router.prefer_throughput_engine == "mlx-lm"
        assert cfg.smart_router.prefer_latency_engine == "llama.cpp"

    def test_empty_dict_uses_safe_defaults(self) -> None:
        """Empty dict should produce safe fallback values."""
        cfg = DeviceConfig.from_dict({})

        assert cfg.quant_speed_factors == {"Q4_K_M": 1.0}
        assert cfg.quant_preference_order == ["Q4_K_M"]
        assert cfg.early_exit_presets["quality"].threshold == 0.3
        assert cfg.early_exit_presets["balanced"].threshold == 0.3
        assert cfg.early_exit_presets["fast"].threshold == 0.3
        assert cfg.routing_offsets.quality_score_offset == 0.0
        assert cfg.routing_offsets.balanced_score_offset == 0.0
        assert cfg.routing_offsets.latency_offset == 0.0
        assert cfg.routing_offsets.throughput_offset == 0.0
        assert cfg.smart_router.long_gen_threshold == 256
        assert cfg.smart_router.prefer_throughput_engine == "auto"
        assert cfg.smart_router.prefer_latency_engine == "auto"

    def test_partial_data_fills_defaults(self) -> None:
        """Missing sections should fall back to safe defaults."""
        data = {
            "quant_speed_factors": {"Q4_0": 1.1},
            # other sections missing
        }
        cfg = DeviceConfig.from_dict(data)

        assert cfg.quant_speed_factors == {"Q4_0": 1.1}
        assert cfg.quant_preference_order == ["Q4_K_M"]  # default
        assert cfg.smart_router.prefer_latency_engine == "auto"

    def test_invalid_quant_speed_factors_type(self) -> None:
        """Non-dict quant_speed_factors should fallback."""
        cfg = DeviceConfig.from_dict({"quant_speed_factors": "invalid"})
        assert cfg.quant_speed_factors == {"Q4_K_M": 1.0}

    def test_empty_quant_preference_order(self) -> None:
        """Empty list should fallback to single-element default."""
        cfg = DeviceConfig.from_dict({"quant_preference_order": []})
        assert cfg.quant_preference_order == ["Q4_K_M"]

    def test_fallback_data_parses_cleanly(self) -> None:
        """The _FALLBACK_DATA constant should parse without error."""
        cfg = DeviceConfig.from_dict(_FALLBACK_DATA)
        assert cfg.quant_speed_factors == {"Q4_K_M": 1.0}
        assert cfg.smart_router.long_gen_threshold == 256


# ---------------------------------------------------------------------------
# Fallback defaults — must be safe / minimal
# ---------------------------------------------------------------------------


class TestFallbackDefaults:
    def test_fallback_quant_speed_neutral(self) -> None:
        """Fallback quant speed should be neutral baseline only."""
        cfg = DeviceConfig.from_dict(_FALLBACK_DATA)
        assert len(cfg.quant_speed_factors) == 1
        assert cfg.quant_speed_factors["Q4_K_M"] == 1.0

    def test_fallback_quant_pref_single(self) -> None:
        """Fallback quant preference should reveal only Q4_K_M."""
        cfg = DeviceConfig.from_dict(_FALLBACK_DATA)
        assert cfg.quant_preference_order == ["Q4_K_M"]

    def test_fallback_early_exit_all_balanced(self) -> None:
        """Fallback early exit presets should all be identical (balanced)."""
        cfg = DeviceConfig.from_dict(_FALLBACK_DATA)
        for name in ("quality", "balanced", "fast"):
            assert cfg.early_exit_presets[name].threshold == 0.3
            assert cfg.early_exit_presets[name].min_layers_fraction == 0.5

    def test_fallback_routing_offsets_zero(self) -> None:
        """Fallback routing offsets should be zero (no bias)."""
        cfg = DeviceConfig.from_dict(_FALLBACK_DATA)
        assert cfg.routing_offsets.quality_score_offset == 0.0
        assert cfg.routing_offsets.balanced_score_offset == 0.0
        assert cfg.routing_offsets.latency_offset == 0.0
        assert cfg.routing_offsets.throughput_offset == 0.0

    def test_fallback_smart_router_auto(self) -> None:
        """Fallback smart router engine prefs should be 'auto'."""
        cfg = DeviceConfig.from_dict(_FALLBACK_DATA)
        assert cfg.smart_router.prefer_throughput_engine == "auto"
        assert cfg.smart_router.prefer_latency_engine == "auto"


# ---------------------------------------------------------------------------
# DeviceConfigClient
# ---------------------------------------------------------------------------


class TestDeviceConfigClient:
    def test_returns_fallback_when_offline(self) -> None:
        """When server and disk cache are both unavailable, return fallback."""
        client = DeviceConfigClient(
            api_base="http://localhost:9999",
            api_key="test",
        )
        cfg = client.get_config()
        assert isinstance(cfg, DeviceConfig)
        assert cfg.quant_speed_factors == {"Q4_K_M": 1.0}

    def test_caches_parsed_config(self) -> None:
        """Second call should return the same parsed DeviceConfig object."""
        client = DeviceConfigClient(
            api_base="http://localhost:9999",
            api_key="test",
        )
        cfg1 = client.get_config()
        cfg2 = client.get_config()
        assert cfg1 is cfg2


# ---------------------------------------------------------------------------
# Integration: model_optimizer reads from DeviceConfig
# ---------------------------------------------------------------------------


class TestModelOptimizerIntegration:
    def test_quant_speed_factors_from_config(self) -> None:
        """model_optimizer._get_quant_speed_factors returns server values."""
        from octomil.model_optimizer import _get_quant_speed_factors

        factors = _get_quant_speed_factors()
        # conftest injects the original hardcoded values
        assert "Q4_K_M" in factors
        assert factors["Q4_K_M"] == 1.0
        assert "Q8_0" in factors
        assert factors["Q8_0"] == 0.7

    def test_quant_preference_order_from_config(self) -> None:
        """model_optimizer._get_quant_preference_order returns server values."""
        from octomil.model_optimizer import _get_quant_preference_order

        order = _get_quant_preference_order()
        assert order[0] == "Q8_0"
        assert "Q4_K_M" in order


# ---------------------------------------------------------------------------
# Integration: early_exit reads from DeviceConfig
# ---------------------------------------------------------------------------


class TestEarlyExitIntegration:
    def test_preset_thresholds_loaded(self) -> None:
        """PRESET_THRESHOLDS should load values from server config."""
        from octomil.early_exit import PRESET_THRESHOLDS, SpeedQualityPreset, _ensure_presets_loaded

        _ensure_presets_loaded()
        assert PRESET_THRESHOLDS[SpeedQualityPreset.QUALITY] == 0.1
        assert PRESET_THRESHOLDS[SpeedQualityPreset.BALANCED] == 0.3
        assert PRESET_THRESHOLDS[SpeedQualityPreset.FAST] == 0.5

    def test_preset_min_layers_loaded(self) -> None:
        """PRESET_MIN_LAYERS_FRACTION should load values from server config."""
        from octomil.early_exit import PRESET_MIN_LAYERS_FRACTION, SpeedQualityPreset, _ensure_presets_loaded

        _ensure_presets_loaded()
        assert PRESET_MIN_LAYERS_FRACTION[SpeedQualityPreset.QUALITY] == 0.75
        assert PRESET_MIN_LAYERS_FRACTION[SpeedQualityPreset.BALANCED] == 0.5
        assert PRESET_MIN_LAYERS_FRACTION[SpeedQualityPreset.FAST] == 0.25


# ---------------------------------------------------------------------------
# Integration: routing reads offsets from DeviceConfig
# ---------------------------------------------------------------------------


class TestRoutingIntegration:
    def test_policy_from_dict_ios_scoring_defaults(self) -> None:
        """RoutingPolicy.from_dict should use iOS-aligned scoring defaults."""
        from octomil.routing import RoutingPolicy

        policy = RoutingPolicy.from_dict({"version": 1, "thresholds": {}})
        assert policy.length_weight == 0.5
        assert policy.indicator_weight == 0.5
        assert policy.fast_threshold == 0.3
        assert policy.quality_threshold == 0.7
        assert policy.indicator_normalizor == 3.0

    def test_policy_from_dict_explicit_weights_override(self) -> None:
        """Explicit iOS-aligned weights in policy dict should be used."""
        from octomil.routing import RoutingPolicy

        policy = RoutingPolicy.from_dict(
            {
                "version": 1,
                "thresholds": {},
                "length_weight": 0.6,
                "indicator_weight": 0.4,
                "fast_threshold": 0.2,
                "quality_threshold": 0.8,
                "indicator_normalizor": 5.0,
            }
        )
        assert policy.length_weight == 0.6
        assert policy.indicator_weight == 0.4
        assert policy.fast_threshold == 0.2
        assert policy.quality_threshold == 0.8
        assert policy.indicator_normalizor == 5.0


# ---------------------------------------------------------------------------
# Integration: smart_router reads from DeviceConfig
# ---------------------------------------------------------------------------


class TestSmartRouterIntegration:
    def test_default_router_config_from_server(self) -> None:
        """_default_router_config() should read from device config."""
        from octomil.smart_router import _default_router_config

        cfg = _default_router_config()
        # conftest injects original values
        assert cfg.long_gen_threshold == 512
        assert cfg.concurrency_threshold == 2
        assert cfg.prefer_throughput_engine == "mlx-lm"
        assert cfg.prefer_latency_engine == "llama.cpp"
