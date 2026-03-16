"""Tests for PolicyEngine — all policy rules."""

from __future__ import annotations

from octomil.device_agent.policy.policy_engine import (
    PolicyConfig,
    PolicyEngine,
    WorkClass,
)


def _engine(**kwargs: object) -> PolicyEngine:
    """Create an engine with optional config overrides."""
    config = PolicyConfig(**kwargs)  # type: ignore[arg-type]
    return PolicyEngine(config)


class TestDownloadPolicy:
    def test_allows_download_on_wifi_with_storage(self) -> None:
        engine = _engine()
        engine.update_device_state(battery_pct=80, network_type="wifi", free_storage_bytes=5_000_000_000)
        allowed, reason = engine.should_allow_download(100_000_000)
        assert allowed
        assert reason == "ok"

    def test_blocks_download_below_storage_reserve(self) -> None:
        engine = _engine(reserve_storage_bytes=2_000_000_000)
        engine.update_device_state(free_storage_bytes=1_000_000_000)
        allowed, reason = engine.should_allow_download(100)
        assert not allowed
        assert reason == "storage_below_reserve"

    def test_blocks_background_download_on_low_battery(self) -> None:
        engine = _engine()
        engine.update_device_state(battery_pct=10, is_charging=False, network_type="wifi")
        allowed, reason = engine.should_allow_download(100)
        assert not allowed
        assert reason == "battery_low"

    def test_allows_user_initiated_on_low_battery(self) -> None:
        engine = _engine()
        engine.update_device_state(battery_pct=10, is_charging=False, network_type="wifi")
        allowed, _ = engine.should_allow_download(100, user_initiated=True)
        assert allowed

    def test_blocks_large_cellular_download(self) -> None:
        engine = _engine(max_cellular_download_bytes=20_000_000)
        engine.update_device_state(network_type="cellular", battery_pct=80)
        allowed, reason = engine.should_allow_download(50_000_000)
        assert not allowed
        assert reason == "cellular_size_limit"

    def test_allows_small_cellular_download(self) -> None:
        engine = _engine()
        engine.update_device_state(network_type="cellular", battery_pct=80)
        allowed, _ = engine.should_allow_download(5_000_000)
        assert allowed

    def test_allows_large_cellular_download_if_user_initiated(self) -> None:
        engine = _engine()
        engine.update_device_state(network_type="cellular", battery_pct=80)
        allowed, _ = engine.should_allow_download(50_000_000, user_initiated=True)
        assert allowed

    def test_blocks_download_on_unknown_network(self) -> None:
        engine = _engine()
        engine.update_device_state(network_type="unknown", battery_pct=80)
        allowed, reason = engine.should_allow_download(100)
        assert not allowed
        assert reason == "network_not_allowed"


class TestUploadPolicy:
    def test_allows_upload_on_wifi(self) -> None:
        engine = _engine()
        engine.update_device_state(battery_pct=80, network_type="wifi")
        allowed, _ = engine.should_allow_upload(1_000_000)
        assert allowed

    def test_blocks_upload_on_low_battery(self) -> None:
        engine = _engine()
        engine.update_device_state(battery_pct=10, is_charging=False)
        allowed, reason = engine.should_allow_upload(100)
        assert not allowed
        assert reason == "battery_low"

    def test_blocks_large_cellular_upload(self) -> None:
        engine = _engine()
        engine.update_device_state(network_type="cellular", battery_pct=80)
        allowed, reason = engine.should_allow_upload(500_000)
        assert not allowed
        assert reason == "cellular_size_limit"

    def test_allows_small_cellular_upload(self) -> None:
        engine = _engine()
        engine.update_device_state(network_type="cellular", battery_pct=80)
        allowed, _ = engine.should_allow_upload(100_000)
        assert allowed


class TestTrainingPolicy:
    def test_allows_training_when_conditions_met(self) -> None:
        engine = _engine()
        engine.update_device_state(
            battery_pct=80,
            is_charging=True,
            network_type="wifi",
            thermal_state="nominal",
            is_foreground=False,
            free_storage_bytes=5_000_000_000,
        )
        allowed, _ = engine.should_allow_training()
        assert allowed

    def test_blocks_training_not_charging(self) -> None:
        engine = _engine()
        engine.update_device_state(
            battery_pct=80,
            is_charging=False,
            network_type="wifi",
        )
        allowed, reason = engine.should_allow_training()
        assert not allowed
        assert reason == "not_charging"

    def test_blocks_training_low_battery(self) -> None:
        engine = _engine()
        engine.update_device_state(
            battery_pct=30,
            is_charging=True,
            network_type="wifi",
        )
        allowed, reason = engine.should_allow_training()
        assert not allowed
        assert reason == "battery_low"

    def test_blocks_training_on_cellular(self) -> None:
        engine = _engine()
        engine.update_device_state(
            battery_pct=80,
            is_charging=True,
            network_type="cellular",
        )
        allowed, reason = engine.should_allow_training()
        assert not allowed
        assert reason == "network_not_allowed"

    def test_blocks_training_on_thermal_serious(self) -> None:
        engine = _engine()
        engine.update_device_state(
            battery_pct=80,
            is_charging=True,
            network_type="wifi",
            thermal_state="serious",
        )
        allowed, reason = engine.should_allow_training()
        assert not allowed
        assert reason == "thermal_throttle"

    def test_blocks_training_on_foreground(self) -> None:
        engine = _engine()
        engine.update_device_state(
            battery_pct=80,
            is_charging=True,
            network_type="wifi",
            is_foreground=True,
        )
        allowed, reason = engine.should_allow_training()
        assert not allowed
        assert reason == "foreground_active"

    def test_blocks_training_low_storage(self) -> None:
        engine = _engine()
        engine.update_device_state(
            battery_pct=80,
            is_charging=True,
            network_type="wifi",
            free_storage_bytes=500_000_000,
        )
        allowed, reason = engine.should_allow_training()
        assert not allowed
        assert reason == "storage_below_reserve"


class TestFederatedTrainingPolicy:
    def test_blocks_federated_when_foreground(self) -> None:
        engine = _engine()
        engine.update_device_state(
            battery_pct=80,
            is_charging=True,
            network_type="wifi",
            is_foreground=True,
        )
        allowed, reason = engine.should_allow_federated_training()
        assert not allowed

    def test_allows_federated_when_idle(self) -> None:
        engine = _engine()
        engine.update_device_state(
            battery_pct=80,
            is_charging=True,
            network_type="wifi",
            is_foreground=False,
            free_storage_bytes=5_000_000_000,
        )
        allowed, _ = engine.should_allow_federated_training()
        assert allowed


class TestWarmupPolicy:
    def test_blocks_warmup_on_thermal_serious(self) -> None:
        engine = _engine()
        engine.update_device_state(thermal_state="serious")
        allowed, reason = engine.should_allow_warmup()
        assert not allowed
        assert reason == "thermal_throttle"

    def test_blocks_warmup_on_thermal_critical(self) -> None:
        engine = _engine()
        engine.update_device_state(thermal_state="critical")
        allowed, reason = engine.should_allow_warmup()
        assert not allowed
        assert reason == "thermal_throttle"

    def test_allows_warmup_on_nominal(self) -> None:
        engine = _engine()
        engine.update_device_state(thermal_state="nominal", battery_pct=50)
        allowed, _ = engine.should_allow_warmup()
        assert allowed

    def test_blocks_warmup_low_battery(self) -> None:
        engine = _engine()
        engine.update_device_state(battery_pct=10, is_charging=False)
        allowed, reason = engine.should_allow_warmup()
        assert not allowed
        assert reason == "battery_low"


class TestTelemetryPolicy:
    def test_low_battery_restricts_classes(self) -> None:
        engine = _engine()
        engine.update_device_state(battery_pct=10, is_charging=False)
        policy = engine.get_telemetry_policy()
        assert policy["allowed_classes"] == ["MUST_KEEP"]
        assert policy["max_batch_size"] == 10

    def test_cellular_limits_classes(self) -> None:
        engine = _engine()
        engine.update_device_state(battery_pct=80, network_type="cellular")
        policy = engine.get_telemetry_policy()
        assert "BEST_EFFORT" not in policy["allowed_classes"]

    def test_normal_conditions_allow_all(self) -> None:
        engine = _engine()
        engine.update_device_state(battery_pct=80, network_type="wifi")
        policy = engine.get_telemetry_policy()
        assert "BEST_EFFORT" in policy["allowed_classes"]
        assert policy["max_batch_size"] == 100


class TestWorkClassification:
    def test_inference_is_critical(self) -> None:
        engine = _engine()
        assert engine.classify_work("inference") == WorkClass.CRITICAL_FOREGROUND

    def test_download_is_important(self) -> None:
        engine = _engine()
        assert engine.classify_work("download") == WorkClass.BACKGROUND_IMPORTANT

    def test_unknown_is_best_effort(self) -> None:
        engine = _engine()
        assert engine.classify_work("telemetry_flush") == WorkClass.BACKGROUND_BEST_EFFORT


class TestDeviceState:
    def test_get_device_state_returns_snapshot(self) -> None:
        engine = _engine()
        engine.update_device_state(
            battery_pct=42,
            is_charging=True,
            network_type="wifi",
        )
        state = engine.get_device_state()
        assert state["battery_pct"] == 42
        assert state["is_charging"] is True
        assert state["network_type"] == "wifi"

    def test_partial_update(self) -> None:
        engine = _engine()
        engine.update_device_state(battery_pct=50)
        engine.update_device_state(network_type="cellular")
        state = engine.get_device_state()
        assert state["battery_pct"] == 50
        assert state["network_type"] == "cellular"
