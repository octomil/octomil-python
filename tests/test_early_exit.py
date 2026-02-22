"""Tests for edgeml.early_exit — early exit / adaptive computation depth.

Covers:
- EarlyExitConfig construction and property computation
- config_from_cli() builder
- SpeedQualityPreset mapping to thresholds
- compute_entropy() correctness
- should_exit_early() decision logic
- EarlyExitMonitor aggregate statistics
- EarlyExitRequestMetrics serialization
- serve.py integration: /v1/early-exit/stats endpoint, usage.early_exit in responses
- CLI --early-exit-threshold and --speed-quality flags
- telemetry report_early_exit_stats and report_generation_completed with early_exit_stats
"""

from __future__ import annotations

import asyncio
import math
import time
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from edgeml.early_exit import (
    PRESET_MIN_LAYERS_FRACTION,
    PRESET_THRESHOLDS,
    EarlyExitConfig,
    EarlyExitMonitor,
    EarlyExitRequestMetrics,
    EarlyExitStats,
    SpeedQualityPreset,
    TokenExitRecord,
    compute_entropy,
    config_from_cli,
    should_exit_early,
)


# ---------------------------------------------------------------------------
# SpeedQualityPreset
# ---------------------------------------------------------------------------


class TestSpeedQualityPreset:
    def test_enum_values(self):
        assert SpeedQualityPreset.QUALITY.value == "quality"
        assert SpeedQualityPreset.BALANCED.value == "balanced"
        assert SpeedQualityPreset.FAST.value == "fast"

    def test_preset_thresholds_exist_for_all_presets(self):
        for preset in SpeedQualityPreset:
            assert preset in PRESET_THRESHOLDS

    def test_preset_min_layers_exist_for_all_presets(self):
        for preset in SpeedQualityPreset:
            assert preset in PRESET_MIN_LAYERS_FRACTION

    def test_quality_threshold_lowest(self):
        """Quality preset should have the lowest threshold (fewest exits)."""
        assert (
            PRESET_THRESHOLDS[SpeedQualityPreset.QUALITY]
            < PRESET_THRESHOLDS[SpeedQualityPreset.BALANCED]
        )

    def test_fast_threshold_highest(self):
        """Fast preset should have the highest threshold (most exits)."""
        assert (
            PRESET_THRESHOLDS[SpeedQualityPreset.FAST]
            > PRESET_THRESHOLDS[SpeedQualityPreset.BALANCED]
        )

    def test_quality_min_layers_highest(self):
        """Quality preset should use the most layers before allowing exit."""
        assert (
            PRESET_MIN_LAYERS_FRACTION[SpeedQualityPreset.QUALITY]
            > PRESET_MIN_LAYERS_FRACTION[SpeedQualityPreset.BALANCED]
        )

    def test_fast_min_layers_lowest(self):
        """Fast preset should require the fewest layers before allowing exit."""
        assert (
            PRESET_MIN_LAYERS_FRACTION[SpeedQualityPreset.FAST]
            < PRESET_MIN_LAYERS_FRACTION[SpeedQualityPreset.BALANCED]
        )


# ---------------------------------------------------------------------------
# EarlyExitConfig
# ---------------------------------------------------------------------------


class TestEarlyExitConfig:
    def test_defaults(self):
        cfg = EarlyExitConfig()
        assert cfg.enabled is False
        assert cfg.threshold == 0.3
        assert cfg.preset is None
        assert cfg.min_layers_fraction == 0.5
        assert cfg.total_layers is None

    def test_effective_threshold_without_preset(self):
        cfg = EarlyExitConfig(threshold=0.4)
        assert cfg.effective_threshold == 0.4

    def test_effective_threshold_with_preset(self):
        cfg = EarlyExitConfig(preset=SpeedQualityPreset.FAST)
        assert cfg.effective_threshold == PRESET_THRESHOLDS[SpeedQualityPreset.FAST]

    def test_preset_overrides_threshold(self):
        """When preset is set, effective_threshold uses preset value, not raw threshold."""
        cfg = EarlyExitConfig(threshold=0.99, preset=SpeedQualityPreset.QUALITY)
        assert cfg.effective_threshold == PRESET_THRESHOLDS[SpeedQualityPreset.QUALITY]

    def test_effective_min_layers_fraction_without_preset(self):
        cfg = EarlyExitConfig(min_layers_fraction=0.6)
        assert cfg.effective_min_layers_fraction == 0.6

    def test_effective_min_layers_fraction_with_preset(self):
        cfg = EarlyExitConfig(preset=SpeedQualityPreset.FAST)
        assert (
            cfg.effective_min_layers_fraction
            == PRESET_MIN_LAYERS_FRACTION[SpeedQualityPreset.FAST]
        )

    def test_min_layers_without_total(self):
        cfg = EarlyExitConfig()
        assert cfg.min_layers is None

    def test_min_layers_with_total(self):
        cfg = EarlyExitConfig(total_layers=32, min_layers_fraction=0.5)
        assert cfg.min_layers == 16

    def test_min_layers_at_least_one(self):
        cfg = EarlyExitConfig(total_layers=2, min_layers_fraction=0.0)
        assert cfg.min_layers == 1

    def test_min_layers_with_preset(self):
        cfg = EarlyExitConfig(total_layers=32, preset=SpeedQualityPreset.FAST)
        expected = max(1, int(32 * PRESET_MIN_LAYERS_FRACTION[SpeedQualityPreset.FAST]))
        assert cfg.min_layers == expected

    def test_frozen(self):
        cfg = EarlyExitConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# config_from_cli
# ---------------------------------------------------------------------------


class TestConfigFromCli:
    def test_no_args_disabled(self):
        cfg = config_from_cli()
        assert cfg.enabled is False

    def test_threshold_only(self):
        cfg = config_from_cli(early_exit_threshold=0.4)
        assert cfg.enabled is True
        assert cfg.threshold == 0.4
        assert cfg.preset is None

    def test_speed_quality_only(self):
        cfg = config_from_cli(speed_quality="fast")
        assert cfg.enabled is True
        assert cfg.preset == SpeedQualityPreset.FAST
        assert cfg.threshold == PRESET_THRESHOLDS[SpeedQualityPreset.FAST]

    def test_threshold_overrides_preset(self):
        cfg = config_from_cli(early_exit_threshold=0.7, speed_quality="quality")
        assert cfg.enabled is True
        assert cfg.threshold == 0.7
        assert cfg.preset == SpeedQualityPreset.QUALITY
        # effective_threshold uses preset, but raw threshold is stored separately
        assert cfg.effective_threshold == PRESET_THRESHOLDS[SpeedQualityPreset.QUALITY]

    def test_invalid_preset_ignored(self):
        cfg = config_from_cli(speed_quality="turbo")
        # Invalid preset is ignored; enabled via speed_quality arg presence
        assert cfg.enabled is True
        assert cfg.preset is None
        assert cfg.threshold == 0.3  # default

    def test_balanced_preset(self):
        cfg = config_from_cli(speed_quality="balanced")
        assert cfg.preset == SpeedQualityPreset.BALANCED
        assert cfg.enabled is True


# ---------------------------------------------------------------------------
# compute_entropy
# ---------------------------------------------------------------------------


class TestComputeEntropy:
    def test_empty_returns_zero(self):
        assert compute_entropy([]) == 0.0

    def test_single_element_returns_zero(self):
        assert compute_entropy([1.0]) == 0.0

    def test_uniform_distribution_returns_one(self):
        """Uniform distribution should have maximum entropy (normalized to 1)."""
        logits = [0.0, 0.0, 0.0, 0.0]
        result = compute_entropy(logits)
        assert abs(result - 1.0) < 0.001

    def test_peaked_distribution_low_entropy(self):
        """A distribution peaked on one value should have low entropy."""
        logits = [100.0, 0.0, 0.0, 0.0]
        result = compute_entropy(logits)
        assert result < 0.1

    def test_entropy_range(self):
        """Entropy should always be in [0, 1] for normalized output."""
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_entropy(logits)
        assert 0.0 <= result <= 1.0

    def test_two_equal_logits(self):
        """Two equal logits should give normalized entropy of 1.0."""
        logits = [1.0, 1.0]
        result = compute_entropy(logits)
        assert abs(result - 1.0) < 0.001

    def test_numerical_stability_large_logits(self):
        """Should handle very large logit values without overflow."""
        logits = [1000.0, 999.0, 998.0]
        result = compute_entropy(logits)
        assert 0.0 <= result <= 1.0
        assert not math.isnan(result)

    def test_numerical_stability_negative_logits(self):
        """Should handle negative logits correctly."""
        logits = [-1.0, -2.0, -3.0]
        result = compute_entropy(logits)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# should_exit_early
# ---------------------------------------------------------------------------


class TestShouldExitEarly:
    def test_disabled_config(self):
        cfg = EarlyExitConfig(enabled=False, threshold=0.5)
        assert should_exit_early(0.1, cfg, 10) is False

    def test_below_threshold_exits(self):
        cfg = EarlyExitConfig(enabled=True, threshold=0.3, total_layers=32)
        assert should_exit_early(0.1, cfg, 20) is True

    def test_above_threshold_does_not_exit(self):
        cfg = EarlyExitConfig(enabled=True, threshold=0.3, total_layers=32)
        assert should_exit_early(0.5, cfg, 20) is False

    def test_equal_to_threshold_does_not_exit(self):
        cfg = EarlyExitConfig(enabled=True, threshold=0.3, total_layers=32)
        assert should_exit_early(0.3, cfg, 20) is False

    def test_min_layer_not_reached(self):
        """Should not exit before min_layers even if entropy is low."""
        cfg = EarlyExitConfig(
            enabled=True, threshold=0.5, total_layers=32, min_layers_fraction=0.5
        )
        # min_layers = 16, current_layer = 5 → too early
        assert should_exit_early(0.01, cfg, 5) is False

    def test_min_layer_reached(self):
        """Should exit after min_layers if entropy is low."""
        cfg = EarlyExitConfig(
            enabled=True, threshold=0.5, total_layers=32, min_layers_fraction=0.5
        )
        # min_layers = 16, current_layer = 17 → ok
        assert should_exit_early(0.01, cfg, 17) is True

    def test_no_total_layers_no_min_check(self):
        """When total_layers is None, min_layers is None and no layer check is done."""
        cfg = EarlyExitConfig(enabled=True, threshold=0.5, total_layers=None)
        assert should_exit_early(0.1, cfg, 1) is True

    def test_preset_threshold_used(self):
        cfg = EarlyExitConfig(
            enabled=True,
            preset=SpeedQualityPreset.FAST,
            total_layers=32,
        )
        # FAST threshold = 0.5, min_layers_frac = 0.25 → min_layers = 8
        assert should_exit_early(0.4, cfg, 10) is True
        assert should_exit_early(0.6, cfg, 10) is False


# ---------------------------------------------------------------------------
# TokenExitRecord
# ---------------------------------------------------------------------------


class TestTokenExitRecord:
    def test_defaults(self):
        record = TokenExitRecord()
        assert record.layer_exited is None
        assert record.total_layers is None
        assert record.entropy == 0.0
        assert record.exited_early is False

    def test_with_values(self):
        record = TokenExitRecord(
            layer_exited=10, total_layers=32, entropy=0.15, exited_early=True
        )
        assert record.layer_exited == 10
        assert record.exited_early is True


# ---------------------------------------------------------------------------
# EarlyExitRequestMetrics
# ---------------------------------------------------------------------------


class TestEarlyExitRequestMetrics:
    def test_exit_percentage_zero_tokens(self):
        m = EarlyExitRequestMetrics(total_tokens=0)
        assert m.exit_percentage == 0.0

    def test_exit_percentage_calculation(self):
        m = EarlyExitRequestMetrics(total_tokens=100, early_exit_tokens=30)
        assert m.exit_percentage == 30.0

    def test_to_dict(self):
        m = EarlyExitRequestMetrics(
            total_tokens=50,
            early_exit_tokens=15,
            avg_layers_used=20.5,
            avg_entropy=0.2345,
            min_layers_used=8,
            max_layers_used=32,
        )
        d = m.to_dict()
        assert d["total_tokens"] == 50
        assert d["early_exit_tokens"] == 15
        assert d["exit_percentage"] == 30.0
        assert d["avg_layers_used"] == 20.5
        assert d["avg_entropy"] == 0.2345
        assert d["min_layers_used"] == 8
        assert d["max_layers_used"] == 32


# ---------------------------------------------------------------------------
# EarlyExitStats
# ---------------------------------------------------------------------------


class TestEarlyExitStats:
    def test_defaults(self):
        stats = EarlyExitStats()
        assert stats.exit_percentage == 0.0
        assert stats.avg_layers_used == 0.0
        assert stats.avg_entropy == 0.0

    def test_exit_percentage(self):
        stats = EarlyExitStats(total_tokens=100, total_early_exit_tokens=25)
        assert stats.exit_percentage == 25.0

    def test_avg_layers_used(self):
        stats = EarlyExitStats(total_tokens=10, total_layers_used=200.0)
        assert stats.avg_layers_used == 20.0

    def test_to_dict(self):
        stats = EarlyExitStats(
            total_requests=5,
            total_tokens=100,
            total_early_exit_tokens=30,
            total_layers_used=2000.0,
            total_entropy_sum=25.0,
        )
        d = stats.to_dict()
        assert d["total_requests"] == 5
        assert d["exit_percentage"] == 30.0
        assert d["avg_layers_used"] == 20.0
        assert d["avg_entropy"] == 0.25


# ---------------------------------------------------------------------------
# EarlyExitMonitor
# ---------------------------------------------------------------------------


class TestEarlyExitMonitor:
    def test_record_request(self):
        cfg = EarlyExitConfig(enabled=True, threshold=0.3)
        monitor = EarlyExitMonitor(cfg)

        metrics = EarlyExitRequestMetrics(
            total_tokens=50,
            early_exit_tokens=15,
            avg_layers_used=20.0,
            avg_entropy=0.2,
        )
        monitor.record_request(metrics)

        stats = monitor.stats
        assert stats.total_requests == 1
        assert stats.total_tokens == 50
        assert stats.total_early_exit_tokens == 15
        assert stats.exit_percentage == 30.0

    def test_multiple_requests(self):
        cfg = EarlyExitConfig(enabled=True, threshold=0.3)
        monitor = EarlyExitMonitor(cfg)

        m1 = EarlyExitRequestMetrics(
            total_tokens=100,
            early_exit_tokens=30,
            avg_layers_used=20.0,
            avg_entropy=0.2,
        )
        m2 = EarlyExitRequestMetrics(
            total_tokens=50,
            early_exit_tokens=25,
            avg_layers_used=16.0,
            avg_entropy=0.15,
        )
        monitor.record_request(m1)
        monitor.record_request(m2)

        stats = monitor.stats
        assert stats.total_requests == 2
        assert stats.total_tokens == 150
        assert stats.total_early_exit_tokens == 55

    def test_simulate_token_exits_disabled(self):
        cfg = EarlyExitConfig(enabled=False)
        monitor = EarlyExitMonitor(cfg)

        result = monitor.simulate_token_exits(token_count=50, total_layers=32)
        assert result.total_tokens == 50
        assert result.early_exit_tokens == 0
        assert result.avg_layers_used == 32.0

    def test_simulate_token_exits_enabled(self):
        cfg = EarlyExitConfig(enabled=True, threshold=0.3, total_layers=32)
        monitor = EarlyExitMonitor(cfg)

        result = monitor.simulate_token_exits(token_count=100, total_layers=32)
        assert result.total_tokens == 100
        assert result.early_exit_tokens > 0
        assert result.early_exit_tokens < 100
        assert result.avg_layers_used < 32.0
        assert result.avg_layers_used > 0.0

    def test_simulate_token_exits_zero_tokens(self):
        cfg = EarlyExitConfig(enabled=True, threshold=0.3)
        monitor = EarlyExitMonitor(cfg)

        result = monitor.simulate_token_exits(token_count=0, total_layers=32)
        assert result.total_tokens == 0
        assert result.early_exit_tokens == 0

    def test_simulate_fast_preset_more_exits(self):
        """Fast preset should produce more early exits than quality."""
        fast_cfg = EarlyExitConfig(
            enabled=True, preset=SpeedQualityPreset.FAST, total_layers=32
        )
        quality_cfg = EarlyExitConfig(
            enabled=True, preset=SpeedQualityPreset.QUALITY, total_layers=32
        )
        fast_monitor = EarlyExitMonitor(fast_cfg)
        quality_monitor = EarlyExitMonitor(quality_cfg)

        fast_result = fast_monitor.simulate_token_exits(
            token_count=100, total_layers=32
        )
        quality_result = quality_monitor.simulate_token_exits(
            token_count=100, total_layers=32
        )

        assert fast_result.early_exit_tokens > quality_result.early_exit_tokens

    def test_get_stats_dict(self):
        cfg = EarlyExitConfig(
            enabled=True,
            threshold=0.3,
            preset=SpeedQualityPreset.BALANCED,
            total_layers=32,
        )
        monitor = EarlyExitMonitor(cfg)

        d = monitor.get_stats_dict()
        assert "config" in d
        assert "stats" in d
        assert d["config"]["enabled"] is True
        assert d["config"]["preset"] == "balanced"
        assert d["config"]["total_layers"] == 32
        assert d["config"]["min_layers"] == 16

    def test_get_stats_dict_no_preset(self):
        cfg = EarlyExitConfig(enabled=True, threshold=0.4)
        monitor = EarlyExitMonitor(cfg)

        d = monitor.get_stats_dict()
        assert "preset" not in d["config"]


# ---------------------------------------------------------------------------
# Serve integration — FastAPI endpoints with early exit
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_app_with_early_exit():
    """Create a FastAPI app with EchoBackend and early exit enabled."""
    from edgeml.serve import EchoBackend, create_app

    ee_cfg = EarlyExitConfig(enabled=True, threshold=0.3, total_layers=32)

    with patch("edgeml.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo

        app = create_app("test-model", early_exit_config=ee_cfg)

        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())

    return app


@pytest.fixture
def echo_app_without_early_exit():
    """Create a FastAPI app with EchoBackend and no early exit."""
    from edgeml.serve import EchoBackend, create_app

    with patch("edgeml.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo

        app = create_app("test-model")

        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())

    return app


@pytest.mark.asyncio
async def test_early_exit_stats_endpoint_enabled(echo_app_with_early_exit):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=echo_app_with_early_exit)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/early-exit/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["config"]["enabled"] is True
    assert data["config"]["threshold"] == 0.3
    assert data["stats"]["total_requests"] == 0


@pytest.mark.asyncio
async def test_early_exit_stats_endpoint_disabled(echo_app_without_early_exit):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=echo_app_without_early_exit)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/early-exit/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False


@pytest.mark.asyncio
async def test_early_exit_in_completion_response(echo_app_with_early_exit):
    """Non-streaming completion should include early_exit in usage when enabled."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=echo_app_with_early_exit)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello world"}],
                "max_tokens": 50,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    usage = data["usage"]
    assert "early_exit" in usage
    ee = usage["early_exit"]
    assert "total_tokens" in ee
    assert "early_exit_tokens" in ee
    assert "exit_percentage" in ee
    assert "avg_layers_used" in ee


@pytest.mark.asyncio
async def test_early_exit_stats_update_after_request(echo_app_with_early_exit):
    """After a request, early exit stats should be updated."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=echo_app_with_early_exit)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Make a request
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        # Check stats
        resp = await client.get("/v1/early-exit/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["stats"]["total_requests"] == 1
    assert data["stats"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_no_early_exit_in_response_when_disabled(echo_app_without_early_exit):
    """When early exit is disabled, usage should not include early_exit."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=echo_app_without_early_exit)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "early_exit" not in data["usage"]


@pytest.mark.asyncio
async def test_health_includes_early_exit_when_enabled(echo_app_with_early_exit):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=echo_app_with_early_exit)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "early_exit" in data
    assert data["early_exit"]["enabled"] is True


@pytest.mark.asyncio
async def test_health_no_early_exit_when_disabled(echo_app_without_early_exit):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=echo_app_without_early_exit)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "early_exit" not in data


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------


class TestCliEarlyExitFlags:
    def test_serve_help_shows_early_exit_threshold(self):
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--early-exit-threshold" in result.output

    def test_serve_help_shows_speed_quality(self):
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--speed-quality" in result.output

    def test_speed_quality_choices(self):
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert "quality" in result.output
        assert "balanced" in result.output
        assert "fast" in result.output


# ---------------------------------------------------------------------------
# Telemetry — report_early_exit_stats
# ---------------------------------------------------------------------------


class TestTelemetryEarlyExit:
    def test_report_early_exit_stats_payload(self):
        from edgeml.telemetry import TelemetryReporter

        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_early_exit_stats(
                session_id="s1",
                model_id="model-a",
                version="2.0",
                total_tokens=100,
                early_exit_tokens=30,
                exit_percentage=30.0,
                avg_layers_used=20.5,
                avg_entropy=0.22,
            )
            time.sleep(0.15)
            reporter.close()

        assert len(sent) >= 1
        p = sent[0]
        assert p["event_type"] == "early_exit_stats"
        m = p["metrics"]
        assert m["total_tokens"] == 100
        assert m["early_exit_tokens"] == 30
        assert m["exit_percentage"] == 30.0
        assert m["avg_layers_used"] == 20.5
        assert m["avg_entropy"] == 0.22

    def test_generation_completed_includes_early_exit(self):
        from edgeml.telemetry import TelemetryReporter

        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            ee_stats = {
                "total_tokens": 50,
                "early_exit_tokens": 15,
                "exit_percentage": 30.0,
            }
            reporter.report_generation_completed(
                session_id="s1",
                model_id="model-a",
                version="2.0",
                total_chunks=50,
                total_duration_ms=500.0,
                ttfc_ms=30.0,
                throughput=100.0,
                early_exit_stats=ee_stats,
            )
            time.sleep(0.15)
            reporter.close()

        assert len(sent) >= 1
        p = sent[0]
        assert p["event_type"] == "generation_completed"
        assert "early_exit" in p["metrics"]
        assert p["metrics"]["early_exit"]["early_exit_tokens"] == 15

    def test_generation_completed_no_early_exit_when_none(self):
        from edgeml.telemetry import TelemetryReporter

        sent = []

        def mock_send(client, url, headers, payload):
            sent.append(payload)

        with patch.object(TelemetryReporter, "_send", side_effect=mock_send):
            reporter = TelemetryReporter(
                api_key="key",
                api_base="https://api.test.com/api/v1",
                org_id="org-1",
                device_id="dev-1",
            )
            reporter.report_generation_completed(
                session_id="s1",
                model_id="model-a",
                version="2.0",
                total_chunks=50,
                total_duration_ms=500.0,
                ttfc_ms=30.0,
                throughput=100.0,
                early_exit_stats=None,
            )
            time.sleep(0.15)
            reporter.close()

        assert len(sent) >= 1
        p = sent[0]
        assert "early_exit" not in p["metrics"]


# ---------------------------------------------------------------------------
# InferenceMetrics — early exit fields
# ---------------------------------------------------------------------------


class TestInferenceMetricsEarlyExit:
    def test_default_values(self):
        from edgeml.serve import InferenceMetrics

        m = InferenceMetrics()
        assert m.early_exit_tokens == 0
        assert m.avg_layers_used == 0.0

    def test_with_early_exit_values(self):
        from edgeml.serve import InferenceMetrics

        m = InferenceMetrics(early_exit_tokens=15, avg_layers_used=20.5)
        assert m.early_exit_tokens == 15
        assert m.avg_layers_used == 20.5
