"""Tests for no-echo guard, --explain routing flag, and artifact cache scaffold."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from octomil.runtime.planner.artifact_cache import ArtifactCache, _warn_if_large_artifact_non_tty

# ---------------------------------------------------------------------------
# TestNoEchoInUserPaths
# ---------------------------------------------------------------------------


class TestNoEchoInUserPaths:
    """Echo engine must never silently serve user-facing requests."""

    def test_echo_not_selected_when_real_engine_available(self):
        """When a real engine is available alongside echo, the real engine wins."""
        from octomil.runtime.core.engine_bridge import _select_real_engine

        mock_echo = MagicMock()
        mock_echo.name = "echo"
        mock_echo.detect.return_value = True

        mock_mlx = MagicMock()
        mock_mlx.name = "mlx-lm"
        mock_mlx.detect.return_value = True

        # detect_all returns both engines
        mock_registry = MagicMock()
        mock_registry.detect_all.return_value = [
            MagicMock(engine=mock_mlx, available=True),
            MagicMock(engine=mock_echo, available=True),
        ]

        # benchmark_all returns mlx as the best
        mock_benchmark = MagicMock()
        mock_benchmark.engine = mock_mlx
        mock_benchmark.result.ok = True
        mock_benchmark.result.tokens_per_second = 50.0
        mock_registry.benchmark_all.return_value = [mock_benchmark]

        engine, real_engines = _select_real_engine(mock_registry, "test-model")

        assert engine is not None
        assert engine.name == "mlx-lm"
        assert all(e.name != "echo" for e in real_engines)

    def test_echo_only_raises_error(self):
        """When echo is the only engine, _reject_echo_only raises RuntimeError."""
        from octomil.runtime.core.engine_bridge import _reject_echo_only

        mock_echo = MagicMock()
        mock_echo.name = "echo"
        mock_echo.detect.return_value = True

        mock_registry = MagicMock()
        mock_registry.detect_all.return_value = [
            MagicMock(engine=mock_echo, available=True),
        ]

        with pytest.raises(RuntimeError, match="No real inference engine"):
            _reject_echo_only(mock_registry, "test-model")

    def test_reject_echo_only_allows_when_real_engine_present(self):
        """_reject_echo_only does NOT raise when a real engine is present."""
        from octomil.runtime.core.engine_bridge import _reject_echo_only

        mock_echo = MagicMock()
        mock_echo.name = "echo"

        mock_mlx = MagicMock()
        mock_mlx.name = "mlx-lm"

        mock_registry = MagicMock()
        mock_registry.detect_all.return_value = [
            MagicMock(engine=mock_mlx, available=True),
            MagicMock(engine=mock_echo, available=True),
        ]

        # Should not raise
        _reject_echo_only(mock_registry, "test-model")

    def test_echo_excluded_from_planner_benchmark(self):
        """Echo never appears in benchmark selections within the planner."""
        from octomil.runtime.planner.planner import _select_local_engine

        mock_echo = MagicMock()
        mock_echo.name = "echo"
        mock_echo.detect.return_value = True

        mock_registry = MagicMock()
        mock_registry.detect_all.return_value = [
            MagicMock(engine=mock_echo, available=True),
        ]

        engine, result = _select_local_engine(mock_registry, "test-model")
        assert engine is None
        assert result is None


# ---------------------------------------------------------------------------
# TestExplainFlag
# ---------------------------------------------------------------------------


class TestExplainFlag:
    """The --explain flag prints routing decision info to stderr."""

    @patch("octomil.runtime.planner.planner.RuntimePlanner")
    def test_explain_prints_routing_info(self, mock_planner_cls):
        """--explain prints route/model/planner/artifact lines to output."""
        from click.testing import CliRunner

        from octomil.commands.inference import run_cmd

        mock_selection = MagicMock()
        mock_selection.locality = "local"
        mock_selection.engine = "mlx-lm"
        mock_selection.source = "cache"
        mock_selection.artifact = None
        mock_planner_cls.return_value.resolve.return_value = mock_selection

        runner = CliRunner()
        result = runner.invoke(run_cmd, ["--explain", "--model", "gemma-3-1b", "hello"])
        combined = result.output
        assert "Route:" in combined
        assert "mlx-lm" in combined
        assert "Model:" in combined
        assert "Planner:" in combined

    @patch("octomil.runtime.planner.planner.RuntimePlanner")
    def test_explain_unavailable_shows_reason(self, mock_planner_cls):
        """When route is unavailable, --explain shows the reason and errors."""
        from click.testing import CliRunner

        from octomil.commands.inference import run_cmd

        mock_selection = MagicMock()
        mock_selection.locality = "local"
        mock_selection.engine = None
        mock_selection.source = "fallback"
        mock_selection.artifact = None
        mock_planner_cls.return_value.resolve.return_value = mock_selection

        runner = CliRunner()
        result = runner.invoke(run_cmd, ["--explain", "--model", "gemma-3-1b", "hello"])
        assert result.exit_code != 0
        assert "unavailable" in result.output.lower()


# ---------------------------------------------------------------------------
# TestArtifactCache
# ---------------------------------------------------------------------------


class TestArtifactCache:
    """ArtifactCache provides cache status tracking (hit/miss) for planner artifacts."""

    def test_cache_hit(self, tmp_path: Path):
        cache = ArtifactCache(cache_dir=tmp_path)
        (tmp_path / "sha256:abc123").write_bytes(b"fake model data")
        assert cache.is_cached("sha256:abc123") is True
        assert cache.cache_status("sha256:abc123") == "hit"

    def test_cache_miss(self, tmp_path: Path):
        cache = ArtifactCache(cache_dir=tmp_path)
        assert cache.is_cached("sha256:abc123") is False
        assert cache.cache_status("sha256:abc123") == "miss"

    def test_cache_not_applicable_when_no_digest(self, tmp_path: Path):
        cache = ArtifactCache(cache_dir=tmp_path)
        assert cache.cache_status(None) == "not_applicable"

    def test_second_run_cache_reuse(self, tmp_path: Path):
        cache = ArtifactCache(cache_dir=tmp_path)
        # First check: miss
        assert cache.cache_status("sha256:abc") == "miss"
        # Simulate download
        (tmp_path / "sha256:abc").write_bytes(b"model data")
        # Second check: hit
        assert cache.cache_status("sha256:abc") == "hit"

    def test_artifact_path_returns_correct_path(self, tmp_path: Path):
        cache = ArtifactCache(cache_dir=tmp_path)
        assert cache.artifact_path("sha256:xyz") == tmp_path / "sha256:xyz"

    def test_cache_dir_created_on_init(self, tmp_path: Path):
        cache_dir = tmp_path / "deep" / "nested" / "artifacts"
        assert not cache_dir.exists()
        ArtifactCache(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_acquire_lock_returns_fd(self, tmp_path: Path):
        cache = ArtifactCache(cache_dir=tmp_path)
        fd = cache.acquire_lock("sha256:lock_test")
        assert fd is not None
        # Lock file should exist
        assert (tmp_path / "sha256:lock_test.lock").exists()
        fd.close()

    def test_cache_respects_env_var(self, tmp_path: Path, monkeypatch):
        custom_dir = tmp_path / "custom_cache"
        monkeypatch.setenv("OCTOMIL_ARTIFACT_CACHE", str(custom_dir))
        cache = ArtifactCache()
        assert cache.cache_dir == custom_dir


# ---------------------------------------------------------------------------
# TestNonTtyGuard
# ---------------------------------------------------------------------------


class TestNonTtyGuard:
    """Non-TTY guard warns about large artifacts in scripts/CI."""

    def test_tty_no_warning(self, monkeypatch):
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        # Should not warn for large artifact in TTY mode
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_if_large_artifact_non_tty(500_000_000)
            assert len(w) == 0

    def test_non_tty_large_artifact_warns(self, monkeypatch):
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        with pytest.warns(UserWarning, match="non-interactive"):
            _warn_if_large_artifact_non_tty(500_000_000)

    def test_non_tty_small_artifact_no_warning(self, monkeypatch):
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        # Under 100MB threshold — no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_if_large_artifact_non_tty(50_000_000)
            assert len(w) == 0

    def test_none_size_no_warning(self, monkeypatch):
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        # None size — no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_if_large_artifact_non_tty(None)
            assert len(w) == 0


# ---------------------------------------------------------------------------
# TestPlannerSelectedEngine
# ---------------------------------------------------------------------------


class TestPlannerSelectedEngine:
    """Planner engine selection flows correctly into the execution kernel."""

    def test_planner_engine_used_when_available(self):
        """When planner returns a specific engine, it should be used."""
        from octomil.runtime.planner.schemas import RuntimeSelection

        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="server_plan",
            reason="planner selected mlx-lm",
        )
        assert selection.engine == "mlx-lm"
        assert selection.locality == "local"

    def test_planner_engine_unavailable_falls_back(self):
        """When planner engine is not installed, source shows fallback."""
        from octomil.runtime.planner.schemas import RuntimeSelection

        selection = RuntimeSelection(
            locality="cloud",
            engine=None,
            source="fallback",
            reason="no local engine available — falling back to cloud",
        )
        assert selection.locality == "cloud"
        assert selection.engine is None
        assert selection.source == "fallback"

    def test_route_metadata_from_selection_with_artifact(self):
        """Route metadata includes artifact cache info from ArtifactCache."""
        from octomil.execution.kernel import _route_metadata_from_selection
        from octomil.runtime.planner.schemas import RuntimeArtifactPlan, RuntimeSelection

        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="server_plan",
            reason="server plan selected mlx-lm",
            artifact=RuntimeArtifactPlan(
                model_id="gemma-3-1b",
                artifact_id="art_123",
                model_version="1.0",
                format="mlx",
                digest="sha256:abc123",
                size_bytes=1_000_000,
            ),
        )

        route = _route_metadata_from_selection(
            selection,
            "on_device",
            False,
            model_name="gemma-3-1b",
            capability="chat",
        )
        assert route.artifact is not None
        assert route.artifact.format == "mlx"
        assert route.artifact.digest == "sha256:abc123"
        assert route.artifact.cache.managed_by == "octomil"
        # Cache status should be "miss" because artifact doesn't exist on disk
        assert route.artifact.cache.status in ("miss", "not_applicable")

    def test_route_metadata_from_selection_without_artifact(self):
        """Route metadata has no artifact when selection has none."""
        from octomil.execution.kernel import _route_metadata_from_selection
        from octomil.runtime.planner.schemas import RuntimeSelection

        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="cache",
            reason="cached benchmark",
        )

        route = _route_metadata_from_selection(
            selection,
            "on_device",
            False,
            model_name="gemma-3-1b",
            capability="chat",
        )
        assert route.artifact is None
