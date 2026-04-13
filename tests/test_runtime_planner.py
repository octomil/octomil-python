"""Tests for the runtime planner module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from octomil.runtime.planner.client import RuntimePlannerClient
from octomil.runtime.planner.device_profile import collect_device_runtime_profile
from octomil.runtime.planner.planner import RuntimePlanner
from octomil.runtime.planner.schemas import (
    DeviceRuntimeProfile,
    InstalledRuntime,
    RuntimeCandidatePlan,
    RuntimePlanResponse,
    RuntimeSelection,
)
from octomil.runtime.planner.store import RuntimePlannerStore

# ---------------------------------------------------------------------------
# Device Profile
# ---------------------------------------------------------------------------


class TestDeviceProfile:
    def test_device_profile_collects_basic_info(self):
        """Platform, arch, sdk should always be populated."""
        profile = collect_device_runtime_profile()
        assert profile.sdk == "python"
        assert profile.sdk_version != ""
        assert profile.platform != ""
        assert profile.arch != ""

    def test_device_profile_excludes_echo(self):
        """Echo engine should not appear in installed_runtimes by default."""
        profile = collect_device_runtime_profile(exclude_echo=True)
        engine_names = [r.engine for r in profile.installed_runtimes]
        assert "echo" not in engine_names

    def test_device_profile_includes_echo_when_requested(self):
        """When exclude_echo=False, echo should appear if registry has it."""
        profile = collect_device_runtime_profile(exclude_echo=False)
        engine_names = [r.engine for r in profile.installed_runtimes]
        # Echo is always available in the registry
        assert "echo" in engine_names


# ---------------------------------------------------------------------------
# SQLite Store
# ---------------------------------------------------------------------------


class TestPlannerStore:
    def test_store_put_get_plan(self, tmp_path: Path):
        """Round-trip plan cache: put then get."""
        db = tmp_path / "test.sqlite3"
        store = RuntimePlannerStore(db_path=db)

        plan_data = {"model": "gemma-2b", "candidates": [{"locality": "local", "engine": "mlx-lm"}]}
        store.put_plan(
            "key1",
            model="gemma-2b",
            capability="text",
            policy="local_first",
            plan_json=json.dumps(plan_data),
            source="server_plan",
            ttl_seconds=3600,
        )

        result = store.get_plan("key1")
        assert result is not None
        assert result["model"] == "gemma-2b"
        store.close()

    def test_store_expired_plan_returns_none(self, tmp_path: Path):
        """Expired cache entry should return None."""
        db = tmp_path / "test.sqlite3"
        store = RuntimePlannerStore(db_path=db)

        # Insert with TTL of 0 so it's already expired
        store.put_plan(
            "expired_key",
            model="gemma-2b",
            capability="text",
            policy="local_first",
            plan_json='{"model": "gemma-2b"}',
            source="server_plan",
            ttl_seconds=0,
        )

        result = store.get_plan("expired_key")
        assert result is None
        store.close()

    def test_store_put_get_benchmark(self, tmp_path: Path):
        """Round-trip benchmark cache."""
        db = tmp_path / "test.sqlite3"
        store = RuntimePlannerStore(db_path=db)

        store.put_benchmark(
            "bm_key1",
            model="gemma-2b",
            capability="text",
            engine="mlx-lm",
            tokens_per_second=42.5,
            ttft_ms=100.0,
            memory_mb=1024.0,
        )

        result = store.get_benchmark("bm_key1")
        assert result is not None
        assert result["engine"] == "mlx-lm"
        assert result["tokens_per_second"] == 42.5
        store.close()

    def test_store_corrupt_db_recreated(self, tmp_path: Path):
        """Corrupt DB should be handled gracefully by recreation."""
        db = tmp_path / "corrupt.sqlite3"
        # Write garbage to the file
        db.write_bytes(b"this is not a valid sqlite database")

        # Should not raise — it should recreate the DB
        store = RuntimePlannerStore(db_path=db)
        # Should be functional after recreation
        store.put_plan(
            "key_after_corrupt",
            model="test",
            capability="text",
            policy="local_first",
            plan_json='{"model": "test"}',
            source="test",
            ttl_seconds=3600,
        )
        result = store.get_plan("key_after_corrupt")
        assert result is not None
        store.close()

    def test_store_cache_key_deterministic(self):
        """Cache key should be deterministic and include all components."""
        key1 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", policy="local_first")
        key2 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", policy="local_first")
        key3 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", policy="cloud_first")
        assert key1 == key2
        assert key1 != key3

    def test_store_cache_key_includes_required_fields(self):
        """Different input fields produce different keys."""
        key_a = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", sdk_version="4.5.0")
        key_b = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", sdk_version="4.6.0")
        assert key_a != key_b


# ---------------------------------------------------------------------------
# Planner Client
# ---------------------------------------------------------------------------


class TestPlannerClient:
    def test_client_fetch_plan_returns_none_on_failure(self):
        """HTTP errors should return None, not raise."""
        client = RuntimePlannerClient(base_url="http://localhost:1", api_key="test")
        device = DeviceRuntimeProfile(
            sdk="python",
            sdk_version="4.6.0",
            platform="Darwin",
            arch="arm64",
        )
        result = client.fetch_plan(
            model="gemma-2b",
            capability="text",
            device=device,
        )
        assert result is None

    def test_client_upload_benchmark_returns_false_on_failure(self):
        """HTTP errors should return False."""
        client = RuntimePlannerClient(base_url="http://localhost:1", api_key="test")
        result = client.upload_benchmark({"model": "gemma-2b", "engine": "mlx-lm"})
        assert result is False

    def test_client_fetch_plan_parses_response(self):
        """Mock a successful HTTP response and verify parsing."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={
                "model": "gemma-2b",
                "capability": "text",
                "policy": "local_first",
                "candidates": [
                    {
                        "locality": "local",
                        "priority": 1,
                        "confidence": 0.95,
                        "reason": "mlx available on Apple Silicon",
                        "engine": "mlx-lm",
                        "benchmark_required": False,
                    }
                ],
                "fallback_candidates": [],
                "plan_ttl_seconds": 86400,
                "server_generated_at": "2026-04-12T00:00:00Z",
            },
            request=httpx.Request("POST", "http://test/api/v2/runtime/plan"),
        )

        client = RuntimePlannerClient(base_url="http://test", api_key="key")
        device = DeviceRuntimeProfile(
            sdk="python",
            sdk_version="4.6.0",
            platform="Darwin",
            arch="arm64",
        )

        with patch("httpx.Client") as mock_client_cls:
            mock_http = MagicMock()
            mock_http.__enter__ = MagicMock(return_value=mock_http)
            mock_http.__exit__ = MagicMock(return_value=False)
            mock_http.post.return_value = mock_response
            mock_client_cls.return_value = mock_http

            plan = client.fetch_plan(model="gemma-2b", capability="text", device=device)

        assert plan is not None
        assert plan.model == "gemma-2b"
        assert len(plan.candidates) == 1
        assert plan.candidates[0].engine == "mlx-lm"
        assert plan.candidates[0].locality == "local"


# ---------------------------------------------------------------------------
# RuntimePlanner
# ---------------------------------------------------------------------------


class TestRuntimePlanner:
    def _make_planner(self, tmp_path: Path, *, client: RuntimePlannerClient | None = None) -> RuntimePlanner:
        db = tmp_path / "planner.sqlite3"
        store = RuntimePlannerStore(db_path=db)
        return RuntimePlanner(store=store, client=client)

    def test_planner_local_only_no_cloud(self, tmp_path: Path):
        """Private/local_only policy should return local selection, never cloud."""
        planner = self._make_planner(tmp_path)

        # Mock the local engine selection to return a local engine
        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile:
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
                installed_runtimes=[InstalledRuntime(engine="mlx-lm")],
            )
            with patch("octomil.runtime.planner.planner.RuntimePlanner._resolve_locally") as mock_local:
                mock_local.return_value = RuntimeSelection(
                    locality="local",
                    engine="mlx-lm",
                    source="local_benchmark",
                    reason="local engine selected",
                )
                result = planner.resolve(
                    model="gemma-2b",
                    capability="text",
                    routing_policy="private",
                )
        assert result.locality == "local"

    def test_planner_cloud_only_no_local(self, tmp_path: Path):
        """cloud_only policy should return cloud selection."""
        planner = self._make_planner(tmp_path)

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile:
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
            )
            result = planner.resolve(
                model="gemma-2b",
                capability="text",
                routing_policy="cloud_only",
            )
        assert result.locality == "cloud"
        assert "cloud_only" in result.reason

    def test_planner_uses_server_plan_when_available(self, tmp_path: Path):
        """When server returns a plan, the planner should use it."""
        mock_client = MagicMock(spec=RuntimePlannerClient)
        mock_client.fetch_plan.return_value = RuntimePlanResponse(
            model="gemma-2b",
            capability="text",
            policy="local_first",
            candidates=[
                RuntimeCandidatePlan(
                    locality="local",
                    priority=1,
                    confidence=0.95,
                    reason="server recommended mlx-lm",
                    engine="mlx-lm",
                )
            ],
        )

        planner = self._make_planner(tmp_path, client=mock_client)

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile:
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
                installed_runtimes=[InstalledRuntime(engine="mlx-lm")],
            )
            result = planner.resolve(
                model="gemma-2b",
                capability="text",
                routing_policy="local_first",
            )

        assert result.source == "server_plan"
        assert result.engine == "mlx-lm"
        mock_client.fetch_plan.assert_called_once()

    def test_planner_falls_back_to_local_when_server_unreachable(self, tmp_path: Path):
        """When server is unreachable, planner should still work via local path."""
        mock_client = MagicMock(spec=RuntimePlannerClient)
        mock_client.fetch_plan.return_value = None  # Server unreachable

        planner = self._make_planner(tmp_path, client=mock_client)

        with (
            patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile,
            patch("octomil.runtime.planner.planner.RuntimePlanner._resolve_locally") as mock_local,
        ):
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
            )
            mock_local.return_value = RuntimeSelection(
                locality="local",
                engine="llama.cpp",
                source="local_benchmark",
                reason="fallback to local",
            )
            result = planner.resolve(
                model="gemma-2b",
                capability="text",
                routing_policy="local_first",
            )

        assert result.locality == "local"
        assert result.source == "local_benchmark"

    def test_planner_private_no_telemetry_upload(self, tmp_path: Path):
        """Private policy should never call fetch_plan or upload_benchmark."""
        mock_client = MagicMock(spec=RuntimePlannerClient)
        planner = self._make_planner(tmp_path, client=mock_client)

        with (
            patch("octomil.runtime.planner.planner.collect_device_runtime_profile") as mock_profile,
            patch("octomil.runtime.planner.planner.RuntimePlanner._resolve_locally") as mock_local,
        ):
            mock_profile.return_value = DeviceRuntimeProfile(
                sdk="python",
                sdk_version="4.6.0",
                platform="Darwin",
                arch="arm64",
            )
            mock_local.return_value = RuntimeSelection(
                locality="local",
                engine="mlx-lm",
                source="local_benchmark",
                reason="private local selection",
            )
            planner.resolve(
                model="gemma-2b",
                capability="text",
                routing_policy="private",
            )

        # With private policy, fetch_plan should NOT be called
        mock_client.fetch_plan.assert_not_called()

    def test_planner_caches_server_plan(self, tmp_path: Path):
        """After fetching from server, subsequent calls should use cache."""
        mock_client = MagicMock(spec=RuntimePlannerClient)
        server_plan = RuntimePlanResponse(
            model="gemma-2b",
            capability="text",
            policy="local_first",
            candidates=[
                RuntimeCandidatePlan(
                    locality="local",
                    priority=1,
                    confidence=0.95,
                    reason="cached hit",
                    engine="mlx-lm",
                )
            ],
            plan_ttl_seconds=3600,
        )
        mock_client.fetch_plan.return_value = server_plan

        planner = self._make_planner(tmp_path, client=mock_client)

        device = DeviceRuntimeProfile(
            sdk="python",
            sdk_version="4.6.0",
            platform="Darwin",
            arch="arm64",
            installed_runtimes=[InstalledRuntime(engine="mlx-lm")],
        )

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile", return_value=device):
            # First call fetches from server
            result1 = planner.resolve(model="gemma-2b", capability="text")
            assert result1.source == "server_plan"
            assert mock_client.fetch_plan.call_count == 1

            # Second call should use cache
            result2 = planner.resolve(model="gemma-2b", capability="text")
            assert result2.source == "cache"
            # fetch_plan should NOT have been called again
            assert mock_client.fetch_plan.call_count == 1


# ---------------------------------------------------------------------------
# Engine Bridge Integration
# ---------------------------------------------------------------------------


class TestEngineBridgeIntegration:
    def test_planner_bridge_disabled_returns_none(self):
        """When OCTOMIL_RUNTIME_PLANNER_CACHE=0, should return None."""
        from octomil.runtime.core.engine_bridge import _select_engine_with_planner

        with patch.dict("os.environ", {"OCTOMIL_RUNTIME_PLANNER_CACHE": "0"}):
            result = _select_engine_with_planner("gemma-2b", "text", "local_first")
        assert result is None

    def test_planner_bridge_returns_none_on_exception(self):
        """Any exception in planner should gracefully return None."""
        from octomil.runtime.core.engine_bridge import _select_engine_with_planner

        with patch("octomil.runtime.planner.planner.RuntimePlanner.resolve", side_effect=RuntimeError("boom")):
            result = _select_engine_with_planner("gemma-2b", "text", "local_first")
        assert result is None
