"""Integration tests for runtime planner wired into the execution kernel.

Covers:
- Planner request shape (correct device profile fields)
- Cache hit returns cached plan without server call
- Cache miss fetches from server
- Stale cache used as fallback with warning
- Policy routing matrix: private, local_only, cloud_only, local_first,
  cloud_first, performance_first
- Benchmark upload sanitization (banned keys rejected)
- Route metadata in ExecutionResult (locality, engine, planner_source)
- --json CLI output includes route metadata
- Existing serve behaviour is not broken by planner integration
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from octomil.config.local import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_TRANSCRIPTION,
    CapabilityDefault,
    CloudProfile,
    LoadedConfigSet,
    LocalOctomilConfig,
)
from octomil.execution.kernel import (
    ExecutionKernel,
    ExecutionResult,
    FallbackInfo,
    PlannerInfo,
    RouteExecution,
    RouteMetadata,
    RouteModel,
    RouteModelRequested,
    RouteReason,
    _resolve_planner_selection,
    _route_metadata_from_selection,
    _sanitize_benchmark_payload,
    _upload_benchmark_async,
)
from octomil.runtime.core.types import RuntimeChunk, RuntimeResponse, RuntimeUsage
from octomil.runtime.planner.client import RuntimePlannerClient
from octomil.runtime.planner.schemas import (
    DeviceRuntimeProfile,
    InstalledRuntime,
    RuntimeCandidatePlan,
    RuntimePlanResponse,
    RuntimeSelection,
)
from octomil.runtime.planner.store import RuntimePlannerStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_kernel(
    model: str = "test-model",
    policy: str = "local_first",
    capability: str = CAPABILITY_CHAT,
) -> ExecutionKernel:
    config = LocalOctomilConfig(
        capabilities={
            CAPABILITY_CHAT: CapabilityDefault(model=model, policy=policy),
            CAPABILITY_EMBEDDING: CapabilityDefault(model="embed-model", policy=policy),
            CAPABILITY_TRANSCRIPTION: CapabilityDefault(model="whisper-test", policy=policy),
        }
    )
    return ExecutionKernel(config_set=LoadedConfigSet(project=config))


def _mock_runtime_response(text: str = "Hello!") -> RuntimeResponse:
    return RuntimeResponse(
        text=text,
        usage=RuntimeUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
    )


# ---------------------------------------------------------------------------
# Planner request shape
# ---------------------------------------------------------------------------


class TestPlannerRequestShape:
    def test_device_profile_has_required_fields(self):
        """Device profile sent to planner must include sdk, sdk_version,
        platform, arch, os_version, and installed_runtimes."""
        with patch("octomil.runtime.planner.device_profile._detect_installed_runtimes") as mock_detect:
            mock_detect.return_value = [
                InstalledRuntime(engine="mlx-lm", version="0.16.0"),
            ]
            from octomil.runtime.planner.device_profile import collect_device_runtime_profile

            profile = collect_device_runtime_profile()

        assert profile.sdk == "python"
        assert isinstance(profile.sdk_version, str) and profile.sdk_version
        assert isinstance(profile.platform, str) and profile.platform
        assert isinstance(profile.arch, str) and profile.arch
        assert profile.os_version is not None
        assert isinstance(profile.installed_runtimes, list)

    def test_planner_selection_returns_runtime_selection_or_none(self):
        """_resolve_planner_selection should return RuntimeSelection or None,
        never raise."""
        with patch("octomil.runtime.planner.planner.RuntimePlanner.resolve") as mock_resolve:
            mock_resolve.return_value = RuntimeSelection(
                locality="local",
                engine="mlx-lm",
                source="cache",
                reason="cached plan",
            )
            result = _resolve_planner_selection("gemma-2b", CAPABILITY_CHAT, "local_first")
            assert result is not None
            assert result.engine == "mlx-lm"

    def test_planner_selection_returns_none_on_exception(self):
        """_resolve_planner_selection must return None when planner raises."""
        with patch(
            "octomil.runtime.planner.planner.RuntimePlanner.resolve",
            side_effect=RuntimeError("boom"),
        ):
            result = _resolve_planner_selection("gemma-2b", CAPABILITY_CHAT, "local_first")
            assert result is None

    def test_planner_disabled_returns_none(self, monkeypatch):
        """When OCTOMIL_RUNTIME_PLANNER_CACHE=0, _resolve_planner_selection returns None."""
        monkeypatch.setenv("OCTOMIL_RUNTIME_PLANNER_CACHE", "0")
        result = _resolve_planner_selection("gemma-2b", CAPABILITY_CHAT, "local_first")
        assert result is None

    def test_plan_response_parser_preserves_gates_and_fallback_allowed(self):
        """Server-emitted gates and fallback_allowed must survive client parsing."""
        from octomil.runtime.planner.client import _parse_plan_response

        plan = _parse_plan_response(
            {
                "model": "gemma3-1b",
                "capability": "responses",
                "policy": "local_first",
                "candidates": [
                    {
                        "locality": "local",
                        "engine": "mlx-lm",
                        "priority": 0,
                        "confidence": 0.9,
                        "reason": "fast local path",
                        "gates": [
                            {
                                "code": "min_tokens_per_second",
                                "required": True,
                                "threshold_number": 12.5,
                                "source": "server",
                            }
                        ],
                    }
                ],
                "fallback_candidates": [{"locality": "cloud", "priority": 1, "confidence": 0.5, "reason": "hosted"}],
                "fallback_allowed": False,
                "server_generated_at": "2026-04-20T00:00:00Z",
            }
        )

        assert plan.fallback_allowed is False
        assert len(plan.candidates[0].gates) == 1
        assert plan.candidates[0].gates[0].code == "min_tokens_per_second"
        assert plan.candidates[0].gates[0].threshold_number == 12.5


# ---------------------------------------------------------------------------
# Plan cache hit / miss / stale
# ---------------------------------------------------------------------------


class TestPlanCache:
    def test_cache_hit_returns_cached_plan_no_server_call(self, tmp_path: Path):
        """When cache has a valid plan, fetch_plan should not be called."""
        from octomil.runtime.planner.planner import RuntimePlanner

        db = tmp_path / "test.sqlite3"
        store = RuntimePlannerStore(db_path=db)
        mock_client = MagicMock(spec=RuntimePlannerClient)

        # Pre-populate cache with a server plan
        mock_client.fetch_plan.return_value = RuntimePlanResponse(
            model="gemma-2b",
            capability="responses",
            policy="local_first",
            candidates=[
                RuntimeCandidatePlan(
                    locality="local",
                    priority=1,
                    confidence=0.95,
                    reason="cached",
                    engine="mlx-lm",
                )
            ],
            plan_ttl_seconds=3600,
        )

        planner = RuntimePlanner(store=store, client=mock_client)
        device = DeviceRuntimeProfile(
            sdk="python",
            sdk_version="4.6.0",
            platform="Darwin",
            arch="arm64",
            installed_runtimes=[InstalledRuntime(engine="mlx-lm")],
        )

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile", return_value=device):
            # First call populates cache
            r1 = planner.resolve(model="gemma-2b", capability="responses")
            assert r1.source == "server"
            assert mock_client.fetch_plan.call_count == 1

            # Second call should use cache, no server call
            r2 = planner.resolve(model="gemma-2b", capability="responses")
            assert r2.source == "cache"
            assert mock_client.fetch_plan.call_count == 1

        store.close()

    def test_cache_miss_fetches_from_server(self, tmp_path: Path):
        """Empty cache should trigger a server fetch."""
        from octomil.runtime.planner.planner import RuntimePlanner

        db = tmp_path / "test.sqlite3"
        store = RuntimePlannerStore(db_path=db)
        mock_client = MagicMock(spec=RuntimePlannerClient)
        mock_client.fetch_plan.return_value = RuntimePlanResponse(
            model="gemma-2b",
            capability="responses",
            policy="local_first",
            candidates=[
                RuntimeCandidatePlan(
                    locality="local",
                    priority=1,
                    confidence=0.9,
                    reason="server plan",
                    engine="llama.cpp",
                )
            ],
        )

        planner = RuntimePlanner(store=store, client=mock_client)
        device = DeviceRuntimeProfile(
            sdk="python",
            sdk_version="4.6.0",
            platform="Linux",
            arch="x86_64",
            installed_runtimes=[InstalledRuntime(engine="llama.cpp")],
        )

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile", return_value=device):
            result = planner.resolve(model="gemma-2b", capability="responses")

        assert result.source == "server"
        assert result.engine == "llama.cpp"
        mock_client.fetch_plan.assert_called_once()
        store.close()

    def test_stale_cache_used_as_fallback_when_server_unavailable(self, tmp_path: Path):
        """When server is unreachable and cache is expired, fallback to local selection."""
        from octomil.runtime.planner.planner import RuntimePlanner

        db = tmp_path / "test.sqlite3"
        store = RuntimePlannerStore(db_path=db)
        mock_client = MagicMock(spec=RuntimePlannerClient)

        # First call: server returns plan with short TTL
        mock_client.fetch_plan.return_value = RuntimePlanResponse(
            model="gemma-2b",
            capability="responses",
            policy="local_first",
            candidates=[
                RuntimeCandidatePlan(
                    locality="local",
                    priority=1,
                    confidence=0.9,
                    reason="short ttl",
                    engine="mlx-lm",
                )
            ],
            plan_ttl_seconds=0,  # immediately expired
        )

        planner = RuntimePlanner(store=store, client=mock_client)
        device = DeviceRuntimeProfile(
            sdk="python",
            sdk_version="4.6.0",
            platform="Darwin",
            arch="arm64",
            installed_runtimes=[InstalledRuntime(engine="mlx-lm")],
        )

        with patch("octomil.runtime.planner.planner.collect_device_runtime_profile", return_value=device):
            # First call populates cache (but TTL=0 means it's immediately stale)
            r1 = planner.resolve(model="gemma-2b", capability="responses")
            assert r1.source == "server"

            # Now server is unavailable
            mock_client.fetch_plan.return_value = None

            # Second call: cache is stale, server is down -> falls back to local
            with patch("octomil.runtime.planner.planner.RuntimePlanner._resolve_locally") as mock_local:
                mock_local.return_value = RuntimeSelection(
                    locality="local",
                    engine="mlx-lm",
                    source="fallback",
                    reason="stale cache fallback",
                )
                r2 = planner.resolve(model="gemma-2b", capability="responses")
                assert r2.source == "fallback"

        store.close()


# ---------------------------------------------------------------------------
# Policy routing matrix
# ---------------------------------------------------------------------------


class TestPolicyRoutingMatrix:
    """Verify that each policy preset routes correctly through the planner
    and execution kernel."""

    def test_private_routes_local_only(self):
        """private policy: local only, never cloud, never telemetry."""
        selection = RuntimeSelection(locality="local", engine="mlx-lm", source="cache", reason="private")
        route = _route_metadata_from_selection(selection, "on_device", False)
        assert route.execution is not None
        assert route.execution.locality == "local"
        assert route.execution.engine == "mlx-lm"

    def test_private_never_contacts_server(self, tmp_path: Path):
        """private policy should never call fetch_plan."""
        from octomil.runtime.planner.planner import RuntimePlanner

        mock_client = MagicMock(spec=RuntimePlannerClient)
        store = RuntimePlannerStore(db_path=tmp_path / "test.sqlite3")
        planner = RuntimePlanner(store=store, client=mock_client)
        device = DeviceRuntimeProfile(sdk="python", sdk_version="4.6.0", platform="Darwin", arch="arm64")

        with (
            patch("octomil.runtime.planner.planner.collect_device_runtime_profile", return_value=device),
            patch("octomil.runtime.planner.planner.RuntimePlanner._resolve_locally") as mock_local,
        ):
            mock_local.return_value = RuntimeSelection(
                locality="local", engine="mlx-lm", source="fallback", reason="private"
            )
            planner.resolve(model="gemma-2b", capability="responses", routing_policy="private")

        mock_client.fetch_plan.assert_not_called()
        mock_client.upload_benchmark.assert_not_called()
        store.close()

    def test_local_only_routes_local(self):
        """local_only should behave identically to private for routing."""
        selection = RuntimeSelection(locality="local", engine="llama.cpp", source="cache", reason="local_only")
        route = _route_metadata_from_selection(selection, "on_device", False)
        assert route.execution is not None
        assert route.execution.locality == "local"

    def test_cloud_only_routes_cloud(self):
        """cloud_only: cloud only, never local."""
        selection = RuntimeSelection(locality="cloud", engine=None, source="fallback", reason="cloud_only")
        route = _route_metadata_from_selection(selection, "cloud", False)
        assert route.execution is not None
        assert route.execution.locality == "cloud"
        assert route.planner.source == "offline"

    def test_local_first_prefers_local_with_cloud_fallback(self):
        """local_first: local primary, cloud fallback."""
        selection = RuntimeSelection(locality="local", engine="mlx-lm", source="server", reason="local_first")
        route = _route_metadata_from_selection(selection, "on_device", False)
        assert route.execution is not None
        assert route.execution.locality == "local"
        assert route.planner.source == "server"

    def test_local_first_falls_back_to_cloud(self):
        """When local is unavailable, local_first should show fallback.used."""
        selection = RuntimeSelection(locality="cloud", engine=None, source="fallback", reason="no local engine")
        route = _route_metadata_from_selection(selection, "cloud", True)
        assert route.execution is not None
        assert route.execution.locality == "cloud"
        assert route.fallback.used is True

    def test_cloud_first_prefers_cloud_with_local_fallback(self):
        """cloud_first: cloud primary, local fallback."""
        selection = RuntimeSelection(locality="cloud", engine=None, source="server", reason="cloud_first")
        route = _route_metadata_from_selection(selection, "cloud", False)
        assert route.execution is not None
        assert route.execution.locality == "cloud"
        assert route.planner.source == "server"

    def test_performance_first_follows_planner_order(self):
        """performance_first: follow planner candidate order."""
        selection = RuntimeSelection(locality="local", engine="mlx-lm", source="offline", reason="fastest")
        route = _route_metadata_from_selection(selection, "on_device", False)
        assert route.execution is not None
        assert route.execution.locality == "local"
        assert route.planner.source == "offline"


# ---------------------------------------------------------------------------
# Benchmark upload sanitization
# ---------------------------------------------------------------------------


class TestBenchmarkSanitization:
    def test_banned_keys_are_stripped(self):
        """Payload keys containing prompt, input, output, audio, file, text,
        content, or messages must be removed."""
        payload = {
            "model": "gemma-2b",
            "engine": "mlx-lm",
            "tokens_per_second": 42.0,
            "prompt": "tell me a secret",
            "input": "private data",
            "output": "should not leak",
            "audio": b"raw bytes",
            "audio_data": b"more bytes",
            "file": "/path/to/file",
            "file_path": "/path/to/file",
            "text": "should not leak",
            "content": "should not leak",
            "messages": [{"role": "user", "content": "private"}],
            "response": "leaked response",
        }
        clean = _sanitize_benchmark_payload(payload)
        assert "prompt" not in clean
        assert "input" not in clean
        assert "output" not in clean
        assert "audio" not in clean
        assert "audio_data" not in clean
        assert "file" not in clean
        assert "file_path" not in clean
        assert "text" not in clean
        assert "content" not in clean
        assert "messages" not in clean
        assert "response" not in clean
        # Safe keys preserved
        assert clean["model"] == "gemma-2b"
        assert clean["engine"] == "mlx-lm"
        assert clean["tokens_per_second"] == 42.0

    def test_empty_payload_returns_empty(self):
        assert _sanitize_benchmark_payload({}) == {}

    def test_safe_payload_unchanged(self):
        payload = {
            "model": "gemma-2b",
            "engine": "mlx-lm",
            "success": True,
            "tokens_per_second": 42.0,
            "ttft_ms": 120.0,
            "latency_ms": 500.0,
            "peak_memory_bytes": 1024000,
        }
        assert _sanitize_benchmark_payload(payload) == payload

    def test_private_policy_skips_upload(self, monkeypatch):
        """_upload_benchmark_async should be a no-op for private policy."""
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        with patch("octomil.runtime.planner.client.RuntimePlannerClient.upload_benchmark") as mock_upload:
            _upload_benchmark_async(
                model="gemma-2b",
                capability=CAPABILITY_CHAT,
                engine="mlx-lm",
                policy_preset="private",
                tokens_per_second=42.0,
            )
            mock_upload.assert_not_called()

    def test_local_only_policy_skips_upload(self, monkeypatch):
        """local_only policy must also skip benchmark upload."""
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        with patch("octomil.runtime.planner.client.RuntimePlannerClient.upload_benchmark") as mock_upload:
            _upload_benchmark_async(
                model="gemma-2b",
                capability=CAPABILITY_CHAT,
                engine="mlx-lm",
                policy_preset="local_only",
                tokens_per_second=42.0,
            )
            mock_upload.assert_not_called()

    def test_no_credentials_skips_upload(self, monkeypatch):
        """No API key configured should skip upload."""
        monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
        monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
        with patch("octomil.runtime.planner.client.RuntimePlannerClient.upload_benchmark") as mock_upload:
            _upload_benchmark_async(
                model="gemma-2b",
                capability=CAPABILITY_CHAT,
                engine="mlx-lm",
                policy_preset="local_first",
                tokens_per_second=42.0,
            )
            mock_upload.assert_not_called()


# ---------------------------------------------------------------------------
# Route metadata in ExecutionResult
# ---------------------------------------------------------------------------


class TestRouteMetadata:
    def test_route_metadata_from_server_plan(self):
        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="server",
            reason="server recommended mlx-lm",
        )
        route = _route_metadata_from_selection(selection, "on_device", False)
        assert route.execution is not None
        assert route.execution.locality == "local"
        assert route.execution.engine == "mlx-lm"
        assert route.planner.source == "server"
        assert route.fallback.used is False
        assert "mlx-lm" in route.reason.message

    def test_route_metadata_from_cache(self):
        selection = RuntimeSelection(
            locality="local",
            engine="llama.cpp",
            source="cache",
            reason="cached plan",
        )
        route = _route_metadata_from_selection(selection, "on_device", False)
        assert route.planner.source == "cache"

    def test_route_metadata_from_offline(self):
        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="offline",
            reason="local benchmark",
        )
        route = _route_metadata_from_selection(selection, "on_device", False)
        assert route.planner.source == "offline"

    def test_route_metadata_from_fallback(self):
        selection = RuntimeSelection(
            locality="cloud",
            engine=None,
            source="fallback",
            reason="no local engine",
        )
        route = _route_metadata_from_selection(selection, "cloud", True)
        assert route.planner.source == "offline"
        assert route.fallback.used is True

    def test_route_metadata_when_planner_unavailable(self):
        route = _route_metadata_from_selection(None, "on_device", False)
        assert route.planner.source == "offline"
        assert route.reason.message == "planner not available"

    def test_on_device_never_in_public_route_metadata(self):
        """Public route metadata must use 'local', never 'on_device'."""
        selection = RuntimeSelection(locality="local", engine="mlx-lm", source="cache", reason="test")
        route = _route_metadata_from_selection(selection, "on_device", False)
        assert route.execution is not None
        assert route.execution.locality == "local"
        assert "on_device" not in route.execution.locality

        # Also check when planner is unavailable
        route2 = _route_metadata_from_selection(None, "on_device", False)
        assert route2.execution is not None
        assert route2.execution.locality == "local"

    def test_execution_mode_set_correctly(self):
        """execution.mode must be sdk_runtime for local, hosted_gateway for cloud."""
        local_selection = RuntimeSelection(locality="local", engine="mlx-lm", source="cache", reason="local")
        route_local = _route_metadata_from_selection(local_selection, "on_device", False)
        assert route_local.execution is not None
        assert route_local.execution.mode == "sdk_runtime"

        cloud_selection = RuntimeSelection(locality="cloud", engine=None, source="server", reason="cloud")
        route_cloud = _route_metadata_from_selection(cloud_selection, "cloud", False)
        assert route_cloud.execution is not None
        assert route_cloud.execution.mode == "hosted_gateway"

    @pytest.mark.asyncio
    async def test_create_response_includes_route_metadata(self):
        kernel = _make_kernel()

        with patch.object(kernel, "_build_router") as mock_build:
            mock_router = MagicMock()
            mock_router.resolve_locality = MagicMock(return_value=("on_device", False))
            mock_router.run = AsyncMock(return_value=_mock_runtime_response())
            mock_build.return_value = mock_router

            with patch(
                "octomil.execution.kernel._resolve_planner_selection",
                return_value=RuntimeSelection(
                    locality="local",
                    engine="mlx-lm",
                    source="cache",
                    reason="cached plan",
                ),
            ):
                result = await kernel.create_response("Hello!")

        assert result.route is not None
        assert result.route.execution is not None
        assert result.route.execution.locality == "local"
        assert result.route.execution.engine == "mlx-lm"
        assert result.route.planner.source == "cache"

    @pytest.mark.asyncio
    async def test_create_response_attempt_runner_falls_back_on_inference_error(self):
        kernel = _make_kernel()
        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="server",
            reason="server ordered candidates",
            fallback_allowed=True,
            candidates=[
                RuntimeCandidatePlan(
                    locality="local",
                    priority=0,
                    confidence=0.9,
                    reason="try local first",
                    engine="mlx-lm",
                ),
                RuntimeCandidatePlan(
                    locality="cloud",
                    priority=1,
                    confidence=0.6,
                    reason="hosted fallback",
                ),
            ],
        )

        local_router = MagicMock()
        local_router.run = AsyncMock(side_effect=RuntimeError("local crashed"))
        cloud_router = MagicMock()
        cloud_router.run = AsyncMock(return_value=_mock_runtime_response("cloud ok"))

        with (
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection),
            patch.object(kernel, "_build_router", side_effect=[local_router, cloud_router]),
        ):
            result = await kernel.create_response("Hello!")

        assert result.output_text == "cloud ok"
        assert result.fallback_used is True
        assert result.locality == "cloud"
        assert result.route is not None
        assert result.route.fallback.used is True
        assert result.route.fallback.trigger is not None
        assert result.route.fallback.trigger["code"] == "inference_error"
        assert len(result.route.attempts) == 2
        assert result.route.attempts[0]["status"] == "failed"
        assert result.route.attempts[0]["stage"] == "inference"
        assert result.route.attempts[1]["status"] == "selected"

    @pytest.mark.asyncio
    async def test_stream_response_falls_back_before_first_token(self):
        kernel = _make_kernel()
        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="server",
            reason="server ordered candidates",
            fallback_allowed=True,
            candidates=[
                RuntimeCandidatePlan(locality="local", priority=0, confidence=0.9, reason="local", engine="mlx-lm"),
                RuntimeCandidatePlan(locality="cloud", priority=1, confidence=0.6, reason="cloud"),
            ],
        )

        class _PreTokenFailureRouter:
            async def stream(self, request, *, policy=None):
                if request is None:
                    yield RuntimeChunk(text="")
                raise RuntimeError("local crashed before token")

        class _CloudStreamRouter:
            async def stream(self, request, *, policy=None):
                yield RuntimeChunk(text="cloud")

        with (
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection),
            patch.object(kernel, "_build_router", side_effect=[_PreTokenFailureRouter(), _CloudStreamRouter()]),
        ):
            chunks = [chunk async for chunk in kernel.stream_response("Hello!")]

        assert [chunk.delta for chunk in chunks] == ["cloud", ""]
        assert chunks[-1].done is True
        assert chunks[-1].result is not None
        assert chunks[-1].result.fallback_used is True
        assert chunks[-1].result.locality == "cloud"

    @pytest.mark.asyncio
    async def test_stream_response_does_not_fallback_after_first_token(self):
        kernel = _make_kernel()
        selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="server",
            reason="server ordered candidates",
            fallback_allowed=True,
            candidates=[
                RuntimeCandidatePlan(locality="local", priority=0, confidence=0.9, reason="local", engine="mlx-lm"),
                RuntimeCandidatePlan(locality="cloud", priority=1, confidence=0.6, reason="cloud"),
            ],
        )

        class _PostTokenFailureRouter:
            async def stream(self, request, *, policy=None):
                yield RuntimeChunk(text="partial")
                raise RuntimeError("local crashed after token")

        with (
            patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection),
            patch.object(kernel, "_build_router", return_value=_PostTokenFailureRouter()) as mock_build,
        ):
            chunks = []
            with pytest.raises(RuntimeError, match="after token"):
                async for chunk in kernel.stream_response("Hello!"):
                    chunks.append(chunk)

        assert [chunk.delta for chunk in chunks] == ["partial"]
        assert mock_build.call_count == 1

    @pytest.mark.asyncio
    async def test_create_embeddings_includes_route_metadata(self):
        kernel = _make_kernel(policy="private")

        mock_runtime = MagicMock()
        mock_runtime.embed = MagicMock(return_value=[[0.1, 0.2]])

        with patch.object(kernel, "_can_local", return_value=True):
            with patch("octomil.runtime.core.registry.ModelRuntimeRegistry") as mock_reg_cls:
                mock_registry = MagicMock()
                mock_registry.resolve.return_value = mock_runtime
                mock_reg_cls.shared.return_value = mock_registry

                with patch(
                    "octomil.execution.kernel._resolve_planner_selection",
                    return_value=RuntimeSelection(
                        locality="local",
                        engine="mlx-lm",
                        source="server",
                        reason="server",
                    ),
                ):
                    result = await kernel.create_embeddings(["hello"])

        assert result.route is not None
        assert result.route.planner.source == "server"

    @pytest.mark.asyncio
    async def test_transcribe_audio_includes_route_metadata(self):
        kernel = _make_kernel(policy="local_first")

        mock_backend = MagicMock()
        mock_backend.transcribe = MagicMock(return_value={"text": "hello", "segments": []})

        with patch.object(kernel, "_has_local_transcription_backend", return_value=True):
            with patch.object(kernel, "_resolve_local_transcription_backend", return_value=mock_backend):
                with patch(
                    "octomil.execution.kernel._resolve_planner_selection",
                    return_value=RuntimeSelection(
                        locality="local",
                        engine="whisper.cpp",
                        source="offline",
                        reason="local whisper",
                    ),
                ):
                    result = await kernel.transcribe_audio(b"fake_audio")

        assert result.route is not None
        assert result.route.execution is not None
        assert result.route.execution.engine == "whisper.cpp"
        assert result.route.planner.source == "offline"


# ---------------------------------------------------------------------------
# JSON CLI output includes route metadata
# ---------------------------------------------------------------------------


class TestCliJsonRouteMetadata:
    def test_result_to_dict_includes_route(self):
        from octomil.commands.inference import _result_to_dict

        result = ExecutionResult(
            id="resp_abc123",
            model="gemma-2b",
            capability="chat",
            locality="on_device",
            fallback_used=False,
            output_text="Hello!",
            route=RouteMetadata(
                execution=RouteExecution(locality="local", mode="sdk_runtime", engine="mlx-lm"),
                model=RouteModel(requested=RouteModelRequested(ref="gemma-2b", kind="model")),
                planner=PlannerInfo(source="cache"),
                fallback=FallbackInfo(used=False),
                reason=RouteReason(code="cache", message="cached plan"),
            ),
        )
        d = _result_to_dict(result)
        assert "route" in d
        assert d["route"]["execution"]["locality"] == "local"
        assert d["route"]["execution"]["engine"] == "mlx-lm"
        assert d["route"]["planner"]["source"] == "cache"
        assert d["route"]["fallback"]["used"] is False
        assert d["route"]["reason"]["message"] == "cached plan"

    def test_result_to_dict_without_route(self):
        from octomil.commands.inference import _result_to_dict

        result = ExecutionResult(
            id="resp_abc123",
            model="gemma-2b",
            capability="chat",
            locality="on_device",
            output_text="Hello!",
        )
        d = _result_to_dict(result)
        assert "route" not in d

    def test_embed_result_to_dict_includes_route(self):
        from octomil.commands.inference import _embed_result_to_dict

        result = ExecutionResult(
            model="embed-model",
            capability="embedding",
            locality="on_device",
            embeddings=[[0.1, 0.2]],
            dimensions=2,
            route=RouteMetadata(
                execution=RouteExecution(locality="local", mode="sdk_runtime", engine="mlx-lm"),
                planner=PlannerInfo(source="server"),
            ),
        )
        d = _embed_result_to_dict(result)
        assert "route" in d
        assert d["route"]["planner"]["source"] == "server"

    def test_transcribe_result_to_dict_includes_route(self):
        from octomil.commands.inference import _transcribe_result_to_dict

        result = ExecutionResult(
            model="whisper-small",
            capability="transcription",
            locality="on_device",
            output_text="hello world",
            route=RouteMetadata(
                execution=RouteExecution(locality="local", mode="sdk_runtime", engine="whisper.cpp"),
                planner=PlannerInfo(source="offline"),
            ),
        )
        d = _transcribe_result_to_dict(result)
        assert "route" in d
        assert d["route"]["execution"]["engine"] == "whisper.cpp"

    def test_json_output_is_serializable(self):
        """Route metadata must be JSON-serializable."""
        from octomil.commands.inference import _result_to_dict

        result = ExecutionResult(
            id="resp_test",
            model="gemma-2b",
            capability="chat",
            locality="on_device",
            output_text="Hi",
            route=RouteMetadata(
                execution=RouteExecution(locality="local", mode="sdk_runtime", engine="mlx-lm"),
                planner=PlannerInfo(source="cache"),
                fallback=FallbackInfo(used=False),
                reason=RouteReason(code="cache", message="cached plan hit"),
            ),
        )
        d = _result_to_dict(result)
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        assert parsed["route"]["execution"]["engine"] == "mlx-lm"


# ---------------------------------------------------------------------------
# Serve behaviour is not broken
# ---------------------------------------------------------------------------


class TestServeNotBroken:
    def test_resolve_chat_routing_still_works(self):
        """resolve_chat_routing (used by serve) must still work with planner integration."""
        kernel = _make_kernel(policy="local_first")
        decision = kernel.resolve_chat_routing(local_available=True, cloud_available=True)
        assert decision.model == "test-model"
        assert decision.primary_locality == "on_device"
        assert decision.fallback_locality == "cloud"

    def test_resolve_chat_routing_private(self):
        """Private policy routing via serve path must not break."""
        kernel = _make_kernel(policy="private")
        decision = kernel.resolve_chat_routing(local_available=True)
        assert decision.primary_locality == "on_device"
        assert decision.fallback_locality is None

    def test_resolve_chat_routing_cloud_only(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_SERVER_KEY", "test-key")
        config = LocalOctomilConfig(
            capabilities={CAPABILITY_CHAT: CapabilityDefault(model="test-model", policy="cloud_only")},
            cloud_profiles={"default": CloudProfile()},
        )
        kernel = ExecutionKernel(config_set=LoadedConfigSet(project=config))
        decision = kernel.resolve_chat_routing(local_available=False, cloud_available=True)
        assert decision.primary_locality == "cloud"


# ---------------------------------------------------------------------------
# Planner integration with kernel build_router
# ---------------------------------------------------------------------------


class TestPlannerDrivenRouting:
    @pytest.mark.asyncio
    async def test_planner_cloud_selection_forces_cloud_route(self):
        """When planner says locality=cloud, local_factory should return None."""
        kernel = _make_kernel(policy="cloud_first")

        cloud_selection = RuntimeSelection(
            locality="cloud",
            engine=None,
            source="server",
            reason="cloud preferred",
        )

        with patch(
            "octomil.execution.kernel._resolve_planner_selection",
            return_value=cloud_selection,
        ):
            with patch.object(kernel, "_build_router") as mock_build:
                mock_router = MagicMock()
                mock_router.resolve_locality = MagicMock(return_value=("cloud", False))
                mock_router.run = AsyncMock(return_value=_mock_runtime_response())
                mock_build.return_value = mock_router

                result = await kernel.create_response("Hello!")

        assert result.route is not None
        assert result.route.planner.source == "server"

    @pytest.mark.asyncio
    async def test_planner_local_engine_selection_used_by_router(self):
        """When planner recommends a specific local engine, the result should
        reflect that engine in route metadata."""
        kernel = _make_kernel()

        local_selection = RuntimeSelection(
            locality="local",
            engine="mlx-lm",
            source="cache",
            reason="cached mlx-lm selection",
        )

        with patch(
            "octomil.execution.kernel._resolve_planner_selection",
            return_value=local_selection,
        ):
            with patch.object(kernel, "_build_router") as mock_build:
                mock_router = MagicMock()
                mock_router.resolve_locality = MagicMock(return_value=("on_device", False))
                mock_router.run = AsyncMock(return_value=_mock_runtime_response())
                mock_build.return_value = mock_router

                result = await kernel.create_response("Hello!")

        assert result.route is not None
        assert result.route.execution is not None
        assert result.route.execution.engine == "mlx-lm"
        assert result.route.planner.source == "cache"

    @pytest.mark.asyncio
    async def test_planner_failure_does_not_break_execution(self):
        """If planner raises, execution must continue with fallback routing."""
        kernel = _make_kernel()

        with patch(
            "octomil.execution.kernel._resolve_planner_selection",
            return_value=None,
        ):
            with patch.object(kernel, "_build_router") as mock_build:
                mock_router = MagicMock()
                mock_router.resolve_locality = MagicMock(return_value=("on_device", False))
                mock_router.run = AsyncMock(return_value=_mock_runtime_response())
                mock_build.return_value = mock_router

                result = await kernel.create_response("Hello!")

        assert result.output_text == "Hello!"
        assert result.route is not None
        assert result.route.planner.source == "offline"
        assert result.route.reason.message == "planner not available"


# ---------------------------------------------------------------------------
# Planner client fetch_defaults
# ---------------------------------------------------------------------------


class TestPlannerClientFetchDefaults:
    def test_fetch_defaults_returns_none_on_failure(self):
        """HTTP errors should return None, not raise."""
        client = RuntimePlannerClient(base_url="http://localhost:1", api_key="test")
        result = client.fetch_defaults()
        assert result is None

    def test_fetch_defaults_parses_response(self):
        """Mock a successful response and verify it returns a dict."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={"default_model": "gemma-2b", "default_policy": "local_first"},
            request=httpx.Request("GET", "http://test/api/v2/runtime/defaults"),
        )

        client = RuntimePlannerClient(base_url="http://test", api_key="key")

        with patch("httpx.Client") as mock_client_cls:
            mock_http = MagicMock()
            mock_http.__enter__ = MagicMock(return_value=mock_http)
            mock_http.__exit__ = MagicMock(return_value=False)
            mock_http.get.return_value = mock_response
            mock_client_cls.return_value = mock_http

            result = client.fetch_defaults()

        assert result is not None
        assert result["default_model"] == "gemma-2b"


# ---------------------------------------------------------------------------
# Store cache key includes required fields
# ---------------------------------------------------------------------------


class TestCacheKeyCompleteness:
    def test_cache_key_includes_model_capability_policy(self):
        """Different model/capability/policy must produce different keys."""
        k1 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", policy="local_first")
        k2 = RuntimePlannerStore._make_cache_key(model="llama-8b", capability="text", policy="local_first")
        k3 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="embeddings", policy="local_first")
        k4 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", policy="cloud_only")
        assert k1 != k2
        assert k1 != k3
        assert k1 != k4

    def test_cache_key_includes_sdk_version(self):
        k1 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", sdk_version="4.5.0")
        k2 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", sdk_version="4.6.0")
        assert k1 != k2

    def test_cache_key_includes_platform_and_arch(self):
        k1 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", platform="Darwin", arch="arm64")
        k2 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", platform="Linux", arch="x86_64")
        assert k1 != k2

    def test_cache_key_includes_installed_hash(self):
        k1 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", installed_hash="abc123")
        k2 = RuntimePlannerStore._make_cache_key(model="gemma-2b", capability="text", installed_hash="def456")
        assert k1 != k2
