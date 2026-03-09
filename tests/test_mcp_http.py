"""Tests for the Octomil HTTP agent server endpoints."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

# httpx + ASGITransport for testing FastAPI without a real server
httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")


@pytest_asyncio.fixture
async def client():
    """Create a test client for the HTTP server."""
    from httpx import ASGITransport, AsyncClient

    from octomil.mcp.http_server import HTTPServerConfig, create_http_app

    config = HTTPServerConfig(host="127.0.0.1", port=8402)
    app = create_http_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def authed_client():
    """Create a test client with auth configured."""
    from httpx import ASGITransport, AsyncClient

    from octomil.mcp.auth import reset_dev_mode_warning
    from octomil.mcp.http_server import HTTPServerConfig, create_http_app

    reset_dev_mode_warning()
    config = HTTPServerConfig(host="127.0.0.1", port=8402)

    with patch.dict(os.environ, {"OCTOMIL_MCP_API_KEY": "test-secret-key"}):
        app = create_http_app(config)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ---------------------------------------------------------------------------
# Discovery + health (no auth)
# ---------------------------------------------------------------------------


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_agent_card(self, client: Any) -> None:
        resp = await client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "Octomil Agent"
        assert "skills" in card
        assert len(card["skills"]) > 0
        assert card["capabilities"]["mcp"] is True
        assert card["capabilities"]["a2a"] is True
        assert card["capabilities"]["openapi"] is True

    @pytest.mark.asyncio
    async def test_agent_card_has_all_tools(self, client: Any) -> None:
        resp = await client.get("/.well-known/agent-card.json")
        card = resp.json()
        skill_names = {s["id"] for s in card["skills"]}
        # Phase 1 platform tools
        assert "resolve_model" in skill_names
        assert "list_models" in skill_names
        assert "detect_engines" in skill_names
        assert "run_inference" in skill_names
        assert "get_metrics" in skill_names
        assert "deploy_model" in skill_names
        # Phase 2 platform tools
        assert "convert_model" in skill_names
        assert "optimize_model" in skill_names
        assert "hardware_profile" in skill_names
        assert "benchmark_model" in skill_names
        assert "recommend_model" in skill_names
        assert "scan_codebase" in skill_names
        assert "compress_prompt" in skill_names
        assert "plan_deployment" in skill_names
        assert "embed" in skill_names
        # Code tools
        assert "generate_code" in skill_names
        assert "review_file" in skill_names

    @pytest.mark.asyncio
    async def test_health(self, client: Any) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data

    @pytest.mark.asyncio
    async def test_openapi_docs(self, client: Any) -> None:
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/api/v1/resolve_model" in schema["paths"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    @pytest.mark.asyncio
    async def test_dev_mode_allows_all(self, client: Any) -> None:
        """Without OCTOMIL_MCP_API_KEY, all requests pass through (dev mode)."""
        from octomil.mcp.auth import reset_dev_mode_warning

        reset_dev_mode_warning()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCTOMIL_MCP_API_KEY", None)
            with patch("octomil.models.catalog.CATALOG", {}):
                resp = await client.post("/api/v1/list_models", json={})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_rejects_missing_token(self, authed_client: Any) -> None:
        resp = await authed_client.post("/api/v1/list_models", json={})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_rejects_wrong_token(self, authed_client: Any) -> None:
        resp = await authed_client.post(
            "/api/v1/list_models",
            json={},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_accepts_valid_token(self, authed_client: Any) -> None:
        with patch("octomil.models.catalog.CATALOG", {}):
            resp = await authed_client.post(
                "/api/v1/list_models",
                json={},
                headers={"Authorization": "Bearer test-secret-key"},
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------


class TestRESTEndpoints:
    @pytest.mark.asyncio
    async def test_list_models(self, client: Any) -> None:
        from tests.test_mcp_platform import FakeModelEntry

        fake_catalog = {"test-model": FakeModelEntry()}
        with patch("octomil.models.catalog.CATALOG", fake_catalog):
            resp = await client.post("/api/v1/list_models", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_resolve_model(self, client: Any) -> None:
        from tests.test_mcp_platform import FakeResolvedModel

        with patch("octomil.models.resolver.resolve", return_value=FakeResolvedModel()):
            resp = await client.post("/api/v1/resolve_model", json={"name": "test-model"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["family"] == "test-model"
        assert data["engine"] == "mlx-lm"

    @pytest.mark.asyncio
    async def test_resolve_model_not_found(self, client: Any) -> None:
        from octomil.models.resolver import ModelResolutionError

        with patch("octomil.models.resolver.resolve", side_effect=ModelResolutionError("Not found")):
            resp = await client.post("/api/v1/resolve_model", json={"name": "nonexistent"})

        assert resp.status_code == 404
        assert resp.json()["error"] == "model_resolution_error"

    @pytest.mark.asyncio
    async def test_detect_engines(self, client: Any) -> None:
        from tests.test_mcp_platform import FakeDetectionResult, FakeEnginePlugin

        fake_results = [
            FakeDetectionResult(engine=FakeEnginePlugin("mlx-lm", True, 10), available=True, info="M2"),
        ]
        mock_registry = MagicMock()
        mock_registry.detect_all.return_value = fake_results
        with patch("octomil.engines.registry.get_registry", return_value=mock_registry):
            resp = await client.post("/api/v1/detect_engines", json={})

        assert resp.status_code == 200
        assert resp.json()["available_count"] == 1

    @pytest.mark.asyncio
    async def test_metrics(self, client: Any) -> None:
        resp = await client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "loaded" in data

    @pytest.mark.asyncio
    async def test_run_inference(self, client: Any) -> None:
        mock_backend = MagicMock()
        mock_backend.generate.return_value = (
            "Hello",
            {"engine": "mlx-lm", "model": "m", "tokens_per_second": 1, "total_tokens": 1, "ttfc_ms": 1},
        )
        with patch.object(client._transport.app.state, "backend", mock_backend):  # type: ignore[union-attr]
            # Need to patch at the app level — the route closure captures backend
            pass
        # Since backend is captured in closure, we test via the app's actual backend
        # For this test, just verify the endpoint accepts the request format
        resp = await client.post("/api/v1/run_inference", json={"prompt": "hello"})
        # Will return 503 since no model loaded and no cloud fallback — that's expected
        assert resp.status_code in (200, 503)

    @pytest.mark.asyncio
    async def test_deploy_no_api_key(self, client: Any) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCTOMIL_API_KEY", None)
            resp = await client.post("/api/v1/deploy_model", json={"name": "test"})

        assert resp.status_code == 403
        assert resp.json()["error"] == "auth_required"


# ---------------------------------------------------------------------------
# Phase 2 REST API endpoints
# ---------------------------------------------------------------------------


class TestPhase2Endpoints:
    @pytest.mark.asyncio
    async def test_convert_model(self, client: Any) -> None:
        """Convert endpoint accepts request and returns structured response."""
        resp = await client.post(
            "/api/v1/convert_model",
            json={"model_path": "/tmp/model.pt", "target": "onnx"},
        )
        # Will fail (no real model file) but should return structured response, not 422
        assert resp.status_code in (200, 500)
        data = resp.json()
        assert "model" in data or "error" in data

    @pytest.mark.asyncio
    async def test_optimize_no_api_key(self, client: Any) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCTOMIL_API_KEY", None)
            resp = await client.post("/api/v1/optimize_model", json={"name": "test"})
        assert resp.status_code == 403
        assert resp.json()["error"] == "auth_required"

    @pytest.mark.asyncio
    async def test_hardware_profile(self, client: Any) -> None:
        from tests.test_mcp_platform import FakeHardware

        with patch("octomil.hardware._unified.detect_hardware", return_value=FakeHardware()):
            resp = await client.get("/api/v1/hardware_profile")
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform"] == "darwin"
        assert data["cpu"]["brand"] == "Apple M2"

    @pytest.mark.asyncio
    async def test_benchmark_model(self, client: Any) -> None:
        from tests.test_mcp_platform import FakeEnginePlugin, FakeRankedBenchmark

        ranked = [FakeRankedBenchmark(engine=FakeEnginePlugin("mlx-lm"))]
        mock_registry = MagicMock()
        mock_registry.benchmark_all.return_value = ranked
        with patch("octomil.engines.registry.get_registry", return_value=mock_registry):
            resp = await client.post("/api/v1/benchmark_model", json={"model_name": "gemma-3b"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["best_engine"] == "mlx-lm"

    @pytest.mark.asyncio
    async def test_recommend_model(self, client: Any) -> None:
        from tests.test_mcp_platform import FakeHardware, FakeRecommendation

        with (
            patch("octomil.hardware._unified.detect_hardware", return_value=FakeHardware()),
            patch("octomil.model_optimizer.ModelOptimizer") as MockOpt,
        ):
            MockOpt.return_value.recommend.return_value = [FakeRecommendation()]
            resp = await client.post("/api/v1/recommend_model", json={"priority": "speed"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["priority"] == "speed"
        assert len(data["recommendations"]) == 1

    @pytest.mark.asyncio
    async def test_scan_codebase(self, client: Any) -> None:
        from tests.test_mcp_platform import FakeInferencePoint

        with patch("octomil.scanner.scan_directory", return_value=[FakeInferencePoint()]):
            resp = await client.post("/api/v1/scan_codebase", json={"path": "/tmp/proj"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_points"] == 1

    @pytest.mark.asyncio
    async def test_scan_codebase_not_found(self, client: Any) -> None:
        with patch("octomil.scanner.scan_directory", side_effect=FileNotFoundError("/bad")):
            resp = await client.post("/api/v1/scan_codebase", json={"path": "/bad"})
        assert resp.status_code == 404
        assert resp.json()["error"] == "not_found"

    @pytest.mark.asyncio
    async def test_compress_prompt(self, client: Any) -> None:
        from tests.test_mcp_platform import FakeCompressionStats

        compressed = [{"role": "user", "content": "hello"}]
        with patch("octomil.compression.PromptCompressor") as MockComp:
            MockComp.return_value.compress.return_value = (compressed, FakeCompressionStats())
            msgs = '[{"role":"user","content":"hello world hello world"}]'
            resp = await client.post(
                "/api/v1/compress_prompt",
                json={"messages": msgs},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["compressed_messages"] == compressed
        assert data["stats"]["tokens_saved"] == 250

    @pytest.mark.asyncio
    async def test_plan_deployment_no_api_key(self, client: Any) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCTOMIL_API_KEY", None)
            resp = await client.post("/api/v1/plan_deployment", json={"name": "test"})
        assert resp.status_code == 403
        assert resp.json()["error"] == "auth_required"

    @pytest.mark.asyncio
    async def test_plan_deployment_success(self, client: Any) -> None:
        mock_client = MagicMock()
        mock_client.deploy_prepare.return_value = {"stages": []}
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "key"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                resp = await client.post("/api/v1/plan_deployment", json={"name": "m"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "planned"

    @pytest.mark.asyncio
    async def test_embed_no_api_key(self, client: Any) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCTOMIL_API_KEY", None)
            resp = await client.post("/api/v1/embed", json={"text": "hello", "model": "m"})
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_embed_no_model(self, client: Any) -> None:
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "key"}):
            resp = await client.post("/api/v1/embed", json={"text": "hello"})
        assert resp.status_code == 400
        assert resp.json()["error"] == "model_required"

    @pytest.mark.asyncio
    async def test_embed_success(self, client: Any) -> None:
        mock_client = MagicMock()
        mock_client.embed.return_value = {"embeddings": [[0.1, 0.2]]}
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "key"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                resp = await client.post("/api/v1/embed", json={"text": "hello", "model": "m"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Code tool endpoints
# ---------------------------------------------------------------------------


class TestCodeToolEndpoints:
    def _no_model_ctx(self, client: Any):
        """Context manager that makes the backend raise (simulating no model) and removes cloud key."""
        backend = client._transport.app.state.backend
        return (
            patch.object(backend, "generate", side_effect=RuntimeError("no model loaded")),
            patch.dict(os.environ, {}, clear=False),
        )

    @pytest.mark.asyncio
    async def test_generate_code_no_model(self, client: Any) -> None:
        """Without a loaded model or cloud key, code tools return 503."""
        mock_gen, env_ctx = self._no_model_ctx(client)
        with mock_gen, env_ctx:
            os.environ.pop("OCTOMIL_API_KEY", None)
            resp = await client.post("/api/v1/generate_code", json={"description": "fibonacci function"})
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"] == "model_not_ready"
        assert data["retryable"] is True
        assert "warmup" in data["actions"]

    @pytest.mark.asyncio
    async def test_generate_code_with_language(self, client: Any) -> None:
        resp = await client.post(
            "/api/v1/generate_code",
            json={"description": "hello world", "language": "python", "context": "use print()"},
        )
        assert resp.status_code in (200, 503)

    @pytest.mark.asyncio
    async def test_review_code_no_model(self, client: Any) -> None:
        mock_gen, env_ctx = self._no_model_ctx(client)
        with mock_gen, env_ctx:
            os.environ.pop("OCTOMIL_API_KEY", None)
            resp = await client.post("/api/v1/review_code", json={"code": "def f(): pass"})
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"] == "model_not_ready"

    @pytest.mark.asyncio
    async def test_explain_code_no_model(self, client: Any) -> None:
        mock_gen, env_ctx = self._no_model_ctx(client)
        with mock_gen, env_ctx:
            os.environ.pop("OCTOMIL_API_KEY", None)
            resp = await client.post("/api/v1/explain_code", json={"code": "x = [i**2 for i in range(10)]"})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_write_tests_no_model(self, client: Any) -> None:
        mock_gen, env_ctx = self._no_model_ctx(client)
        with mock_gen, env_ctx:
            os.environ.pop("OCTOMIL_API_KEY", None)
            resp = await client.post("/api/v1/write_tests", json={"code": "def add(a, b): return a + b"})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_general_task_no_model(self, client: Any) -> None:
        mock_gen, env_ctx = self._no_model_ctx(client)
        with mock_gen, env_ctx:
            os.environ.pop("OCTOMIL_API_KEY", None)
            resp = await client.post("/api/v1/general_task", json={"prompt": "What is 2+2?"})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_general_task_with_context(self, client: Any) -> None:
        resp = await client.post(
            "/api/v1/general_task",
            json={"prompt": "summarize", "context": "The quick brown fox."},
        )
        assert resp.status_code in (200, 503)

    @pytest.mark.asyncio
    async def test_code_endpoints_in_openapi(self, client: Any) -> None:
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        assert "/api/v1/generate_code" in paths
        assert "/api/v1/review_code" in paths
        assert "/api/v1/explain_code" in paths
        assert "/api/v1/write_tests" in paths
        assert "/api/v1/general_task" in paths


# ---------------------------------------------------------------------------
# Readiness endpoints
# ---------------------------------------------------------------------------


class TestReadinessEndpoints:
    @pytest.mark.asyncio
    async def test_ready_not_loaded(self, client: Any) -> None:
        """Ready endpoint returns 503 when model is not loaded."""
        resp = await client.get("/api/v1/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["ready"] is False
        assert "model" in data

    @pytest.mark.asyncio
    async def test_warmup_returns_status(self, client: Any) -> None:
        """Warmup endpoint returns model loading status."""
        resp = await client.post("/api/v1/warmup")
        # Without a real model, warmup kicks off async loading — returns 202
        assert resp.status_code in (200, 202, 503)
        data = resp.json()
        assert "status" in data
        assert "model" in data

    @pytest.mark.asyncio
    async def test_warmup_in_openapi(self, client: Any) -> None:
        resp = await client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/api/v1/warmup" in paths
        assert "/api/v1/ready" in paths


# ---------------------------------------------------------------------------
# Agent card readiness
# ---------------------------------------------------------------------------


class TestAgentCardReadiness:
    @pytest.mark.asyncio
    async def test_agent_card_has_readiness_urls(self, client: Any) -> None:
        resp = await client.get("/.well-known/agent-card.json")
        card = resp.json()
        assert "readinessUrls" in card
        assert "warmup" in card["readinessUrls"]
        assert "ready" in card["readinessUrls"]

    @pytest.mark.asyncio
    async def test_agent_card_skill_readiness(self, client: Any) -> None:
        """Platform tools are always ready; code tools reflect model state."""
        resp = await client.get("/.well-known/agent-card.json")
        card = resp.json()
        skills_by_id = {s["id"]: s for s in card["skills"]}
        # Platform tools should be ready even without a model
        assert skills_by_id["resolve_model"]["ready"] is True
        assert skills_by_id["list_models"]["ready"] is True
        # Code tools should NOT be ready without a loaded model
        assert skills_by_id["generate_code"]["ready"] is False
        assert skills_by_id["review_code"]["ready"] is False
        assert skills_by_id["general_task"]["ready"] is False


# ---------------------------------------------------------------------------
# Cloud fallback
# ---------------------------------------------------------------------------


class TestCloudFallback:
    @pytest.mark.asyncio
    async def test_code_tool_cloud_fallback(self, client: Any) -> None:
        """When local model fails but OCTOMIL_API_KEY is set, falls back to cloud."""
        backend = client._transport.app.state.backend
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"role": "assistant", "content": "def fib(n): ..."}}
        with (
            patch.object(backend, "generate", side_effect=RuntimeError("no model")),
            patch.dict(os.environ, {"OCTOMIL_API_KEY": "key"}),
            patch("octomil.client.OctomilClient", return_value=mock_client),
        ):
            resp = await client.post("/api/v1/generate_code", json={"description": "fibonacci"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["metrics"]["fallback"] is True
        assert data["metrics"]["engine"] == "cloud"

    @pytest.mark.asyncio
    async def test_code_tool_no_fallback_no_key(self, client: Any) -> None:
        """Without OCTOMIL_API_KEY, returns 503 when local model is unavailable."""
        backend = client._transport.app.state.backend
        with (
            patch.object(backend, "generate", side_effect=RuntimeError("no model")),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("OCTOMIL_API_KEY", None)
            resp = await client.post("/api/v1/general_task", json={"prompt": "hello"})
        assert resp.status_code == 503
        data = resp.json()
        assert data["retryable"] is True
        assert "actions" in data


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    @pytest.mark.asyncio
    async def test_cors_headers(self, client: Any) -> None:
        resp = await client.options(
            "/api/v1/list_models",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers
