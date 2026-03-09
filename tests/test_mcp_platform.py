"""Tests for platform MCP tools (resolve, list, detect, inference, metrics, deploy)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures — lightweight fakes for SDK internals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeGGUFSource:
    repo: str = "octomil/test-model-GGUF"
    filename: str = "model-q4.gguf"


@dataclass(frozen=True)
class FakeVariantSpec:
    mlx: Optional[str] = "octomil/test-model-mlx-4bit"
    gguf: Optional[FakeGGUFSource] = None
    ort: Optional[str] = None
    mlc: Optional[str] = None
    ollama: Optional[str] = None
    source_repo: Optional[str] = None


@dataclass
class FakeModelEntry:
    publisher: str = "octomil"
    params: str = "3B"
    default_quant: str = "4bit"
    variants: dict[str, Any] = None  # type: ignore[assignment]
    engines: frozenset[str] = frozenset({"mlx-lm", "llama.cpp"})
    architecture: str = "dense"
    moe: Any = None

    def __post_init__(self) -> None:
        if self.variants is None:
            self.variants = {"4bit": FakeVariantSpec()}


@dataclass(frozen=True)
class FakeResolvedModel:
    family: Optional[str] = "test-model"
    quant: str = "4bit"
    engine: Optional[str] = "mlx-lm"
    hf_repo: str = "octomil/test-model-mlx-4bit"
    filename: Optional[str] = None
    mlx_repo: Optional[str] = "octomil/test-model-mlx-4bit"
    source_repo: Optional[str] = None
    raw: str = "test-model"
    architecture: str = "dense"
    moe: Any = None

    @property
    def is_gguf(self) -> bool:
        return self.filename is not None and self.filename.endswith(".gguf")

    @property
    def is_moe(self) -> bool:
        return self.architecture == "moe" and self.moe is not None


class FakeEnginePlugin:
    def __init__(self, name: str = "mlx-lm", available: bool = True, priority: int = 10) -> None:
        self._name = name
        self._available = available
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority


@dataclass
class FakeDetectionResult:
    engine: Any
    available: bool
    info: str = ""


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_resolves_successfully(self) -> None:
        tools = _get_tool_funcs()
        with patch("octomil.models.resolver.resolve", return_value=FakeResolvedModel()):
            result = json.loads(tools["resolve_model"]("test-model"))

        assert result["family"] == "test-model"
        assert result["engine"] == "mlx-lm"
        assert result["hf_repo"] == "octomil/test-model-mlx-4bit"
        assert result["quant"] == "4bit"

    def test_resolve_with_engine(self) -> None:
        tools = _get_tool_funcs()
        resolved = FakeResolvedModel(engine="llama.cpp", hf_repo="octomil/test-GGUF", filename="model.gguf")
        with patch("octomil.models.resolver.resolve", return_value=resolved):
            result = json.loads(tools["resolve_model"]("test-model", engine="llama.cpp"))

        assert result["engine"] == "llama.cpp"
        assert result["filename"] == "model.gguf"

    def test_resolve_unknown_model(self) -> None:
        tools = _get_tool_funcs()

        from octomil.models.resolver import ModelResolutionError

        with patch("octomil.models.resolver.resolve", side_effect=ModelResolutionError("Unknown model 'foo'")):
            result = json.loads(tools["resolve_model"]("foo"))

        assert result["error"] == "model_resolution_error"
        assert "Unknown model" in result["message"]


# ---------------------------------------------------------------------------
# Helper to get registered tool functions
# ---------------------------------------------------------------------------


def _get_tool_funcs(backend: Any = None) -> dict[str, Any]:
    """Register platform tools on a mock MCP and return the captured functions."""
    import octomil.mcp.platform_tools as pt

    mock_mcp = MagicMock()
    tool_funcs: dict[str, Any] = {}

    def capture_tool(fn: Any = None) -> Any:
        if fn is not None:
            tool_funcs[fn.__name__] = fn
            return fn

        def decorator(f: Any) -> Any:
            tool_funcs[f.__name__] = f
            return f

        return decorator

    mock_mcp.tool = capture_tool
    pt.register_platform_tools(mock_mcp, backend or MagicMock())
    return tool_funcs


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_lists_models(self) -> None:
        fake_catalog = {
            "gemma-3b": FakeModelEntry(publisher="google", params="3B"),
            "phi-mini": FakeModelEntry(publisher="microsoft", params="3.8B"),
        }
        tools = _get_tool_funcs()
        with patch("octomil.models.catalog.CATALOG", fake_catalog):
            result = json.loads(tools["list_models"]())

        assert result["count"] == 2
        names = [m["name"] for m in result["models"]]
        assert "gemma-3b" in names
        assert "phi-mini" in names

    def test_empty_catalog(self) -> None:
        tools = _get_tool_funcs()
        with patch("octomil.models.catalog.CATALOG", {}):
            result = json.loads(tools["list_models"]())

        assert result["count"] == 0
        assert result["models"] == []


# ---------------------------------------------------------------------------
# detect_engines
# ---------------------------------------------------------------------------


class TestDetectEngines:
    def test_detects_available_engines(self) -> None:
        fake_results = [
            FakeDetectionResult(engine=FakeEnginePlugin("mlx-lm", True, 10), available=True, info="Apple M2"),
            FakeDetectionResult(engine=FakeEnginePlugin("llama.cpp", False, 50), available=False, info=""),
        ]
        mock_registry = MagicMock()
        mock_registry.detect_all.return_value = fake_results

        tools = _get_tool_funcs()
        with patch("octomil.engines.registry.get_registry", return_value=mock_registry):
            result = json.loads(tools["detect_engines"]())

        assert result["available_count"] == 1
        assert result["engines"][0]["engine"] == "mlx-lm"
        assert result["engines"][0]["available"] is True

    def test_detect_with_model_filter(self) -> None:
        fake_results = [
            FakeDetectionResult(engine=FakeEnginePlugin("mlx-lm", True, 10), available=True, info=""),
        ]
        mock_registry = MagicMock()
        mock_registry.detect_all.return_value = fake_results

        tools = _get_tool_funcs()
        with patch("octomil.engines.registry.get_registry", return_value=mock_registry):
            result = json.loads(tools["detect_engines"]("gemma-3b"))

        assert result["model_filter"] == "gemma-3b"
        mock_registry.detect_all.assert_called_once_with("gemma-3b")


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------


class TestRunInference:
    def test_runs_inference(self) -> None:
        mock_backend = MagicMock()
        mock_backend.generate.return_value = (
            "Hello world",
            {
                "engine": "mlx-lm",
                "model": "qwen-coder-7b",
                "tokens_per_second": 42.0,
                "total_tokens": 10,
                "ttfc_ms": 50.0,
            },
        )
        tools = _get_tool_funcs(backend=mock_backend)
        result = json.loads(tools["run_inference"]("Say hello"))

        assert result["text"] == "Hello world"
        assert result["metrics"]["engine"] == "mlx-lm"
        mock_backend.generate.assert_called_once()
        # Verify no system prompt wrapping — just a user message
        call_args = mock_backend.generate.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_inference_with_params(self) -> None:
        mock_backend = MagicMock()
        mock_backend.generate.return_value = (
            "ok",
            {"engine": "mlx-lm", "model": "m", "tokens_per_second": 0, "total_tokens": 0, "ttfc_ms": 0},
        )
        tools = _get_tool_funcs(backend=mock_backend)
        tools["run_inference"]("test", max_tokens=512, temperature=0.1)

        call_args = mock_backend.generate.call_args
        assert call_args[1]["max_tokens"] == 512
        assert call_args[1]["temperature"] == 0.1

    def test_inference_error(self) -> None:
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = RuntimeError("Model failed to load")
        tools = _get_tool_funcs(backend=mock_backend)
        result = json.loads(tools["run_inference"]("test"))

        assert result["error"] == "inference_error"
        assert "Model failed to load" in result["message"]


# ---------------------------------------------------------------------------
# get_metrics
# ---------------------------------------------------------------------------


class TestGetMetrics:
    def test_metrics_not_loaded(self) -> None:
        mock_backend = MagicMock()
        mock_backend.model_name = "qwen-coder-7b"
        mock_backend._engine_name = "unknown"
        mock_backend.is_loaded = False

        tools = _get_tool_funcs(backend=mock_backend)
        result = json.loads(tools["get_metrics"]())

        assert result["model"] == "qwen-coder-7b"
        assert result["loaded"] is False
        assert result["engine"] == "unknown"

    def test_metrics_loaded(self) -> None:
        mock_backend = MagicMock()
        mock_backend.model_name = "gemma-3b"
        mock_backend._engine_name = "mlx-lm"
        mock_backend.is_loaded = True

        tools = _get_tool_funcs(backend=mock_backend)
        result = json.loads(tools["get_metrics"]())

        assert result["model"] == "gemma-3b"
        assert result["loaded"] is True
        assert result["engine"] == "mlx-lm"


# ---------------------------------------------------------------------------
# deploy_model
# ---------------------------------------------------------------------------


class TestDeployModel:
    def test_deploy_requires_api_key(self) -> None:
        tools = _get_tool_funcs()
        with patch.dict(os.environ, {}, clear=True):
            # Ensure OCTOMIL_API_KEY is not set
            os.environ.pop("OCTOMIL_API_KEY", None)
            result = json.loads(tools["deploy_model"]("test-model"))

        assert result["error"] == "auth_required"
        assert "OCTOMIL_API_KEY" in result["message"]

    def test_deploy_success(self) -> None:
        mock_client = MagicMock()
        mock_client.deploy.return_value = {"deployment_id": "dep-123", "status": "rolling_out"}

        tools = _get_tool_funcs()
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "test-key-123"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                result = json.loads(tools["deploy_model"]("test-model"))

        assert result["status"] == "deployed"
        assert result["result"]["deployment_id"] == "dep-123"
        mock_client.deploy.assert_called_once()

    def test_deploy_with_devices(self) -> None:
        mock_client = MagicMock()
        mock_client.deploy.return_value = {"deployment_id": "dep-456"}

        tools = _get_tool_funcs()
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "test-key"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                tools["deploy_model"]("test-model", devices="dev-1, dev-2", strategy="rolling")

        call_kwargs = mock_client.deploy.call_args[1]
        assert call_kwargs["devices"] == ["dev-1", "dev-2"]
        assert call_kwargs["strategy"] == "rolling"

    def test_deploy_error(self) -> None:
        mock_client = MagicMock()
        mock_client.deploy.side_effect = RuntimeError("API unreachable")

        tools = _get_tool_funcs()
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "test-key"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                result = json.loads(tools["deploy_model"]("test-model"))

        assert result["error"] == "deploy_error"
        assert "API unreachable" in result["message"]


# ---------------------------------------------------------------------------
# Integration: tools register on FastMCP
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_all_platform_tools_registered(self) -> None:
        tools = _get_tool_funcs()
        expected = {"resolve_model", "list_models", "detect_engines", "run_inference", "get_metrics", "deploy_model"}
        assert set(tools.keys()) == expected

    def test_platform_tools_on_real_fastmcp(self) -> None:
        """Verify tools register on a real FastMCP instance without errors."""
        pytest.importorskip("mcp.server.fastmcp", reason="mcp not installed")
        from mcp.server.fastmcp import FastMCP

        from octomil.mcp.platform_tools import register_platform_tools

        mcp = FastMCP("test")
        backend = MagicMock()
        register_platform_tools(mcp, backend)
        # No assertion needed — if this doesn't raise, tools registered successfully
