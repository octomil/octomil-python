"""Tests for platform MCP tools (Phase 1 + Phase 2)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
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

    def capture_tool(fn: Any = None, **_kwargs: Any) -> Any:
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


# ---------------------------------------------------------------------------
# Phase 2 fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeCPU:
    brand: str = "Apple M2"
    cores: int = 8
    threads: int = 8
    architecture: str = "arm64"
    base_speed_ghz: float = 3.49
    has_avx2: bool = False
    has_avx512: bool = False
    has_neon: bool = True
    estimated_gflops: float = 111.7


@dataclass
class FakeGPUDevice:
    name: str = "Apple M2"
    memory: Any = None

    def __post_init__(self) -> None:
        if self.memory is None:
            self.memory = MagicMock(total_gb=16.0)


@dataclass
class FakeGPU:
    backend: str = "metal"
    total_vram_gb: float = 16.0
    is_multi_gpu: bool = False
    speed_coefficient: float = 1.0
    gpus: list[Any] = field(default_factory=lambda: [FakeGPUDevice()])
    driver_version: str | None = None
    cuda_version: str | None = None


@dataclass
class FakeHardware:
    platform: str = "darwin"
    best_backend: str = "mlx"
    total_ram_gb: float = 16.0
    available_ram_gb: float = 10.5
    cpu: Any = None
    gpu: Any = None
    diagnostics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cpu is None:
            self.cpu = FakeCPU()
        if self.gpu is None:
            self.gpu = FakeGPU()


@dataclass
class FakeBenchmarkEntry:
    tokens_per_second: float = 42.5
    ok: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class FakeRankedBenchmark:
    engine: Any = None
    result: Any = None

    def __post_init__(self) -> None:
        if self.engine is None:
            self.engine = FakeEnginePlugin("mlx-lm")
        if self.result is None:
            self.result = FakeBenchmarkEntry()


@dataclass
class FakeSpeedEstimate:
    tokens_per_second: float = 35.0
    backend: str = "mlx"
    confidence: str = "measured"


@dataclass
class FakeInferenceConfig:
    strategy: str = "gpu_only"
    gpu_layers: int = -1
    vram_gb: float = 4.0
    ram_gb: float = 2.0


@dataclass
class FakeRecommendation:
    model_size: str = "3B"
    quantization: str = "4bit"
    reason: str = "Fits in 16GB unified memory"
    speed: Any = None
    config: Any = None
    serve_command: str = "octomil serve gemma-3b"

    def __post_init__(self) -> None:
        if self.speed is None:
            self.speed = FakeSpeedEstimate()
        if self.config is None:
            self.config = FakeInferenceConfig()


@dataclass
class FakeInferencePoint:
    file: str = "main.py"
    line: int = 42
    type: str = "pytorch"
    platform: str = "python"
    pattern: str = "torch.load"
    suggestion: str = "Consider ONNX Runtime for faster inference"
    context: str = "model = torch.load('model.pt')"


@dataclass
class FakeCompressionStats:
    original_tokens: int = 500
    compressed_tokens: int = 250
    compression_ratio: float = 0.5
    tokens_saved: int = 250
    savings_pct: float = 50.0
    strategy: str = "token_pruning"
    duration_ms: float = 12.34


# ---------------------------------------------------------------------------
# convert_model
# ---------------------------------------------------------------------------


class TestConvertModel:
    def test_convert_torch_not_installed(self) -> None:
        tools = _get_tool_funcs()
        with patch.dict("sys.modules", {"torch": None}):
            result = json.loads(tools["convert_model"]("/tmp/model.pt"))

        assert result["model"] == "model"
        assert "output_dir" in result
        assert "error" in result["conversions"].get("onnx", {})

    def test_convert_error(self) -> None:
        tools = _get_tool_funcs()
        result = json.loads(tools["convert_model"]("/nonexistent/model.pt"))
        # Should get an onnx error since the file doesn't exist
        assert "conversions" in result or "error" in result


# ---------------------------------------------------------------------------
# optimize_model
# ---------------------------------------------------------------------------


class TestOptimizeModel:
    def test_optimize_requires_api_key(self) -> None:
        tools = _get_tool_funcs()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OCTOMIL_API_KEY", None)
            result = json.loads(tools["optimize_model"]("test-model"))
        assert result["error"] == "auth_required"

    def test_optimize_success(self) -> None:
        mock_client = MagicMock()
        mock_client._registry.resolve_model_id.return_value = "model-123"
        mock_client._registry.optimize.return_value = {"status": "complete", "size_mb": 45.2}

        tools = _get_tool_funcs()
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "test-key"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                result = json.loads(tools["optimize_model"]("test-model", target_devices="ios,android"))

        assert result["status"] == "optimized"
        mock_client._registry.optimize.assert_called_once()

    def test_optimize_error(self) -> None:
        mock_client = MagicMock()
        mock_client._registry.resolve_model_id.side_effect = RuntimeError("not found")

        tools = _get_tool_funcs()
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "key"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                result = json.loads(tools["optimize_model"]("bad"))

        assert result["error"] == "optimize_error"


# ---------------------------------------------------------------------------
# detect_hardware_profile
# ---------------------------------------------------------------------------


class TestHardwareProfile:
    def test_detects_hardware(self) -> None:
        tools = _get_tool_funcs()
        with patch("octomil.hardware._unified.detect_hardware", return_value=FakeHardware()):
            result = json.loads(tools["detect_hardware_profile"]())

        assert result["platform"] == "darwin"
        assert result["best_backend"] == "mlx"
        assert result["cpu"]["brand"] == "Apple M2"
        assert result["cpu"]["cores"] == 8
        assert result["gpu"]["backend"] == "metal"

    def test_hardware_no_gpu(self) -> None:
        hw = FakeHardware()
        hw.gpu = None  # Override post_init
        tools = _get_tool_funcs()
        with patch("octomil.hardware._unified.detect_hardware", return_value=hw):
            result = json.loads(tools["detect_hardware_profile"]())

        assert result["gpu"] is None

    def test_hardware_error(self) -> None:
        tools = _get_tool_funcs()
        with patch("octomil.hardware._unified.detect_hardware", side_effect=RuntimeError("detection failed")):
            result = json.loads(tools["detect_hardware_profile"]())

        assert result["error"] == "hardware_error"


# ---------------------------------------------------------------------------
# benchmark_model
# ---------------------------------------------------------------------------


class TestBenchmarkModel:
    def test_benchmark_success(self) -> None:
        ranked = [
            FakeRankedBenchmark(
                engine=FakeEnginePlugin("mlx-lm"),
                result=FakeBenchmarkEntry(tokens_per_second=42.5, ok=True),
            ),
            FakeRankedBenchmark(
                engine=FakeEnginePlugin("llama.cpp"),
                result=FakeBenchmarkEntry(tokens_per_second=30.0, ok=True),
            ),
        ]
        mock_registry = MagicMock()
        mock_registry.benchmark_all.return_value = ranked

        tools = _get_tool_funcs()
        with patch("octomil.engines.registry.get_registry", return_value=mock_registry):
            result = json.loads(tools["benchmark_model"]("gemma-3b"))

        assert result["best_engine"] == "mlx-lm"
        assert len(result["results"]) == 2
        assert result["results"][0]["tokens_per_second"] == 42.5

    def test_benchmark_specific_engine(self) -> None:
        mock_engine = FakeEnginePlugin("mlx-lm")
        ranked = [FakeRankedBenchmark(engine=mock_engine)]
        mock_registry = MagicMock()
        mock_registry.get_engine.return_value = mock_engine
        mock_registry.benchmark_all.return_value = ranked

        tools = _get_tool_funcs()
        with patch("octomil.engines.registry.get_registry", return_value=mock_registry):
            result = json.loads(tools["benchmark_model"]("gemma-3b", engine="mlx-lm"))

        assert result["best_engine"] == "mlx-lm"
        mock_registry.get_engine.assert_called_once_with("mlx-lm")

    def test_benchmark_unknown_engine(self) -> None:
        mock_registry = MagicMock()
        mock_registry.get_engine.return_value = None
        mock_registry.engines = [FakeEnginePlugin("mlx-lm")]

        tools = _get_tool_funcs()
        with patch("octomil.engines.registry.get_registry", return_value=mock_registry):
            result = json.loads(tools["benchmark_model"]("gemma-3b", engine="nope"))

        assert result["error"] == "unknown_engine"
        assert "mlx-lm" in result["available"]


# ---------------------------------------------------------------------------
# recommend_model
# ---------------------------------------------------------------------------


class TestRecommendModel:
    def test_recommend_success(self) -> None:
        tools = _get_tool_funcs()
        with (
            patch("octomil.hardware._unified.detect_hardware", return_value=FakeHardware()),
            patch("octomil.model_optimizer.ModelOptimizer") as MockOptimizer,
        ):
            MockOptimizer.return_value.recommend.return_value = [FakeRecommendation()]
            result = json.loads(tools["recommend_model"]("balanced"))

        assert result["priority"] == "balanced"
        assert result["hardware_summary"]["platform"] == "darwin"
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["model_size"] == "3B"

    def test_recommend_error(self) -> None:
        tools = _get_tool_funcs()
        with patch("octomil.hardware._unified.detect_hardware", side_effect=RuntimeError("fail")):
            result = json.loads(tools["recommend_model"]())

        assert result["error"] == "recommend_error"


# ---------------------------------------------------------------------------
# scan_codebase
# ---------------------------------------------------------------------------


class TestScanCodebase:
    def test_scan_success(self) -> None:
        points = [FakeInferencePoint(), FakeInferencePoint(type="coreml", file="Model.swift", line=10)]
        tools = _get_tool_funcs()
        with patch("octomil.scanner.scan_directory", return_value=points):
            result = json.loads(tools["scan_codebase"]("/tmp/myproject"))

        assert result["total_points"] == 2
        assert result["by_type"]["pytorch"] == 1
        assert result["by_type"]["coreml"] == 1

    def test_scan_with_platform(self) -> None:
        tools = _get_tool_funcs()
        with patch("octomil.scanner.scan_directory", return_value=[]) as mock_scan:
            result = json.loads(tools["scan_codebase"]("/tmp/proj", platform="ios"))

        assert result["platform_filter"] == "ios"
        mock_scan.assert_called_once_with("/tmp/proj", platform="ios")

    def test_scan_not_found(self) -> None:
        tools = _get_tool_funcs()
        with patch("octomil.scanner.scan_directory", side_effect=FileNotFoundError("/bad/path")):
            result = json.loads(tools["scan_codebase"]("/bad/path"))

        assert result["error"] == "not_found"


# ---------------------------------------------------------------------------
# compress_prompt
# ---------------------------------------------------------------------------


class TestCompressPrompt:
    def test_compress_success(self) -> None:
        compressed = [{"role": "user", "content": "hello"}]
        stats = FakeCompressionStats()

        tools = _get_tool_funcs()
        with patch("octomil.compression.PromptCompressor") as MockCompressor:
            MockCompressor.return_value.compress.return_value = (compressed, stats)
            msgs = json.dumps([{"role": "user", "content": "hello world " * 50}])
            result = json.loads(tools["compress_prompt"](msgs))

        assert result["compressed_messages"] == compressed
        assert result["stats"]["original_tokens"] == 500
        assert result["stats"]["tokens_saved"] == 250

    def test_compress_invalid_json(self) -> None:
        tools = _get_tool_funcs()
        result = json.loads(tools["compress_prompt"]("not json"))
        assert result["error"] == "invalid_json"

    def test_compress_not_array(self) -> None:
        tools = _get_tool_funcs()
        result = json.loads(tools["compress_prompt"](json.dumps({"role": "user"})))
        assert result["error"] == "invalid_input"


# ---------------------------------------------------------------------------
# plan_deployment
# ---------------------------------------------------------------------------


class TestPlanDeployment:
    def test_plan_requires_api_key(self) -> None:
        tools = _get_tool_funcs()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OCTOMIL_API_KEY", None)
            result = json.loads(tools["plan_deployment"]("test-model"))
        assert result["error"] == "auth_required"

    def test_plan_success(self) -> None:
        mock_client = MagicMock()
        mock_client.deploy_prepare.return_value = {"stages": [{"device": "d1", "format": "coreml"}]}

        tools = _get_tool_funcs()
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "key"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                result = json.loads(tools["plan_deployment"]("test-model", devices="d1,d2"))

        assert result["status"] == "planned"
        mock_client.deploy_prepare.assert_called_once()
        call_kwargs = mock_client.deploy_prepare.call_args[1]
        assert call_kwargs["devices"] == ["d1", "d2"]

    def test_plan_error(self) -> None:
        mock_client = MagicMock()
        mock_client.deploy_prepare.side_effect = RuntimeError("fail")

        tools = _get_tool_funcs()
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "key"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                result = json.loads(tools["plan_deployment"]("bad"))
        assert result["error"] == "plan_error"


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_embed_requires_api_key(self) -> None:
        tools = _get_tool_funcs()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OCTOMIL_API_KEY", None)
            result = json.loads(tools["embed_text"]("hello"))
        assert result["error"] == "auth_required"

    def test_embed_requires_model(self) -> None:
        tools = _get_tool_funcs()
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "key"}):
            with patch("octomil.client.OctomilClient"):
                result = json.loads(tools["embed_text"]("hello"))
        assert result["error"] == "model_required"

    def test_embed_success(self) -> None:
        mock_client = MagicMock()
        mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]], "model": "text-embed-v1"}

        tools = _get_tool_funcs()
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "key"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                result = json.loads(tools["embed_text"]("hello world", model="text-embed-v1"))

        assert result["status"] == "ok"
        mock_client.embed.assert_called_once_with("text-embed-v1", "hello world")

    def test_embed_json_array_input(self) -> None:
        mock_client = MagicMock()
        mock_client.embed.return_value = {"embeddings": [[0.1], [0.2]]}

        tools = _get_tool_funcs()
        texts = json.dumps(["hello", "world"])
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "key"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                result = json.loads(tools["embed_text"](texts, model="m"))

        assert result["status"] == "ok"
        mock_client.embed.assert_called_once_with("m", ["hello", "world"])

    def test_embed_error(self) -> None:
        mock_client = MagicMock()
        mock_client.embed.side_effect = RuntimeError("API error")

        tools = _get_tool_funcs()
        with patch.dict(os.environ, {"OCTOMIL_API_KEY": "key"}):
            with patch("octomil.client.OctomilClient", return_value=mock_client):
                result = json.loads(tools["embed_text"]("x", model="m"))

        assert result["error"] == "embed_error"


# ---------------------------------------------------------------------------
# Integration: tools register on FastMCP
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_all_platform_tools_registered(self) -> None:
        tools = _get_tool_funcs()
        expected = {
            # Phase 1
            "resolve_model",
            "list_models",
            "detect_engines",
            "run_inference",
            "get_metrics",
            "deploy_model",
            # Phase 2
            "convert_model",
            "optimize_model",
            "detect_hardware_profile",
            "benchmark_model",
            "recommend_model",
            "scan_codebase",
            "compress_prompt",
            "plan_deployment",
            "embed_text",
        }
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
