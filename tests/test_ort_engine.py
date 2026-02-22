"""Tests for ONNX Runtime engine plugin (EDG-63)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from octomil.engines.base import BenchmarkResult
from octomil.engines.ort_engine import (
    ONNXRuntimeEngine,
    _ORTBackend,
    _get_execution_providers,
    _has_onnxruntime,
    _has_onnxruntime_genai,
)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


class TestHasOnnxruntime:
    def test_available(self) -> None:
        mock_ort = MagicMock()
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                mock_ort if name == "onnxruntime" else __import__(name, *a, **kw)
            ),
        ):
            assert _has_onnxruntime() is True

    def test_unavailable(self) -> None:
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError())
                if name == "onnxruntime"
                else __import__(name, *a, **kw)
            ),
        ):
            assert _has_onnxruntime() is False


class TestHasOnnxruntimeGenai:
    def test_available(self) -> None:
        mock_og = MagicMock()
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                mock_og
                if name == "onnxruntime_genai"
                else __import__(name, *a, **kw)
            ),
        ):
            assert _has_onnxruntime_genai() is True

    def test_unavailable(self) -> None:
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError())
                if name == "onnxruntime_genai"
                else __import__(name, *a, **kw)
            ),
        ):
            assert _has_onnxruntime_genai() is False


class TestGetExecutionProviders:
    def test_returns_providers(self) -> None:
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ]
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            providers = _get_execution_providers()
            assert "CPUExecutionProvider" in providers
            assert "CUDAExecutionProvider" in providers

    def test_returns_empty_when_unavailable(self) -> None:
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError())
                if name == "onnxruntime"
                else __import__(name, *a, **kw)
            ),
        ):
            # Clear any cached module
            saved = sys.modules.pop("onnxruntime", None)
            try:
                assert _get_execution_providers() == []
            finally:
                if saved is not None:
                    sys.modules["onnxruntime"] = saved


# ---------------------------------------------------------------------------
# ONNXRuntimeEngine
# ---------------------------------------------------------------------------


class TestONNXRuntimeEngine:
    def setup_method(self) -> None:
        self.engine = ONNXRuntimeEngine()

    def test_name(self) -> None:
        assert self.engine.name == "onnxruntime"

    def test_display_name_cpu(self) -> None:
        with patch(
            "octomil.engines.ort_engine._get_execution_providers",
            return_value=["CPUExecutionProvider"],
        ):
            assert "CPU" in self.engine.display_name

    def test_display_name_cuda(self) -> None:
        with patch(
            "octomil.engines.ort_engine._get_execution_providers",
            return_value=["CPUExecutionProvider", "CUDAExecutionProvider"],
        ):
            assert "CUDA" in self.engine.display_name

    def test_display_name_coreml(self) -> None:
        with patch(
            "octomil.engines.ort_engine._get_execution_providers",
            return_value=["CPUExecutionProvider", "CoreMLExecutionProvider"],
        ):
            assert "CoreML" in self.engine.display_name

    def test_display_name_tensorrt(self) -> None:
        with patch(
            "octomil.engines.ort_engine._get_execution_providers",
            return_value=[
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
            ],
        ):
            assert "TensorRT" in self.engine.display_name

    def test_display_name_directml(self) -> None:
        with patch(
            "octomil.engines.ort_engine._get_execution_providers",
            return_value=["CPUExecutionProvider", "DmlExecutionProvider"],
        ):
            assert "DirectML" in self.engine.display_name

    def test_display_name_openvino(self) -> None:
        with patch(
            "octomil.engines.ort_engine._get_execution_providers",
            return_value=["CPUExecutionProvider", "OpenVINOExecutionProvider"],
        ):
            assert "OpenVINO" in self.engine.display_name

    def test_display_name_empty_providers(self) -> None:
        with patch(
            "octomil.engines.ort_engine._get_execution_providers",
            return_value=[],
        ):
            assert "CPU" in self.engine.display_name

    def test_priority(self) -> None:
        assert self.engine.priority == 30

    def test_detect_with_onnxruntime(self) -> None:
        with patch(
            "octomil.engines.ort_engine._has_onnxruntime", return_value=True
        ):
            assert self.engine.detect() is True

    def test_detect_without_onnxruntime(self) -> None:
        with patch(
            "octomil.engines.ort_engine._has_onnxruntime", return_value=False
        ):
            assert self.engine.detect() is False

    def test_detect_info_with_providers(self) -> None:
        with (
            patch(
                "octomil.engines.ort_engine._get_execution_providers",
                return_value=["CPUExecutionProvider", "CUDAExecutionProvider"],
            ),
            patch(
                "octomil.engines.ort_engine._has_onnxruntime_genai",
                return_value=True,
            ),
        ):
            info = self.engine.detect_info()
            assert "CPUExecutionProvider" in info
            assert "CUDAExecutionProvider" in info
            assert "GenAI available" in info

    def test_detect_info_without_genai(self) -> None:
        with (
            patch(
                "octomil.engines.ort_engine._get_execution_providers",
                return_value=["CPUExecutionProvider"],
            ),
            patch(
                "octomil.engines.ort_engine._has_onnxruntime_genai",
                return_value=False,
            ),
        ):
            info = self.engine.detect_info()
            assert "CPUExecutionProvider" in info
            assert "GenAI" not in info

    def test_detect_info_empty(self) -> None:
        with patch(
            "octomil.engines.ort_engine._get_execution_providers",
            return_value=[],
        ):
            assert self.engine.detect_info() == ""

    def test_supports_catalog_model(self) -> None:
        # gemma-1b has "onnxruntime" in its engines frozenset
        assert self.engine.supports_model("gemma-1b") is True
        assert self.engine.supports_model("llama-1b") is True

    def test_supports_onnx_file(self) -> None:
        assert self.engine.supports_model("model.onnx") is True

    def test_supports_hf_repo(self) -> None:
        assert self.engine.supports_model("microsoft/phi-3-onnx") is True

    def test_does_not_support_unknown(self) -> None:
        assert self.engine.supports_model("unknown-model") is False

    def test_benchmark_genai_preferred(self) -> None:
        with (
            patch(
                "octomil.engines.ort_engine._has_onnxruntime_genai",
                return_value=True,
            ),
            patch.object(
                self.engine,
                "_benchmark_genai",
                return_value=BenchmarkResult(
                    engine_name="onnxruntime",
                    tokens_per_second=50.0,
                    metadata={"method": "genai"},
                ),
            ) as mock_genai,
        ):
            result = self.engine.benchmark("gemma-1b")
            assert result.ok
            assert result.tokens_per_second == 50.0
            mock_genai.assert_called_once()

    def test_benchmark_session_fallback(self) -> None:
        with (
            patch(
                "octomil.engines.ort_engine._has_onnxruntime_genai",
                return_value=False,
            ),
            patch(
                "octomil.engines.ort_engine._has_onnxruntime",
                return_value=True,
            ),
            patch.object(
                self.engine,
                "_benchmark_session",
                return_value=BenchmarkResult(
                    engine_name="onnxruntime",
                    tokens_per_second=30.0,
                    metadata={"method": "session"},
                ),
            ) as mock_session,
        ):
            result = self.engine.benchmark("model.onnx")
            assert result.ok
            assert result.tokens_per_second == 30.0
            mock_session.assert_called_once()

    def test_benchmark_unavailable(self) -> None:
        with (
            patch(
                "octomil.engines.ort_engine._has_onnxruntime_genai",
                return_value=False,
            ),
            patch(
                "octomil.engines.ort_engine._has_onnxruntime",
                return_value=False,
            ),
        ):
            result = self.engine.benchmark("gemma-1b")
            assert result.ok is False
            assert "not available" in result.error

    def test_benchmark_genai_error_returns_result(self) -> None:
        with (
            patch(
                "octomil.engines.ort_engine._has_onnxruntime_genai",
                return_value=True,
            ),
            patch.object(
                self.engine,
                "_benchmark_genai",
                return_value=BenchmarkResult(
                    engine_name="onnxruntime",
                    error="Model not found",
                ),
            ),
        ):
            result = self.engine.benchmark("bad-model")
            assert result.ok is False
            assert result.engine_name == "onnxruntime"

    def test_create_backend(self) -> None:
        backend = self.engine.create_backend("gemma-1b")
        assert isinstance(backend, _ORTBackend)
        assert backend.name == "onnxruntime"

    def test_create_backend_returns_proper_type(self) -> None:
        backend = self.engine.create_backend("model.onnx", cache_size_mb=1024)
        assert hasattr(backend, "generate")
        assert hasattr(backend, "generate_stream")
        assert hasattr(backend, "list_models")
        assert hasattr(backend, "load_model")

    def test_resolve_model_path_onnx_file(self) -> None:
        with patch("os.path.isfile", return_value=True):
            path = self.engine._resolve_model_path("model.onnx")
            assert path == "model.onnx"

    def test_resolve_model_path_not_found_returns_name(self) -> None:
        with (
            patch("os.path.isfile", return_value=False),
            patch("os.path.exists", return_value=False),
        ):
            # When nothing is found, returns the name as-is
            path = self.engine._resolve_model_path("nonexistent")
            assert path == "nonexistent"

    def test_resolve_model_path_hf_download(self) -> None:
        with (
            patch("os.path.isfile", return_value=False),
            patch("os.path.exists", return_value=False),
        ):
            mock_download = MagicMock(return_value="/cache/model-dir")
            with patch.dict(
                "sys.modules",
                {"huggingface_hub": MagicMock(snapshot_download=mock_download)},
            ):
                path = self.engine._resolve_model_path("microsoft/phi-3-onnx")
                assert path == "/cache/model-dir"


# ---------------------------------------------------------------------------
# _ORTBackend
# ---------------------------------------------------------------------------


class TestORTBackend:
    def test_name(self) -> None:
        backend = _ORTBackend("test-model")
        assert backend.name == "onnxruntime"

    def test_list_models(self) -> None:
        backend = _ORTBackend("gemma-1b")
        assert backend.list_models() == ["gemma-1b"]

    def test_list_models_empty(self) -> None:
        backend = _ORTBackend("")
        assert backend.list_models() == []

    def test_load_model_genai(self) -> None:
        mock_og = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_og.Model.return_value = mock_model
        mock_og.Tokenizer.return_value = mock_tokenizer

        with (
            patch(
                "octomil.engines.ort_engine._has_onnxruntime_genai",
                return_value=True,
            ),
            patch.dict("sys.modules", {"onnxruntime_genai": mock_og}),
            patch.object(
                ONNXRuntimeEngine,
                "_resolve_model_path",
                return_value="/tmp/model",
            ),
        ):
            backend = _ORTBackend("test-model")
            backend.load_model("test-model")
            assert backend._use_genai is True
            assert backend._model is mock_model
            assert backend._tokenizer is mock_tokenizer

    def test_load_model_session_fallback(self) -> None:
        mock_ort = MagicMock()
        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        with (
            patch(
                "octomil.engines.ort_engine._has_onnxruntime_genai",
                return_value=False,
            ),
            patch(
                "octomil.engines.ort_engine._has_onnxruntime",
                return_value=True,
            ),
            patch.dict("sys.modules", {"onnxruntime": mock_ort}),
            patch.object(
                ONNXRuntimeEngine,
                "_resolve_model_path",
                return_value="/tmp/model.onnx",
            ),
        ):
            backend = _ORTBackend("model.onnx")
            backend.load_model("model.onnx")
            assert backend._use_genai is False
            assert backend._session is mock_session

    def test_load_model_nothing_available(self) -> None:
        with (
            patch(
                "octomil.engines.ort_engine._has_onnxruntime_genai",
                return_value=False,
            ),
            patch(
                "octomil.engines.ort_engine._has_onnxruntime",
                return_value=False,
            ),
            patch.object(
                ONNXRuntimeEngine,
                "_resolve_model_path",
                return_value="/tmp/model",
            ),
        ):
            backend = _ORTBackend("test-model")
            with pytest.raises(RuntimeError, match="onnxruntime is not installed"):
                backend.load_model("test-model")

    def test_generate_genai(self) -> None:
        mock_og = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Hello world"

        mock_generator = MagicMock()
        # is_done() returns False 3 times then True
        mock_generator.is_done.side_effect = [False, False, False, True]
        mock_generator.get_next_tokens.return_value = [42]
        mock_og.Generator.return_value = mock_generator
        mock_og.GeneratorParams.return_value = MagicMock()

        backend = _ORTBackend("test-model")
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer
        backend._use_genai = True

        @dataclass
        class FakeRequest:
            messages: list[dict[str, str]]
            max_tokens: int = 512

        request = FakeRequest(messages=[{"role": "user", "content": "Hi"}])

        with patch.dict("sys.modules", {"onnxruntime_genai": mock_og}):
            text, metrics = backend.generate(request)

        assert text == "Hello world"
        assert metrics.total_tokens == 3
        assert metrics.tokens_per_second > 0

    def test_generate_session(self) -> None:
        import numpy as np

        mock_input = MagicMock()
        mock_input.name = "input_ids"
        mock_input.shape = [1, 10]
        mock_input.type = "tensor(int64)"

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.array([[1, 2, 3]])]

        backend = _ORTBackend("test-model")
        backend._session = mock_session
        backend._use_genai = False

        @dataclass
        class FakeRequest:
            messages: list[dict[str, str]]
            max_tokens: int = 512

        request = FakeRequest(messages=[{"role": "user", "content": "Hi"}])

        text, metrics = backend.generate(request)
        assert text  # Non-empty output
        assert metrics.total_tokens >= 1

    def test_generate_auto_loads_model(self) -> None:
        """generate() calls load_model when no model is loaded."""
        backend = _ORTBackend("test-model")

        @dataclass
        class FakeRequest:
            messages: list[dict[str, str]]
            max_tokens: int = 32

        request = FakeRequest(messages=[{"role": "user", "content": "Hi"}])

        with patch.object(backend, "load_model") as mock_load:
            # After load_model, set up a fake session
            def setup_session(name: str) -> None:
                import numpy as np

                mock_input = MagicMock()
                mock_input.name = "input_ids"
                mock_input.shape = [1, 10]
                mock_input.type = "tensor(float)"
                mock_session = MagicMock()
                mock_session.get_inputs.return_value = [mock_input]
                mock_session.run.return_value = [np.array([[0.5]])]
                backend._session = mock_session
                backend._use_genai = False

            mock_load.side_effect = setup_session
            text, metrics = backend.generate(request)
            mock_load.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    async def test_generate_stream_genai(self) -> None:
        mock_og = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "tok"

        mock_generator = MagicMock()
        # Generate 2 tokens then done
        mock_generator.is_done.side_effect = [False, False, True]
        mock_generator.get_next_tokens.return_value = [42]
        mock_og.Generator.return_value = mock_generator
        mock_og.GeneratorParams.return_value = MagicMock()

        backend = _ORTBackend("test-model")
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer
        backend._use_genai = True

        @dataclass
        class FakeRequest:
            messages: list[dict[str, str]]
            max_tokens: int = 512

        request = FakeRequest(messages=[{"role": "user", "content": "Hi"}])

        chunks = []
        with patch.dict("sys.modules", {"onnxruntime_genai": mock_og}):
            async for chunk in backend.generate_stream(request):
                chunks.append(chunk)

        # Should have token chunks + final stop chunk
        assert len(chunks) >= 1
        assert any(c.finish_reason == "stop" for c in chunks)

    @pytest.mark.asyncio
    async def test_generate_stream_session_fallback(self) -> None:
        import numpy as np

        mock_input = MagicMock()
        mock_input.name = "input_ids"
        mock_input.shape = [1, 10]
        mock_input.type = "tensor(float)"

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.array([[0.5]])]

        backend = _ORTBackend("test-model")
        backend._session = mock_session
        backend._use_genai = False

        @dataclass
        class FakeRequest:
            messages: list[dict[str, str]]
            max_tokens: int = 32

        request = FakeRequest(messages=[{"role": "user", "content": "Hi"}])

        chunks = []
        async for chunk in backend.generate_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestORTRegistry:
    def test_engine_registered(self) -> None:
        from octomil.engines.registry import EngineRegistry

        registry = EngineRegistry()
        engine = ONNXRuntimeEngine()
        registry.register(engine)
        assert registry.get_engine("onnxruntime") is engine

    def test_auto_register_includes_onnxruntime(self) -> None:
        from octomil.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)
        assert registry.get_engine("onnxruntime") is not None

    def test_global_registry_has_onnxruntime(self) -> None:
        from octomil.engines.registry import get_registry, reset_registry

        reset_registry()
        try:
            reg = get_registry()
            names = [e.name for e in reg.engines]
            assert "onnxruntime" in names
        finally:
            reset_registry()

    def test_priority_ordering(self) -> None:
        """onnxruntime (30) should be after llama.cpp (20) and before echo (999)."""
        from octomil.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)

        ort = registry.get_engine("onnxruntime")
        llamacpp = registry.get_engine("llama.cpp")
        echo = registry.get_engine("echo")

        assert ort is not None
        assert llamacpp is not None
        assert echo is not None
        assert llamacpp.priority < ort.priority < echo.priority


# ---------------------------------------------------------------------------
# Catalog integration
# ---------------------------------------------------------------------------


class TestORTCatalog:
    def test_ort_in_catalog_engines(self) -> None:
        from octomil.models.catalog import CATALOG

        # Check that at least some models have onnxruntime
        ort_models = [
            name for name, entry in CATALOG.items() if "onnxruntime" in entry.engines
        ]
        assert len(ort_models) > 0
        assert "gemma-1b" in ort_models
        assert "llama-1b" in ort_models

    def test_variant_spec_has_ort_field(self) -> None:
        from octomil.models.catalog import VariantSpec

        spec = VariantSpec(ort="microsoft/model-onnx")
        assert spec.ort == "microsoft/model-onnx"

    def test_variant_spec_ort_default_none(self) -> None:
        from octomil.models.catalog import VariantSpec

        spec = VariantSpec()
        assert spec.ort is None

    def test_ort_catalog_set_populated(self) -> None:
        from octomil.engines.ort_engine import _ORT_CATALOG

        assert len(_ORT_CATALOG) > 0
        assert "gemma-1b" in _ORT_CATALOG


# ---------------------------------------------------------------------------
# Resolver integration
# ---------------------------------------------------------------------------


class TestORTResolver:
    def test_engine_alias_ort(self) -> None:
        from octomil.models.resolver import _normalize_engine

        assert _normalize_engine("ort") == "onnxruntime"
        assert _normalize_engine("onnx") == "onnxruntime"
        assert _normalize_engine("onnxruntime") == "onnxruntime"

    def test_engine_in_priority(self) -> None:
        from octomil.models.resolver import _ENGINE_PRIORITY

        assert "onnxruntime" in _ENGINE_PRIORITY

    def test_resolve_with_onnxruntime_engine(self) -> None:
        from octomil.models.resolver import resolve

        # gemma-1b has onnxruntime in its engines set and has source_repo
        result = resolve("gemma-1b", engine="onnxruntime")
        assert result.engine == "onnxruntime"
        assert result.hf_repo  # Should resolve to some repo
