"""Tests for ExecuTorch engine plugin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from octomil.engines.executorch_engine import (
    ExecuTorchEngine,
    _get_best_delegate,
    _has_executorch,
)


class TestExecuTorchEngine:
    def setup_method(self) -> None:
        self.engine = ExecuTorchEngine(delegate="xnnpack")

    def test_name(self) -> None:
        assert self.engine.name == "executorch"

    def test_display_name_xnnpack(self) -> None:
        engine = ExecuTorchEngine(delegate="xnnpack")
        assert "XNNPACK" in engine.display_name

    def test_display_name_coreml(self) -> None:
        engine = ExecuTorchEngine(delegate="coreml")
        assert "CoreML" in engine.display_name

    def test_display_name_vulkan(self) -> None:
        engine = ExecuTorchEngine(delegate="vulkan")
        assert "Vulkan" in engine.display_name

    def test_display_name_qnn(self) -> None:
        engine = ExecuTorchEngine(delegate="qnn")
        assert "Qualcomm" in engine.display_name

    def test_priority(self) -> None:
        assert self.engine.priority == 25

    def test_delegate_property(self) -> None:
        engine = ExecuTorchEngine(delegate="coreml")
        assert engine.delegate == "coreml"

    def test_detect_with_executorch(self) -> None:
        with patch(
            "octomil.engines.executorch_engine._has_executorch", return_value=True
        ):
            assert self.engine.detect() is True

    def test_detect_without_executorch(self) -> None:
        with patch(
            "octomil.engines.executorch_engine._has_executorch", return_value=False
        ):
            assert self.engine.detect() is False

    def test_detect_info_available(self) -> None:
        mock_et = MagicMock()
        mock_et.__version__ = "0.4.0"
        with (
            patch(
                "octomil.engines.executorch_engine._has_executorch", return_value=True
            ),
            patch.dict("sys.modules", {"executorch": mock_et}),
        ):
            info = self.engine.detect_info()
            assert "delegate: xnnpack" in info

    def test_detect_info_unavailable(self) -> None:
        with patch(
            "octomil.engines.executorch_engine._has_executorch", return_value=False
        ):
            assert self.engine.detect_info() == ""

    def test_supports_catalog_model(self) -> None:
        assert self.engine.supports_model("llama-3b") is True
        assert self.engine.supports_model("gemma-4b") is True

    def test_supports_pte_file(self) -> None:
        assert self.engine.supports_model("model.pte") is True

    def test_supports_hf_repo(self) -> None:
        assert self.engine.supports_model("meta-llama/Llama-3-8B") is True

    def test_does_not_support_unknown(self) -> None:
        assert self.engine.supports_model("unknown-model") is False

    def test_benchmark_error_when_unavailable(self) -> None:
        with patch(
            "octomil.engines.executorch_engine._has_executorch", return_value=False
        ):
            # benchmark will try to import and fail
            result = self.engine.benchmark("llama-3b")
            assert result.ok is False

    def test_benchmark_returns_result(self) -> None:
        mock_runtime = MagicMock()
        mock_method = MagicMock()
        mock_method.execute.return_value = MagicMock()
        mock_program = MagicMock()
        mock_program.load_method.return_value = mock_method
        mock_runtime.load_program.return_value = mock_program

        with (
            patch(
                "octomil.engines.executorch_engine._has_executorch", return_value=True
            ),
            patch(
                "octomil.engines.executorch_engine.ExecuTorchEngine._resolve_model_path",
                return_value="/tmp/model.pte",
            ),
        ):
            # The benchmark will fail with ImportError for executorch.runtime
            # since we can't fully mock the import chain, but it should return a BenchmarkResult
            result = self.engine.benchmark("llama-3b")
            assert result.engine_name == "executorch"

    def test_create_backend(self) -> None:
        backend = self.engine.create_backend("llama-3b")
        assert backend.model_name == "llama-3b"

    def test_create_backend_custom_delegate(self) -> None:
        backend = self.engine.create_backend("llama-3b", delegate="coreml")
        assert backend._delegate == "coreml"

    def test_resolve_model_path_pte_file(self) -> None:
        assert self.engine._resolve_model_path("model.pte") == "model.pte"

    def test_resolve_model_path_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="No .pte model found"):
            self.engine._resolve_model_path("nonexistent-model")


class TestGetBestDelegate:
    def test_macos_prefers_coreml(self) -> None:
        with patch(
            "octomil.engines.executorch_engine.platform.system", return_value="Darwin"
        ):
            assert _get_best_delegate() == "coreml"

    def test_linux_default_xnnpack(self) -> None:
        with (
            patch(
                "octomil.engines.executorch_engine.platform.system",
                return_value="Linux",
            ),
            patch("os.path.exists", return_value=False),
        ):
            assert _get_best_delegate() == "xnnpack"

    def test_linux_qualcomm_prefers_qnn(self) -> None:
        with (
            patch(
                "octomil.engines.executorch_engine.platform.system",
                return_value="Linux",
            ),
            patch("os.path.exists", return_value=True),
        ):
            assert _get_best_delegate() == "qnn"

    def test_windows_defaults_xnnpack(self) -> None:
        with patch(
            "octomil.engines.executorch_engine.platform.system", return_value="Windows"
        ):
            assert _get_best_delegate() == "xnnpack"


class TestHasExecutorch:
    def test_available(self) -> None:
        with patch.dict("sys.modules", {"executorch": MagicMock()}):
            # _has_executorch does a real import, so we need to patch differently
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *a, **kw: (
                    MagicMock() if name == "executorch" else __import__(name, *a, **kw)
                ),
            ):
                assert _has_executorch() is True

    def test_unavailable(self) -> None:
        # Ensure executorch is not importable
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError())
                if name == "executorch"
                else __import__(name, *a, **kw)
            ),
        ):
            assert _has_executorch() is False


class TestExecuTorchRegistry:
    def test_engine_registered(self) -> None:
        from octomil.engines.registry import EngineRegistry

        registry = EngineRegistry()
        engine = ExecuTorchEngine()
        registry.register(engine)
        assert registry.get_engine("executorch") is engine

    def test_auto_register_includes_executorch(self) -> None:
        from octomil.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)
        assert registry.get_engine("executorch") is not None


class TestExecuTorchBackend:
    def test_list_models(self) -> None:
        backend = ExecuTorchEngine(delegate="xnnpack").create_backend("llama-3b")
        assert backend.list_models() == ["llama-3b"]

    def test_list_models_empty(self) -> None:
        backend = ExecuTorchEngine(delegate="xnnpack").create_backend("")
        assert backend.list_models() == []
