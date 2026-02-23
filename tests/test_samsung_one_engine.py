"""Tests for Samsung ONE engine plugin."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from octomil.engines.samsung_one_engine import (
    SamsungOneEngine,
    _has_onert,
    _get_onert_version,
    _select_backend,
)


class TestSamsungOneEngine:
    def setup_method(self) -> None:
        self.engine = SamsungOneEngine(backend="cpu")

    def test_name(self) -> None:
        assert self.engine.name == "samsung-one"

    def test_display_name_cpu(self) -> None:
        engine = SamsungOneEngine(backend="cpu")
        assert engine.display_name == "Samsung ONE (CPU)"

    def test_display_name_npu(self) -> None:
        engine = SamsungOneEngine(backend="npu")
        assert "Exynos NPU" in engine.display_name

    def test_display_name_gpu(self) -> None:
        engine = SamsungOneEngine(backend="gpu")
        assert "Mali GPU" in engine.display_name

    def test_priority(self) -> None:
        assert self.engine.priority == 18

    def test_backend_property(self) -> None:
        engine = SamsungOneEngine(backend="npu")
        assert engine.backend == "npu"

    def test_detect_with_onert(self) -> None:
        with patch(
            "octomil.engines.samsung_one_engine._has_onert", return_value=True
        ):
            assert self.engine.detect() is True

    def test_detect_without_onert(self) -> None:
        with patch(
            "octomil.engines.samsung_one_engine._has_onert", return_value=False
        ):
            assert self.engine.detect() is False

    def test_detect_info_available(self) -> None:
        with (
            patch(
                "octomil.engines.samsung_one_engine._has_onert", return_value=True
            ),
            patch(
                "octomil.engines.samsung_one_engine._get_onert_version",
                return_value="1.31.0",
            ),
        ):
            info = self.engine.detect_info()
            assert "backend: cpu" in info
            assert "v1.31.0" in info

    def test_detect_info_no_version(self) -> None:
        with (
            patch(
                "octomil.engines.samsung_one_engine._has_onert", return_value=True
            ),
            patch(
                "octomil.engines.samsung_one_engine._get_onert_version",
                return_value=None,
            ),
        ):
            info = self.engine.detect_info()
            assert "backend: cpu" in info
            assert "v" not in info

    def test_detect_info_unavailable(self) -> None:
        with patch(
            "octomil.engines.samsung_one_engine._has_onert", return_value=False
        ):
            assert self.engine.detect_info() == ""

    def test_supports_nnpackage(self) -> None:
        assert self.engine.supports_model("model.nnpackage") is True

    def test_supports_circle(self) -> None:
        assert self.engine.supports_model("model.circle") is True

    def test_supports_tflite(self) -> None:
        assert self.engine.supports_model("model.tflite") is True

    def test_supports_nnpackage_directory(self) -> None:
        with patch("os.path.isdir", return_value=True), patch(
            "os.path.isfile", return_value=True
        ):
            assert self.engine.supports_model("/path/to/my_model") is True

    def test_does_not_support_unknown(self) -> None:
        with patch("os.path.isdir", return_value=False):
            assert self.engine.supports_model("unknown-model") is False

    def test_does_not_support_gguf(self) -> None:
        with patch("os.path.isdir", return_value=False):
            assert self.engine.supports_model("model.gguf") is False

    def test_benchmark_unavailable(self) -> None:
        with patch(
            "octomil.engines.samsung_one_engine._has_onert", return_value=False
        ):
            result = self.engine.benchmark("model.nnpackage")
            assert result.ok is False
            assert "not available" in result.error

    def test_benchmark_with_onert(self) -> None:
        mock_info = MagicMock()
        mock_info.rank = 2
        mock_info.dims = [1, 10]
        mock_info.dtype = "float32"

        mock_session_instance = MagicMock()
        mock_session_instance.get_inputs_tensorinfo.return_value = [mock_info]
        mock_session_instance.infer.return_value = [MagicMock()]

        mock_infer = MagicMock()
        mock_infer.session.return_value = mock_session_instance

        mock_onert = MagicMock()
        mock_onert.infer = mock_infer

        with (
            patch(
                "octomil.engines.samsung_one_engine._has_onert", return_value=True
            ),
            patch(
                "octomil.engines.samsung_one_engine.SamsungOneEngine._resolve_model_path",
                return_value="/tmp/model.nnpackage",
            ),
            patch.dict(
                "sys.modules",
                {"onert": mock_onert, "onert.infer": mock_infer},
            ),
        ):
            result = self.engine.benchmark("model.nnpackage", n_tokens=5)
            assert result.engine_name == "samsung-one"
            assert result.ok is True
            # Session.infer is called 3 warmups + 5 measured = 8 times
            assert mock_session_instance.infer.call_count == 8

    def test_benchmark_exception(self) -> None:
        mock_onert = MagicMock()
        mock_infer = MagicMock()
        mock_onert.infer = mock_infer

        with (
            patch(
                "octomil.engines.samsung_one_engine._has_onert", return_value=True
            ),
            patch(
                "octomil.engines.samsung_one_engine.SamsungOneEngine._resolve_model_path",
                side_effect=FileNotFoundError("not found"),
            ),
            patch.dict(
                "sys.modules",
                {"onert": mock_onert, "onert.infer": mock_infer},
            ),
        ):
            result = self.engine.benchmark("nonexistent.nnpackage")
            assert result.ok is False
            assert "not found" in result.error

    def test_create_backend_without_onert(self) -> None:
        with patch(
            "octomil.engines.samsung_one_engine._has_onert", return_value=False
        ):
            with pytest.raises(RuntimeError, match="onert package is required"):
                self.engine.create_backend("model.nnpackage")

    def test_create_backend_with_onert(self) -> None:
        with patch(
            "octomil.engines.samsung_one_engine._has_onert", return_value=True
        ):
            backend = self.engine.create_backend("model.nnpackage")
            assert backend.model_name == "model.nnpackage"

    def test_create_backend_custom_backend(self) -> None:
        with patch(
            "octomil.engines.samsung_one_engine._has_onert", return_value=True
        ):
            backend = self.engine.create_backend(
                "model.nnpackage", backend="npu"
            )
            assert backend._backend == "npu"

    def test_resolve_model_path_exists(self) -> None:
        with patch("os.path.exists", return_value=True):
            assert self.engine._resolve_model_path("/tmp/m.nnpackage") == "/tmp/m.nnpackage"

    def test_resolve_model_path_not_found(self) -> None:
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="No nnpackage found"):
                self.engine._resolve_model_path("nonexistent-model")


class TestSelectBackend:
    def test_npu_when_mali_device_exists(self) -> None:
        with patch("os.path.exists", side_effect=lambda p: p == "/dev/mali0"):
            assert _select_backend() == "npu"

    def test_npu_when_npu_sysfs_exists(self) -> None:
        with patch("os.path.exists", side_effect=lambda p: p == "/sys/class/npu"):
            assert _select_backend() == "npu"

    def test_npu_when_vertex_exists(self) -> None:
        with patch("os.path.exists", side_effect=lambda p: p == "/dev/vertex0"):
            assert _select_backend() == "npu"

    def test_cpu_when_no_npu_indicators(self) -> None:
        with patch("os.path.exists", return_value=False):
            assert _select_backend() == "cpu"


class TestHasOnert:
    def test_available(self) -> None:
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                MagicMock() if name == "onert" else __import__(name, *a, **kw)
            ),
        ):
            assert _has_onert() is True

    def test_unavailable(self) -> None:
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError())
                if name == "onert"
                else __import__(name, *a, **kw)
            ),
        ):
            assert _has_onert() is False


class TestGetOnertVersion:
    def test_returns_version(self) -> None:
        mock_onert = MagicMock()
        mock_onert.__version__ = "1.31.0"
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                mock_onert if name == "onert" else __import__(name, *a, **kw)
            ),
        ):
            assert _get_onert_version() == "1.31.0"

    def test_returns_none_when_unavailable(self) -> None:
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError())
                if name == "onert"
                else __import__(name, *a, **kw)
            ),
        ):
            assert _get_onert_version() is None


class TestSamsungOneRegistry:
    def test_engine_registered(self) -> None:
        from octomil.engines.registry import EngineRegistry

        registry = EngineRegistry()
        engine = SamsungOneEngine()
        registry.register(engine)
        assert registry.get_engine("samsung-one") is engine

    def test_auto_register_includes_samsung_one(self) -> None:
        from octomil.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)
        assert registry.get_engine("samsung-one") is not None


class TestSamsungOneBackend:
    def test_list_models(self) -> None:
        with patch(
            "octomil.engines.samsung_one_engine._has_onert", return_value=True
        ):
            backend = SamsungOneEngine(backend="cpu").create_backend(
                "model.nnpackage"
            )
            assert backend.list_models() == ["model.nnpackage"]

    def test_list_models_empty(self) -> None:
        with patch(
            "octomil.engines.samsung_one_engine._has_onert", return_value=True
        ):
            backend = SamsungOneEngine(backend="cpu").create_backend("")
            assert backend.list_models() == []
