"""Tests for MNN-LLM engine plugin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from octomil.engines.mnn_engine import MNNEngine, _parse_tps_from_output


class TestMNNEngine:
    def setup_method(self) -> None:
        self.engine = MNNEngine()

    def test_name(self) -> None:
        assert self.engine.name == "mnn"

    def test_display_name_macos(self) -> None:
        with patch("octomil.engines.mnn_engine.platform.system", return_value="Darwin"):
            assert "Metal" in self.engine.display_name

    def test_display_name_linux(self) -> None:
        with patch("octomil.engines.mnn_engine.platform.system", return_value="Linux"):
            assert "Vulkan" in self.engine.display_name

    def test_display_name_windows(self) -> None:
        with patch("octomil.engines.mnn_engine.platform.system", return_value="Windows"):
            assert "CPU" in self.engine.display_name

    def test_priority(self) -> None:
        assert self.engine.priority == 15

    def test_detect_with_pymnn(self) -> None:
        with patch("octomil.engines.mnn_engine._has_pymnn", return_value=True):
            assert self.engine.detect() is True

    def test_detect_with_cli(self) -> None:
        with (
            patch("octomil.engines.mnn_engine._has_pymnn", return_value=False),
            patch(
                "octomil.engines.mnn_engine._find_mnn_cli",
                return_value="/usr/bin/mnn-llm",
            ),
        ):
            assert self.engine.detect() is True

    def test_detect_nothing_available(self) -> None:
        with (
            patch("octomil.engines.mnn_engine._has_pymnn", return_value=False),
            patch("octomil.engines.mnn_engine._find_mnn_cli", return_value=None),
        ):
            assert self.engine.detect() is False

    def test_detect_info(self) -> None:
        with (
            patch("octomil.engines.mnn_engine._has_pymnn", return_value=True),
            patch(
                "octomil.engines.mnn_engine._find_mnn_cli",
                return_value="/usr/bin/mnn-llm",
            ),
            patch("octomil.engines.mnn_engine.platform.system", return_value="Darwin"),
        ):
            info = self.engine.detect_info()
            assert "Python bindings" in info
            assert "Metal" in info

    def test_supports_catalog_model(self) -> None:
        assert self.engine.supports_model("gemma-4b") is True
        assert self.engine.supports_model("llama-8b") is True

    def test_supports_gguf_file(self) -> None:
        assert self.engine.supports_model("model.gguf") is True

    def test_supports_mnn_file(self) -> None:
        assert self.engine.supports_model("model.mnn") is True

    def test_supports_hf_repo(self) -> None:
        assert self.engine.supports_model("google/gemma-2b") is True

    def test_does_not_support_unknown(self) -> None:
        assert self.engine.supports_model("unknown-model") is False

    def test_benchmark_python(self) -> None:
        mock_model = MagicMock()
        mock_model.generate.return_value = "Hello world this is a test"

        with (
            patch("octomil.engines.mnn_engine._has_pymnn", return_value=True),
            patch(
                "octomil.engines.mnn_engine.MNNEngine._benchmark_python"
            ) as mock_bench,
        ):
            mock_bench.return_value = MagicMock(
                engine_name="mnn", tokens_per_second=50.0, ok=True
            )
            result = self.engine.benchmark("gemma-4b")
            assert result.ok

    def test_benchmark_cli(self) -> None:
        with (
            patch("octomil.engines.mnn_engine._has_pymnn", return_value=False),
            patch(
                "octomil.engines.mnn_engine._find_mnn_cli",
                return_value="/usr/bin/mnn-llm",
            ),
            patch("octomil.engines.mnn_engine.MNNEngine._benchmark_cli") as mock_bench,
        ):
            mock_bench.return_value = MagicMock(
                engine_name="mnn", tokens_per_second=45.0, ok=True
            )
            result = self.engine.benchmark("gemma-4b")
            assert result.ok

    def test_benchmark_unavailable(self) -> None:
        with (
            patch("octomil.engines.mnn_engine._has_pymnn", return_value=False),
            patch("octomil.engines.mnn_engine._find_mnn_cli", return_value=None),
        ):
            result = self.engine.benchmark("gemma-4b")
            assert result.ok is False
            assert "not available" in result.error

    def test_create_backend_with_pymnn(self) -> None:
        with patch("octomil.engines.mnn_engine._has_pymnn", return_value=True):
            backend = self.engine.create_backend("gemma-4b")
            assert backend.model_name == "gemma-4b"

    def test_create_backend_without_pymnn(self) -> None:
        with patch("octomil.engines.mnn_engine._has_pymnn", return_value=False):
            with pytest.raises(RuntimeError, match="MNN Python bindings required"):
                self.engine.create_backend("gemma-4b")


class TestParseTpsFromOutput:
    def test_parse_standard_format(self) -> None:
        assert _parse_tps_from_output("Speed: 45.2 tok/s") == pytest.approx(45.2)

    def test_parse_tokens_format(self) -> None:
        assert _parse_tps_from_output("Speed: 100.5 tokens/s") == pytest.approx(100.5)

    def test_parse_no_match(self) -> None:
        assert _parse_tps_from_output("No speed info here") == 0.0

    def test_parse_case_insensitive(self) -> None:
        assert _parse_tps_from_output("Speed: 33.7 Tok/S") == pytest.approx(33.7)

    def test_parse_integer(self) -> None:
        assert _parse_tps_from_output("Speed: 50 tok/s") == pytest.approx(50.0)


class TestMNNRegistry:
    def test_engine_registered(self) -> None:
        from octomil.engines.registry import EngineRegistry

        from octomil.engines.mnn_engine import MNNEngine

        registry = EngineRegistry()
        engine = MNNEngine()
        registry.register(engine)
        assert registry.get_engine("mnn") is engine

    def test_auto_register_includes_mnn(self) -> None:
        from octomil.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)
        assert registry.get_engine("mnn") is not None
