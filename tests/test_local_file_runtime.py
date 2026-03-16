"""Tests for octomil.runtime.engines.local_file_runtime — LocalFileModelRuntime."""

from __future__ import annotations

from pathlib import Path

from octomil.runtime.core.types import RuntimeCapabilities
from octomil.runtime.engines.local_file_runtime import LocalFileModelRuntime


class TestLocalFileRuntime:
    def test_init(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_text("fake")

        rt = LocalFileModelRuntime(model_id="test-model", file_path=model_file)
        assert rt.model_id == "test-model"
        assert rt.file_path == model_file

    def test_capabilities_default(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_text("fake")

        rt = LocalFileModelRuntime(model_id="test-model", file_path=model_file)
        assert rt.capabilities.supports_streaming is True

    def test_capabilities_custom(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_text("fake")

        caps = RuntimeCapabilities(supports_streaming=False, max_context_length=2048)
        rt = LocalFileModelRuntime(model_id="test-model", file_path=model_file, capabilities=caps)
        assert rt.capabilities.supports_streaming is False
        assert rt.capabilities.max_context_length == 2048

    def test_close(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_text("fake")

        rt = LocalFileModelRuntime(model_id="test-model", file_path=model_file)
        rt.close()
        assert rt._engine is None
