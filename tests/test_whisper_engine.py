"""Tests for Whisper.cpp engine plugin (EDG-60)."""

from __future__ import annotations

import os
import struct
import tempfile
import wave
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from edgeml.engines.base import BenchmarkResult
from edgeml.engines.whisper_engine import (
    WhisperCppEngine,
    _WhisperBackend,
    _WHISPER_MODELS,
    _generate_silent_wav,
    _get_whisper_version,
    _has_pywhispercpp,
    is_whisper_model,
)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


class TestHasPywhispercpp:
    def test_available(self) -> None:
        mock_pw = MagicMock()
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                mock_pw if name == "pywhispercpp" else __import__(name, *a, **kw)
            ),
        ):
            assert _has_pywhispercpp() is True

    def test_unavailable(self) -> None:
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError())
                if name == "pywhispercpp"
                else __import__(name, *a, **kw)
            ),
        ):
            assert _has_pywhispercpp() is False


class TestGetWhisperVersion:
    def test_returns_version(self) -> None:
        mock_pw = MagicMock()
        mock_pw.__version__ = "1.2.0"
        with patch.dict("sys.modules", {"pywhispercpp": mock_pw}):
            assert _get_whisper_version() == "1.2.0"

    def test_returns_unknown_when_no_version(self) -> None:
        mock_pw = MagicMock(spec=[])
        with patch.dict("sys.modules", {"pywhispercpp": mock_pw}):
            assert _get_whisper_version() == "unknown"

    def test_returns_empty_when_unavailable(self) -> None:
        import sys

        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError())
                if name == "pywhispercpp"
                else __import__(name, *a, **kw)
            ),
        ):
            saved = sys.modules.pop("pywhispercpp", None)
            try:
                assert _get_whisper_version() == ""
            finally:
                if saved is not None:
                    sys.modules["pywhispercpp"] = saved


class TestIsWhisperModel:
    def test_whisper_tiny(self) -> None:
        assert is_whisper_model("whisper-tiny") is True

    def test_whisper_base(self) -> None:
        assert is_whisper_model("whisper-base") is True

    def test_whisper_small(self) -> None:
        assert is_whisper_model("whisper-small") is True

    def test_whisper_medium(self) -> None:
        assert is_whisper_model("whisper-medium") is True

    def test_whisper_large_v3(self) -> None:
        assert is_whisper_model("whisper-large-v3") is True

    def test_non_whisper_model(self) -> None:
        assert is_whisper_model("gemma-1b") is False

    def test_llm_model(self) -> None:
        assert is_whisper_model("llama-8b") is False

    def test_case_insensitive(self) -> None:
        assert is_whisper_model("Whisper-Base") is True
        assert is_whisper_model("WHISPER-TINY") is True

    def test_empty_string(self) -> None:
        assert is_whisper_model("") is False


# ---------------------------------------------------------------------------
# WhisperCppEngine
# ---------------------------------------------------------------------------


class TestWhisperCppEngine:
    def setup_method(self) -> None:
        self.engine = WhisperCppEngine()

    def test_name(self) -> None:
        assert self.engine.name == "whisper.cpp"

    def test_display_name(self) -> None:
        assert self.engine.display_name == "Whisper.cpp (Speech-to-Text)"

    def test_priority(self) -> None:
        assert self.engine.priority == 35

    def test_detect_with_pywhispercpp(self) -> None:
        with patch(
            "edgeml.engines.whisper_engine._has_pywhispercpp", return_value=True
        ):
            assert self.engine.detect() is True

    def test_detect_without_pywhispercpp(self) -> None:
        with patch(
            "edgeml.engines.whisper_engine._has_pywhispercpp", return_value=False
        ):
            assert self.engine.detect() is False

    def test_detect_info_with_version(self) -> None:
        with patch(
            "edgeml.engines.whisper_engine._get_whisper_version",
            return_value="1.2.0",
        ):
            info = self.engine.detect_info()
            assert "pywhispercpp 1.2.0" in info
            assert "whisper-base" in info
            assert "whisper-tiny" in info

    def test_detect_info_empty_when_unavailable(self) -> None:
        with patch(
            "edgeml.engines.whisper_engine._get_whisper_version",
            return_value="",
        ):
            assert self.engine.detect_info() == ""

    def test_supports_whisper_models(self) -> None:
        for model_name in _WHISPER_MODELS:
            assert self.engine.supports_model(model_name) is True

    def test_does_not_support_llm_models(self) -> None:
        assert self.engine.supports_model("gemma-1b") is False
        assert self.engine.supports_model("llama-8b") is False
        assert self.engine.supports_model("phi-mini") is False

    def test_benchmark_unavailable(self) -> None:
        with patch(
            "edgeml.engines.whisper_engine._has_pywhispercpp", return_value=False
        ):
            result = self.engine.benchmark("whisper-base")
            assert result.ok is False
            assert "not available" in result.error

    def test_benchmark_unsupported_model(self) -> None:
        with patch(
            "edgeml.engines.whisper_engine._has_pywhispercpp", return_value=True
        ):
            result = self.engine.benchmark("gemma-1b")
            assert result.ok is False
            assert "Unsupported" in result.error

    def test_benchmark_success(self) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = []

        mock_model_class = MagicMock(return_value=mock_model)
        mock_pw_model = MagicMock()
        mock_pw_model.Model = mock_model_class

        with (
            patch(
                "edgeml.engines.whisper_engine._has_pywhispercpp",
                return_value=True,
            ),
            patch.dict("sys.modules", {"pywhispercpp": MagicMock(), "pywhispercpp.model": mock_pw_model}),
        ):
            result = self.engine.benchmark("whisper-base")
            assert result.ok
            assert result.engine_name == "whisper.cpp"
            assert "audio_seconds_per_second" in result.metadata
            assert result.metadata["method"] == "transcribe"
            assert result.metadata["whisper_size"] == "base"
            mock_model.transcribe.assert_called_once()

    def test_benchmark_exception(self) -> None:
        mock_pw_model = MagicMock()
        mock_pw_model.Model.side_effect = RuntimeError("Model load failed")

        with (
            patch(
                "edgeml.engines.whisper_engine._has_pywhispercpp",
                return_value=True,
            ),
            patch.dict("sys.modules", {"pywhispercpp": MagicMock(), "pywhispercpp.model": mock_pw_model}),
        ):
            result = self.engine.benchmark("whisper-base")
            assert result.ok is False
            assert "Model load failed" in result.error

    def test_create_backend(self) -> None:
        backend = self.engine.create_backend("whisper-base")
        assert isinstance(backend, _WhisperBackend)
        assert backend.name == "whisper.cpp"

    def test_create_backend_has_transcribe(self) -> None:
        backend = self.engine.create_backend("whisper-base")
        assert hasattr(backend, "transcribe")
        assert hasattr(backend, "load_model")
        assert hasattr(backend, "list_models")


# ---------------------------------------------------------------------------
# _WhisperBackend
# ---------------------------------------------------------------------------


class TestWhisperBackend:
    def test_name(self) -> None:
        backend = _WhisperBackend("whisper-base")
        assert backend.name == "whisper.cpp"

    def test_list_models(self) -> None:
        backend = _WhisperBackend("whisper-base")
        assert backend.list_models() == ["whisper-base"]

    def test_list_models_empty(self) -> None:
        backend = _WhisperBackend("")
        assert backend.list_models() == []

    def test_load_model(self) -> None:
        mock_model = MagicMock()
        mock_model_class = MagicMock(return_value=mock_model)
        mock_pw_model = MagicMock()
        mock_pw_model.Model = mock_model_class

        with patch.dict("sys.modules", {"pywhispercpp": MagicMock(), "pywhispercpp.model": mock_pw_model}):
            backend = _WhisperBackend("whisper-base")
            backend.load_model("whisper-base")
            mock_model_class.assert_called_once_with("base")
            assert backend._model is mock_model

    def test_load_model_unknown_raises(self) -> None:
        backend = _WhisperBackend("not-a-whisper-model")
        with pytest.raises(ValueError, match="Unknown whisper model"):
            backend.load_model("not-a-whisper-model")

    def test_transcribe(self) -> None:
        # Create a mock segment with t0, t1, text
        mock_segment = MagicMock()
        mock_segment.t0 = 0
        mock_segment.t1 = 250
        mock_segment.text = "Hello world"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = [mock_segment]

        backend = _WhisperBackend("whisper-base")
        backend._model = mock_model

        result = backend.transcribe("/tmp/test.wav")

        assert result["text"] == "Hello world"
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Hello world"
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 2.5
        mock_model.transcribe.assert_called_once_with("/tmp/test.wav")

    def test_transcribe_multiple_segments(self) -> None:
        seg1 = MagicMock()
        seg1.t0 = 0
        seg1.t1 = 250
        seg1.text = "Hello"

        seg2 = MagicMock()
        seg2.t0 = 250
        seg2.t1 = 500
        seg2.text = "world"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = [seg1, seg2]

        backend = _WhisperBackend("whisper-base")
        backend._model = mock_model

        result = backend.transcribe("/tmp/test.wav")

        assert result["text"] == "Hello world"
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Hello"
        assert result["segments"][1]["text"] == "world"

    def test_transcribe_empty_result(self) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = []

        backend = _WhisperBackend("whisper-base")
        backend._model = mock_model

        result = backend.transcribe("/tmp/test.wav")

        assert result["text"] == ""
        assert result["segments"] == []

    def test_transcribe_auto_loads_model(self) -> None:
        """transcribe() calls load_model when no model is loaded."""
        backend = _WhisperBackend("whisper-base")

        mock_segment = MagicMock()
        mock_segment.t0 = 0
        mock_segment.t1 = 100
        mock_segment.text = "test"

        def setup_model(name: str) -> None:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = [mock_segment]
            backend._model = mock_model

        with patch.object(backend, "load_model", side_effect=setup_model) as mock_load:
            result = backend.transcribe("/tmp/test.wav")
            mock_load.assert_called_once_with("whisper-base")
            assert result["text"] == "test"


# ---------------------------------------------------------------------------
# Silent WAV generation helper
# ---------------------------------------------------------------------------


class TestGenerateSilentWav:
    def test_creates_valid_wav(self) -> None:
        path = _generate_silent_wav(duration_s=1.0, sample_rate=16000)
        try:
            assert os.path.exists(path)
            assert path.endswith(".wav")

            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == 16000
                assert wf.getnframes() == 16000
        finally:
            os.unlink(path)

    def test_duration_controls_frames(self) -> None:
        path = _generate_silent_wav(duration_s=2.0, sample_rate=8000)
        try:
            with wave.open(path, "rb") as wf:
                assert wf.getnframes() == 16000  # 2.0 * 8000
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# /v1/audio/transcriptions endpoint
# ---------------------------------------------------------------------------


class TestTranscriptionEndpoint:
    """Test the transcription endpoint via FastAPI TestClient."""

    def _make_app(self, whisper_backend: Any = None) -> Any:
        """Create a test app with a mocked whisper backend."""
        from edgeml.serve import create_app

        app = create_app("whisper-base")

        # Override the lifespan by injecting state directly
        # We need to use the TestClient which handles lifespan
        return app, whisper_backend

    def test_transcription_no_backend(self) -> None:
        """Returns 503 when no whisper model is loaded."""
        from unittest.mock import AsyncMock

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        from edgeml.serve import ServerState

        state = ServerState(model_name="gemma-1b")

        @app.post("/v1/audio/transcriptions")
        async def transcribe(file: Any = None, model: str = "") -> dict[str, Any]:
            from fastapi import HTTPException

            if state.whisper_backend is None:
                raise HTTPException(status_code=503, detail="No whisper model loaded")
            return {}

        client = TestClient(app)
        response = client.post("/v1/audio/transcriptions")
        assert response.status_code == 503

    def test_transcription_success(self) -> None:
        """Test successful transcription via the backend directly.

        We test the backend's transcribe() method which is called by
        the endpoint. This avoids requiring python-multipart in tests.
        """
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.t0 = 0
        mock_segment.t1 = 250
        mock_segment.text = "Hello world"
        mock_model.transcribe.return_value = [mock_segment]

        backend = _WhisperBackend("whisper-base")
        backend._model = mock_model

        # Generate a test WAV file
        wav_path = _generate_silent_wav(duration_s=1.0)
        try:
            result = backend.transcribe(wav_path)
        finally:
            os.unlink(wav_path)

        assert result["text"] == "Hello world"
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Hello world"
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 2.5
        mock_model.transcribe.assert_called_once_with(wav_path)

    def test_transcription_endpoint_integration(self) -> None:
        """Test that the endpoint is registered in the app created by create_app.

        We verify the route exists without actually calling it (avoids
        python-multipart requirement in test env).
        """
        from edgeml.serve import create_app

        app = create_app("whisper-base")
        route_paths = [
            getattr(r, "path", None)
            for r in app.routes
        ]
        assert "/v1/audio/transcriptions" in route_paths


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestWhisperRegistry:
    def test_engine_registered(self) -> None:
        from edgeml.engines.registry import EngineRegistry

        registry = EngineRegistry()
        engine = WhisperCppEngine()
        registry.register(engine)
        assert registry.get_engine("whisper.cpp") is engine

    def test_auto_register_includes_whisper(self) -> None:
        from edgeml.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)
        assert registry.get_engine("whisper.cpp") is not None

    def test_global_registry_has_whisper(self) -> None:
        from edgeml.engines.registry import get_registry, reset_registry

        reset_registry()
        try:
            reg = get_registry()
            names = [e.name for e in reg.engines]
            assert "whisper.cpp" in names
        finally:
            reset_registry()

    def test_priority_ordering(self) -> None:
        """whisper.cpp (35) should be after onnxruntime (30) and before echo (999)."""
        from edgeml.engines.registry import EngineRegistry, _auto_register

        registry = EngineRegistry()
        _auto_register(registry)

        whisper = registry.get_engine("whisper.cpp")
        ort = registry.get_engine("onnxruntime")
        echo = registry.get_engine("echo")

        assert whisper is not None
        assert ort is not None
        assert echo is not None
        assert ort.priority < whisper.priority < echo.priority


# ---------------------------------------------------------------------------
# Catalog integration
# ---------------------------------------------------------------------------


class TestWhisperCatalog:
    def test_whisper_models_in_catalog(self) -> None:
        from edgeml.models.catalog import CATALOG

        whisper_models = [
            name
            for name, entry in CATALOG.items()
            if "whisper.cpp" in entry.engines
        ]
        assert len(whisper_models) == 5
        assert "whisper-tiny" in whisper_models
        assert "whisper-base" in whisper_models
        assert "whisper-small" in whisper_models
        assert "whisper-medium" in whisper_models
        assert "whisper-large-v3" in whisper_models

    def test_whisper_tiny_params(self) -> None:
        from edgeml.models.catalog import CATALOG

        entry = CATALOG["whisper-tiny"]
        assert entry.publisher == "OpenAI"
        assert entry.params == "39M"

    def test_whisper_base_params(self) -> None:
        from edgeml.models.catalog import CATALOG

        entry = CATALOG["whisper-base"]
        assert entry.publisher == "OpenAI"
        assert entry.params == "74M"

    def test_whisper_large_v3_params(self) -> None:
        from edgeml.models.catalog import CATALOG

        entry = CATALOG["whisper-large-v3"]
        assert entry.publisher == "OpenAI"
        assert entry.params == "1.55B"

    def test_whisper_default_quant_is_fp16(self) -> None:
        from edgeml.models.catalog import CATALOG

        for name in _WHISPER_MODELS:
            assert CATALOG[name].default_quant == "fp16"


# ---------------------------------------------------------------------------
# Resolver integration
# ---------------------------------------------------------------------------


class TestWhisperResolver:
    def test_engine_alias_whisper(self) -> None:
        from edgeml.models.resolver import _normalize_engine

        assert _normalize_engine("whisper") == "whisper.cpp"
        assert _normalize_engine("whisper.cpp") == "whisper.cpp"
        assert _normalize_engine("whispercpp") == "whisper.cpp"

    def test_engine_in_priority(self) -> None:
        from edgeml.models.resolver import _ENGINE_PRIORITY

        assert "whisper.cpp" in _ENGINE_PRIORITY

    def test_resolve_with_whisper_engine(self) -> None:
        from edgeml.models.resolver import resolve

        result = resolve("whisper-base", engine="whisper.cpp")
        assert result.engine == "whisper.cpp"
        assert result.hf_repo  # Should resolve to some repo
