"""LEGACY pywhispercpp engine — DO NOT use on product paths.

Reference / benchmark only. v0.1.5 PR-2B retired this engine from the
production registry. The product STT path now routes through
:class:`octomil.runtime.native.stt_backend.NativeSttBackend` (cffi
bindings into octomil-runtime + whisper.cpp), with a hard-cutover
contract: no silent fallback to pywhispercpp at runtime.

This module remains for benchmarking / parity comparison ONLY. It is
gated behind the opt-in env var ``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK=1``
in the parity gate (:mod:`scripts.parity_native_stt`); production code
must NOT import from here. ``WhisperCppEngine`` is kept as the class
name so the parity script can construct it without further
indirection.

v0.1.6 PR2 moved :func:`is_whisper_model` and ``_WHISPER_MODELS`` into
the non-legacy module
:mod:`octomil.runtime.engines.whisper.model_names`, and the package
``__init__`` now re-exports from there. This module re-exports them
too (purely for parity / benchmark callers that already imported them
from this dotted path) but is no longer the source of truth.

Guard test
----------

``tests/test_no_legacy_pywhisper_in_product_paths.py`` (v0.1.6 PR2)
asserts statically that no production module imports this file by
dotted path, by relative path, or by dynamic
(``importlib.import_module`` / ``__import__``) form, and that no
product path imports the legacy inference symbols
(``WhisperCppEngine``, ``_WhisperBackend``) by name. The runtime
probe asserts that importing the canonical product entry points
(``octomil``, ``octomil.serve.app``, ``octomil.execution.kernel``)
does NOT bring this module into ``sys.modules``. Set
``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK=1`` to opt in for parity
comparison or research benchmarking only.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any

from octomil.runtime.core.base import BenchmarkResult, EnginePlugin
from octomil.runtime.engines.whisper.model_names import (
    _WHISPER_MODELS,
    is_whisper_model,
)

logger = logging.getLogger(__name__)

# v0.1.6 PR2: ``_WHISPER_MODELS`` and ``is_whisper_model`` moved into
# :mod:`octomil.runtime.engines.whisper.model_names` so the product
# path can detect whisper model names without pulling this legacy
# module into ``sys.modules``. They are re-exported here for parity /
# benchmark code paths that still expect to find them on this module.
__all__ = [
    "_WHISPER_MODELS",
    "is_whisper_model",
    "WhisperCppEngine",
]


def _has_pywhispercpp() -> bool:
    """Check if the pywhispercpp package is importable."""
    try:
        import pywhispercpp  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def _get_whisper_version() -> str:
    """Return pywhispercpp version string, or empty if unavailable."""
    try:
        import pywhispercpp  # type: ignore[import-untyped]

        return getattr(pywhispercpp, "__version__", "unknown")
    except ImportError:
        return ""


class WhisperCppEngine(EnginePlugin):
    """Speech-to-text engine using whisper.cpp via pywhispercpp."""

    @property
    def name(self) -> str:
        return "whisper.cpp"

    @property
    def display_name(self) -> str:
        return "Whisper.cpp (Speech-to-Text)"

    @property
    def priority(self) -> int:
        return 35  # After onnxruntime (30), before echo (999)

    def detect(self) -> bool:
        return _has_pywhispercpp()

    def detect_info(self) -> str:
        version = _get_whisper_version()
        if not version:
            return ""
        models = ", ".join(sorted(_WHISPER_MODELS.keys()))
        return f"pywhispercpp {version}; models: {models}"

    def supports_model(self, model_name: str) -> bool:
        # Whisper models have their own naming; no alias resolution needed
        # but include it for consistency
        return is_whisper_model(model_name)

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        """Benchmark by transcribing a short silent audio segment.

        For Whisper, ``tokens_per_second`` is repurposed as
        ``audio_seconds_per_second`` (real-time factor).
        """
        if not _has_pywhispercpp():
            return BenchmarkResult(engine_name=self.name, error="pywhispercpp not available")

        if not is_whisper_model(model_name):
            return BenchmarkResult(
                engine_name=self.name,
                error=f"Unsupported model: {model_name}",
            )

        try:
            from pywhispercpp.model import Model  # type: ignore[import-untyped]

            whisper_size = _WHISPER_MODELS[model_name.lower()]

            # Generate a short silent WAV file for benchmarking
            silent_path = _generate_silent_wav(duration_s=3.0)
            try:
                model = Model(whisper_size)

                start = time.monotonic()
                model.transcribe(silent_path)
                elapsed = time.monotonic() - start

                audio_duration = 3.0
                realtime_factor = audio_duration / elapsed if elapsed > 0 else 0.0

                return BenchmarkResult(
                    engine_name=self.name,
                    tokens_per_second=realtime_factor,
                    metadata={
                        "method": "transcribe",
                        "audio_seconds_per_second": realtime_factor,
                        "audio_duration_s": audio_duration,
                        "elapsed_s": round(elapsed, 3),
                        "whisper_size": whisper_size,
                    },
                )
            finally:
                os.unlink(silent_path)

        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        return _WhisperBackend(model_name, **kwargs)


class _WhisperBackend:
    """Speech-to-text backend using whisper.cpp.

    Unlike LLM backends, this does NOT implement ``generate()`` or
    ``generate_stream()``.  Instead it provides ``transcribe()`` for
    audio-to-text conversion.  The serve layer adds a dedicated
    ``/v1/audio/transcriptions`` endpoint.
    """

    name = "whisper.cpp"

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._kwargs = kwargs
        # PrepareManager passes ``model_dir`` when the planner has
        # materialized the artifact under <cache>/artifacts/<id>/. With
        # it, ``load_model`` skips pywhispercpp's own download path.
        self._injected_model_dir: str | None = kwargs.get("model_dir")
        self._model: Any = None

    def load_model(self, model_name: str) -> None:
        """Download (if needed) and load a Whisper model.

        When ``model_dir`` was injected, the backend loads from a
        whisper.cpp-compatible file inside that directory (.bin / .gguf
        / .ggml). Otherwise it falls back to the canonical whisper-size
        string and lets ``pywhispercpp`` resolve the download.
        """
        self._model_name = model_name
        whisper_size = _WHISPER_MODELS.get(model_name.lower())
        if whisper_size is None:
            raise ValueError(f"Unknown whisper model '{model_name}'. Available: {', '.join(sorted(_WHISPER_MODELS))}")

        from pywhispercpp.model import Model  # type: ignore[import-untyped]

        model_path = self._resolve_local_model_file()
        if model_path:
            logger.info(
                "Loading whisper model from prepared dir: %s (%s)",
                model_name,
                model_path,
            )
            self._model = Model(model_path)
        else:
            logger.info("Loading whisper model: %s (%s)", model_name, whisper_size)
            self._model = Model(whisper_size)
        logger.info("Whisper model loaded: %s", model_name)

    def _resolve_local_model_file(self) -> str | None:
        """Return the path to a Whisper model file inside the injected
        ``model_dir``, or ``None`` if no dir was injected.

        Resolution order:

        1. The PrepareManager sentinel ``<dir>/artifact`` — that is the
           canonical single-file output when the planner emits an empty
           ``required_files`` list. The earlier draft of this resolver
           only accepted files ending in ``.bin`` / ``.gguf`` / ``.ggml``
           and silently fell through to pywhispercpp's HF download path
           even when prepared bytes were already on disk.
        2. Any file at the top level whose extension matches a
           whisper.cpp-recognized format. Covers the multi-file artifact
           case and the legacy single-file case where the planner names
           the file via ``required_files``.

        Multi-file Whisper artifacts are not emitted by the planner
        today; when they are, the manifest entry will carry which file
        is the model proper and this resolver becomes manifest-aware.
        """
        if not self._injected_model_dir:
            return None
        model_dir = self._injected_model_dir
        if not os.path.isdir(model_dir):
            return None
        # 1. PrepareManager single-file sentinel.
        sentinel = os.path.join(model_dir, "artifact")
        if os.path.isfile(sentinel):
            return sentinel
        # 2. Any whisper.cpp-recognized extension at the top level.
        for entry in sorted(os.listdir(model_dir)):
            lower = entry.lower()
            if lower.endswith((".bin", ".gguf", ".ggml")):
                return os.path.join(model_dir, entry)
        return None

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        """Transcribe an audio file and return text + segments.

        Returns an OpenAI Whisper API-compatible dict::

            {
                "text": "The quick brown fox...",
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "The quick brown fox"},
                    ...
                ]
            }
        """
        if self._model is None:
            self.load_model(self._model_name)
        assert self._model is not None

        segments_raw = self._model.transcribe(audio_path)

        segments: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for seg in segments_raw:
            # pywhispercpp segments have t0, t1 (timestamps in ms) and text
            start = getattr(seg, "t0", 0) / 100.0
            end = getattr(seg, "t1", 0) / 100.0
            text = getattr(seg, "text", str(seg)).strip()
            if text:
                segments.append(
                    {
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "text": text,
                    }
                )
                text_parts.append(text)

        full_text = " ".join(text_parts)

        return {
            "text": full_text,
            "segments": segments,
        }

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []


def _generate_silent_wav(duration_s: float = 3.0, sample_rate: int = 16000) -> str:
    """Generate a silent WAV file for benchmarking. Returns the temp file path."""
    import struct
    import wave

    num_frames = int(sample_rate * duration_s)
    silence = struct.pack(f"<{num_frames}h", *([0] * num_frames))

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(silence)

    return path
