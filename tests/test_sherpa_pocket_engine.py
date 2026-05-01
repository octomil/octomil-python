"""Unit tests for the PocketTTS sherpa-onnx engine family.

Covers configuration, the voice-cloning generate dispatch, and the
streaming bridge — all without a real ``sherpa_onnx`` import. A fake
sherpa module captures the constructor and ``generate`` calls so
tests can pin the exact kwargs the engine sends.
"""

from __future__ import annotations

import os
import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.execution.tts_speaker_resolver import ResolvedTtsSpeaker
from octomil.runtime.engines.sherpa.engine import (
    _POCKET_REQUIRED_FILES,
    _build_pocket_model_config,
    _load_reference_samples,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(path: Path, samples_int16: list[int], sample_rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples_int16)}h", *samples_int16))


def _make_pocket_artifact_dir(root: Path) -> Path:
    """Create a directory that looks like a prepared Pocket bundle."""
    for _, filename in _POCKET_REQUIRED_FILES:
        (root / filename).write_bytes(b"\x00" * 64)
    return root


class _FakePocketModelConfig:
    """Captures the kwargs passed to OfflineTtsPocketModelConfig."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = dict(kwargs)
        self.kwargs = kwargs


# ---------------------------------------------------------------------------
# _POCKET_REQUIRED_FILES — layout contract
# ---------------------------------------------------------------------------


def test_pocket_required_files_match_upstream_bundle_layout():
    """The seven files Pocket needs must match what the
    OfflineTtsPocketModelConfig pybind11 binding expects (matrix is
    the doc snapshot in the engine module). If sherpa-onnx changes
    the binding shape, this test will catch the drift before users
    do."""
    expected = {
        "text_conditioner": "text_conditioner.onnx",
        "encoder": "encoder.onnx",
        "lm_flow": "lm_flow.int8.onnx",
        "decoder": "decoder.int8.onnx",
        "lm_main": "lm_main.int8.onnx",
        "vocab_json": "vocab.json",
        "token_scores_json": "token_scores.json",
    }
    assert dict(_POCKET_REQUIRED_FILES) == expected


# ---------------------------------------------------------------------------
# _build_pocket_model_config — config wiring
# ---------------------------------------------------------------------------


def test_build_pocket_model_config_wires_all_seven_kwargs(tmp_path: Path):
    artifact_dir = _make_pocket_artifact_dir(tmp_path)
    sherpa = SimpleNamespace(OfflineTtsPocketModelConfig=_FakePocketModelConfig)

    _build_pocket_model_config(sherpa, str(artifact_dir))

    kwargs = _FakePocketModelConfig.last_kwargs
    assert set(kwargs.keys()) == {
        "text_conditioner",
        "encoder",
        "lm_flow",
        "decoder",
        "lm_main",
        "vocab_json",
        "token_scores_json",
    }
    # All seven paths point under artifact_dir at the right filenames.
    assert kwargs["text_conditioner"] == os.path.join(str(artifact_dir), "text_conditioner.onnx")
    assert kwargs["lm_flow"] == os.path.join(str(artifact_dir), "lm_flow.int8.onnx")
    assert kwargs["lm_main"] == os.path.join(str(artifact_dir), "lm_main.int8.onnx")
    assert kwargs["decoder"] == os.path.join(str(artifact_dir), "decoder.int8.onnx")
    assert kwargs["encoder"] == os.path.join(str(artifact_dir), "encoder.onnx")
    assert kwargs["vocab_json"] == os.path.join(str(artifact_dir), "vocab.json")
    assert kwargs["token_scores_json"] == os.path.join(str(artifact_dir), "token_scores.json")


def test_build_pocket_model_config_raises_on_missing_files(tmp_path: Path):
    """Partial bundles fail HERE with a precise error, not deep
    inside an ONNX session."""
    artifact_dir = _make_pocket_artifact_dir(tmp_path)
    # Drop one of the required files.
    (artifact_dir / "lm_main.int8.onnx").unlink()

    sherpa = SimpleNamespace(OfflineTtsPocketModelConfig=_FakePocketModelConfig)
    with pytest.raises(OctomilError) as exc_info:
        _build_pocket_model_config(sherpa, str(artifact_dir))
    assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    assert "lm_main.int8.onnx" in exc_info.value.error_message


# ---------------------------------------------------------------------------
# _load_reference_samples — WAV decoding
# ---------------------------------------------------------------------------


def test_load_reference_samples_decodes_mono_16bit_wav(tmp_path: Path):
    wav_path = tmp_path / "ref.wav"
    _write_wav(wav_path, [0, 16384, -16384, 32767, -32768], sample_rate=24000)

    samples, sr = _load_reference_samples(str(wav_path))

    assert sr == 24000
    # numpy or list — both are acceptable; check the length and the
    # known float32 conversion endpoints.
    samples_list = list(samples)
    assert len(samples_list) == 5
    assert samples_list[0] == pytest.approx(0.0, abs=1e-5)
    assert samples_list[3] == pytest.approx(0.99997, abs=1e-3)


def test_load_reference_samples_raises_on_missing_path():
    with pytest.raises(OctomilError) as exc_info:
        _load_reference_samples("/nonexistent/ref.wav")
    assert "reference_audio_not_found" in exc_info.value.error_message


def test_load_reference_samples_raises_on_empty_string():
    with pytest.raises(OctomilError) as exc_info:
        _load_reference_samples("")
    assert "speaker_profile_missing_reference" in exc_info.value.error_message


def test_load_reference_samples_raises_on_non_wav_file(tmp_path: Path):
    bogus = tmp_path / "bogus.wav"
    bogus.write_bytes(b"not a wav header at all")
    with pytest.raises(OctomilError) as exc_info:
        _load_reference_samples(str(bogus))
    assert "reference_audio_format_unsupported" in exc_info.value.error_message


# ---------------------------------------------------------------------------
# _SherpaTtsBackend.synthesize — Pocket dispatch
# ---------------------------------------------------------------------------


class _FakeOfflineTtsAudio:
    def __init__(self, samples: list[float], sample_rate: int) -> None:
        self.samples = samples
        self.sample_rate = sample_rate


class _FakeOfflineTts:
    """Captures generate() calls so tests can pin the dispatched signature."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.sample_rate = 24000
        self.calls: list[dict[str, Any]] = []

    def generate(self, *args: Any, **kwargs: Any) -> _FakeOfflineTtsAudio:
        self.calls.append({"args": args, "kwargs": kwargs})
        # Single-callback fake so the test isn't sensitive to chunking.
        return _FakeOfflineTtsAudio([0.0] * 1024, self.sample_rate)


def _make_pocket_backend(tmp_path: Path):
    """Construct the backend bypassing __init__ so we don't need a
    real sherpa_onnx import. Sets the minimum state synthesize()
    expects: model_name, family, _tts, _sample_rate."""
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    backend = _SherpaTtsBackend.__new__(_SherpaTtsBackend)
    backend._model_name = "pocket-tts-int8"  # type: ignore[attr-defined]
    backend._family = "pocket"  # type: ignore[attr-defined]
    backend._injected_model_dir = str(tmp_path)  # type: ignore[attr-defined]
    backend._kwargs = {}  # type: ignore[attr-defined]
    backend._sample_rate = 24000  # type: ignore[attr-defined]
    backend._default_voice = ""  # type: ignore[attr-defined]
    backend._tts = _FakeOfflineTts()  # type: ignore[attr-defined]
    return backend


def test_pocket_synthesize_dispatches_voice_cloning_generate(tmp_path: Path):
    """Pocket must invoke ``generate(text, prompt_text, prompt_samples,
    sample_rate, speed=, num_steps=)`` — the four-positional-argument
    voice-cloning overload — not ``generate(text, sid=, speed=)``."""
    ref = tmp_path / "ref.wav"
    _write_wav(ref, [0, 100, 200, 300], sample_rate=22050)

    backend = _make_pocket_backend(tmp_path)
    profile = ResolvedTtsSpeaker(
        speaker="madam_ambrose",
        native_voice=None,
        reference_audio=str(ref),
        reference_sample_rate=22050,
        source="planner_profile",
    )

    result = backend.synthesize("hello world", speaker_profile=profile)

    assert backend._tts.calls, "Pocket backend must call generate()"
    args = backend._tts.calls[0]["args"]
    kwargs = backend._tts.calls[0]["kwargs"]

    # Four positional args: (text, prompt_text, prompt_samples, sample_rate).
    assert args[0] == "hello world"
    assert args[1] == ""  # default empty prompt_text when no metadata['reference_text']
    assert len(list(args[2])) == 4  # samples decoded from the WAV
    assert args[3] == 22050  # WAV header sample rate
    # Keyword args: speed + num_steps (default 4).
    assert kwargs.get("speed") == 1.0
    assert kwargs.get("num_steps") == 4
    # ``sid`` MUST NOT appear — that would be the native-voice path.
    assert "sid" not in kwargs

    # Result reports the speaker label, not a native voice.
    assert result["voice"] == "madam_ambrose"


def test_pocket_synthesize_passes_reference_text_and_num_steps_from_metadata(tmp_path: Path):
    ref = tmp_path / "ref.wav"
    _write_wav(ref, [0, 50], sample_rate=24000)

    backend = _make_pocket_backend(tmp_path)
    profile = ResolvedTtsSpeaker(
        speaker="madam_ambrose",
        native_voice=None,
        reference_audio=str(ref),
        reference_sample_rate=24000,
        source="planner_profile",
    )
    profile = ResolvedTtsSpeaker(
        speaker="madam_ambrose",
        native_voice=None,
        reference_audio=str(ref),
        reference_sample_rate=24000,
        source="planner_profile",
    )
    # ``metadata`` lives off the planner-supplied profile object;
    # ResolvedTtsSpeaker doesn't carry it (the resolver discards
    # everything but the four cloning fields). Pocket reads metadata
    # off the profile directly when present, so this test sets the
    # attribute on the resolved object.
    object.__setattr__(profile, "metadata", {"reference_text": "she said hello", "num_steps": 8})

    backend.synthesize("hello world", speaker_profile=profile)

    args = backend._tts.calls[0]["args"]
    kwargs = backend._tts.calls[0]["kwargs"]
    assert args[1] == "she said hello"
    assert kwargs["num_steps"] == 8


def test_pocket_synthesize_raises_when_speaker_profile_missing_reference(tmp_path: Path):
    backend = _make_pocket_backend(tmp_path)
    profile = ResolvedTtsSpeaker(
        speaker=None,
        native_voice=None,
        reference_audio=None,  # no reference -> Pocket can't run
        reference_sample_rate=None,
        source="default",
    )

    with pytest.raises(OctomilError) as exc_info:
        backend.synthesize("hello", speaker_profile=profile)
    assert "speaker_profile_missing_reference" in exc_info.value.error_message


def test_pocket_validate_voice_returns_empty_label(tmp_path: Path):
    """Pocket has no native voice catalog; the kernel calls
    validate_voice with native_voice=None on resolver output. The
    backend must return a successful (sid=0, label) tuple so the
    streaming pre-check doesn't crash."""
    backend = _make_pocket_backend(tmp_path)
    sid, label = backend.validate_voice(None)
    assert sid == 0
    assert label == ""


# ---------------------------------------------------------------------------
# Static recipe — non-default registration
# ---------------------------------------------------------------------------


def test_pocket_static_recipe_resolvable_by_explicit_lookup():
    from octomil.runtime.lifecycle.static_recipes import get_static_recipe

    recipe = get_static_recipe("pocket-tts-int8", "tts")
    assert recipe is not None
    assert recipe.engine == "sherpa-onnx"
    assert len(recipe.files) == 1
    assert recipe.files[0].relative_path == "sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2"
    assert recipe.files[0].digest == "sha256:2f3b88823cbbb9bf0b2477ec8ae7b3fec417b3a87b6bb5f256dba66f2ad967cb"
    assert recipe.files[0].size_bytes == 98_336_520


def test_pocket_static_recipe_voice_manifest_is_empty():
    """Pocket bundle has no native voice catalog — the empty manifest
    signals to the engine that speakers come from the planner's
    ``tts_speakers`` map."""
    from octomil.runtime.lifecycle.static_recipes import get_static_recipe

    recipe = get_static_recipe("pocket-tts-int8", "tts")
    assert recipe is not None
    assert recipe.materialization.voice_manifest == ()


def test_pocket_static_recipe_is_not_in_default_registry():
    """The Pocket recipe must NOT appear in the default ``_RECIPES``
    map — it lives in the non-default opt-in table only. Apps must
    explicitly opt in via planner / app config."""
    from octomil.runtime.lifecycle.static_recipes import _NON_DEFAULT_RECIPES, _RECIPES

    assert ("pocket-tts-int8", "tts") not in _RECIPES
    assert ("pocket-tts-int8", "tts") in _NON_DEFAULT_RECIPES


def test_pocket_static_recipe_notes_call_out_non_commercial():
    """The license restriction must be in the recipe notes so it
    surfaces to anyone reading the recipe table."""
    from octomil.runtime.lifecycle.static_recipes import get_static_recipe

    recipe = get_static_recipe("pocket-tts-int8", "tts")
    assert recipe is not None
    assert "NON-COMMERCIAL" in recipe.notes
    assert "license" in recipe.notes.lower()


# ---------------------------------------------------------------------------
# streaming_capability — Pocket honesty
# ---------------------------------------------------------------------------


def test_pocket_streaming_capability_matches_kokoro_sentence_chunk_pattern():
    """Pocket uses the same per-sentence callback as Kokoro. Multi-
    sentence -> sentence_chunk; single-sentence -> final_chunk. We
    only advertise progressive when the integration probe shows
    sub-sentence chunks (gated on a real artifact)."""
    from octomil.audio.streaming import TtsStreamingMode
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    backend = _SherpaTtsBackend.__new__(_SherpaTtsBackend)

    cap_single = backend.streaming_capability("Hello there.")
    assert cap_single.mode == TtsStreamingMode.FINAL_CHUNK

    cap_multi = backend.streaming_capability("Hello there. How are you?")
    assert cap_multi.mode == TtsStreamingMode.SENTENCE_CHUNK


# ---------------------------------------------------------------------------
# accepts_speaker_profile — opt-in flag the kernel checks
# ---------------------------------------------------------------------------


def test_sherpa_backend_class_advertises_speaker_profile_support():
    """The kernel's ``_backend_synthesize_kwargs`` checks
    ``accepts_speaker_profile`` to decide whether to thread the
    resolved profile to the backend. This pins the contract."""
    from octomil.runtime.engines.sherpa.engine import _SherpaTtsBackend

    assert getattr(_SherpaTtsBackend, "accepts_speaker_profile", False) is True
