"""sherpa-onnx engine plugin -- on-device text-to-speech via sherpa-onnx.

sherpa-onnx (k2-fsa) ships VITS/Piper/Kokoro TTS models packaged as ONNX.
This plugin wraps the sherpa-onnx Python bindings so TTS models register
with the octomil engine registry under the canonical ``sherpa-onnx``
executor id.

Unlike LLM engines, TTS does NOT use ``generate()`` / ``generate_stream()``.
Instead, the backend exposes a ``synthesize()`` method and the serve layer
adds an OpenAI-compatible ``/v1/audio/speech`` endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.core.base import BenchmarkResult, EnginePlugin

if TYPE_CHECKING:
    from octomil.audio.streaming import TtsStreamingCapability

logger = logging.getLogger(__name__)

# Supported TTS models -- name -> (family, default voice).
# family selects the sherpa-onnx config path:
#   "kokoro" -> OfflineTtsKokoroModelConfig (model + voices.bin + tokens + data_dir)
#   "vits"   -> OfflineTtsVitsModelConfig   (Piper-style: model.onnx + tokens + data_dir)
#   "pocket" -> OfflineTtsPocketModelConfig (lm_flow + lm_main + encoder + decoder
#               + text_conditioner + vocab.json + token_scores.json). Pocket is a
#               *few-shot voice-cloning* engine: instead of an integer sid the
#               generation call wants ``prompt_samples`` (reference audio) and
#               ``prompt_text``. Resolution flows through the planner-supplied
#               speaker profile (``ResolvedTtsSpeaker``), NOT a sid catalog.
# Voice catalogs are model-specific; the second tuple element is the default
# voice the backend uses when the request does not specify one. Pocket has no
# native voice catalog — its "voices" are reference profiles published by the
# planner — so the default voice for Pocket models is the empty string.
_SHERPA_TTS_MODELS: dict[str, tuple[str, str]] = {
    "kokoro-82m": ("kokoro", "af_bella"),
    # Legacy v0.19 bundle, retained as an explicit-pin id alongside
    # ``kokoro-82m``. ``af_bella`` is at sid=1 in v0.19's voices.bin
    # and is the default for both bundles.
    "kokoro-en-v0_19": ("kokoro", "af_bella"),
    "piper-en-amy": ("vits", "amy"),
    "piper-en-ryan": ("vits", "ryan"),
    # PocketTTS — int8-quantized few-shot voice-cloning engine. The
    # planner is responsible for selecting this id; clients should
    # call ``@app/<slug>/tts`` instead of pinning the runtime model.
    "pocket-tts-int8": ("pocket", ""),
}


def _model_family(model_name: str) -> str:
    """Return the sherpa-onnx config family ('kokoro' or 'vits') for a model."""
    entry = _SHERPA_TTS_MODELS.get(model_name.lower())
    return entry[0] if entry else ""


def _default_voice(model_name: str) -> str:
    entry = _SHERPA_TTS_MODELS.get(model_name.lower())
    return entry[1] if entry else ""


# Per-artifact Kokoro voice catalogs. Position in the tuple ==
# sherpa-onnx speaker id in the corresponding voices.bin.
#
# IMPORTANT: voice ordering is bundle-specific. A 28-name "modern"
# Kokoro catalog (af_heart, am_echo, …) is NOT interchangeable with
# the 11-name kokoro-en-v0_19 catalog the SDK currently ships —
# sherpa-onnx clamps out-of-range sids to 0, so a mismatched table
# silently aliases every "missing" voice to the default speaker.
#
# These tables are *legacy fallbacks*. The authoritative source is a
# ``voices.txt`` sidecar under the prepared artifact directory,
# materialized from the static recipe's ``voice_manifest`` field.
# The fallback only fires when a sidecar is absent — e.g. an
# artifact someone hand-staged before voices.txt materialization
# shipped — and is keyed by model id rather than a global "kokoro =
# these N names" assumption.

# kokoro-en-v0_19 — the legacy English-only bundle, still resolvable
# under the explicit ``kokoro-en-v0_19`` model id.
_KOKORO_EN_V0_19_VOICES: tuple[str, ...] = (
    "af",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
)

# kokoro-multi-lang-v1_0 — the bundle ``kokoro-82m`` resolves to as
# of the v1.0 cutover. 53 speakers across 8 languages; ordering is
# pinned to upstream's ``scripts/kokoro/v1.0/generate_voices_bin.py``
# so the fallback can never silently drift from voices.bin.
_KOKORO_MULTI_LANG_V1_0_VOICES: tuple[str, ...] = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
    "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel",
    "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex",
    "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
)  # fmt: skip

# Per-model legacy fallback catalog. Used ONLY when no voices.txt
# sidecar is present. Keep tightly scoped: an unknown model id
# falls through to "fail loudly" so callers can't accidentally
# inherit some other artifact's catalog.
_LEGACY_KOKORO_FALLBACK_CATALOGS: dict[str, tuple[str, ...]] = {
    "kokoro-82m": _KOKORO_MULTI_LANG_V1_0_VOICES,
    "kokoro-en-v0_19": _KOKORO_EN_V0_19_VOICES,
}

# Back-compat alias. Old import path
# ``octomil.runtime.engines.sherpa._KOKORO_VOICES`` resolves to the
# active default artifact's catalog so external callers keep working,
# but the canonical accessor is ``catalog_for_model(model_name)``.
_KOKORO_VOICES: tuple[str, ...] = _KOKORO_MULTI_LANG_V1_0_VOICES


def catalog_for_model(model_name: str) -> tuple[str, ...]:
    """Return the legacy fallback voice catalog for ``model_name``.

    Empty tuple when the model has no declared catalog. Callers that
    need authoritative ordering should read ``voices.txt`` from the
    prepared artifact directory; this helper is the *fallback* used
    only when the sidecar is missing.

    Keying solely on ``model_name`` is unsafe when an old prepared
    dir from a previous artifact identity is still on disk (e.g. a
    pre-cutover ``kokoro-82m`` v0.19 dir whose voices.txt sidecar
    predates the manifest patch). The richer
    :func:`fallback_catalog_for_artifact` resolves this by reading
    the materialized ``VERSION`` sidecar / layout signals.
    """
    return _LEGACY_KOKORO_FALLBACK_CATALOGS.get(model_name.lower(), ())


def fallback_catalog_for_artifact(model_name: str, model_dir: str) -> tuple[str, ...]:
    """Pick a legacy fallback catalog using artifact-on-disk signals.

    Resolution order:

      1. ``VERSION`` sidecar materialized by the static recipe.
         ``kokoro-en-v0_19`` → 11-speaker catalog;
         ``kokoro-multi-lang-v1_0`` → 53-speaker catalog.
      2. Directory-shape inference. If neither layout signal is
         present (artifact looks neither v0.19 nor v1.0), or the
         artifact's identity contradicts the model id (e.g. a
         pre-cutover v0.19 artifact still parked under
         ``kokoro-82m``), return an empty tuple — the engine then
         refuses the explicit voice path rather than silently
         aliasing names like ``bm_george`` to the wrong sid.
      3. As a last resort for sidecar-less artifacts whose layout
         IS clearly recognizable, fall back to the model-id catalog
         only when it agrees with the layout signal.
    """
    version_path = os.path.join(model_dir, "VERSION")
    declared_version = ""
    if os.path.isfile(version_path):
        try:
            with open(version_path, encoding="utf-8") as f:
                declared_version = f.read().strip()
        except OSError:
            declared_version = ""

    if declared_version == "kokoro-en-v0_19":
        return _KOKORO_EN_V0_19_VOICES
    if declared_version == "kokoro-multi-lang-v1_0":
        return _KOKORO_MULTI_LANG_V1_0_VOICES

    # No VERSION sidecar — infer from layout. v1.0 uniquely ships
    # dict/jieba.dict.utf8; v0.19 uniquely ships espeak-ng-data
    # WITHOUT the lexicon files (v1.0 ships both).
    has_dict = os.path.isfile(os.path.join(model_dir, "dict", "jieba.dict.utf8"))
    has_lexicon_us = os.path.isfile(os.path.join(model_dir, "lexicon-us-en.txt"))
    has_espeak = os.path.isfile(os.path.join(model_dir, "espeak-ng-data", "phontab"))

    if has_dict and has_lexicon_us:
        layout_catalog: tuple[str, ...] = _KOKORO_MULTI_LANG_V1_0_VOICES
    elif has_espeak and not has_dict and not has_lexicon_us:
        layout_catalog = _KOKORO_EN_V0_19_VOICES
    else:
        # Layout is ambiguous (or unrecognized). Refuse to guess —
        # an old pre-manifest dir under ``kokoro-82m`` could
        # otherwise inherit the v1.0 catalog and re-introduce the
        # silent aliasing bug.
        return ()

    # Cross-check against the model-id catalog. If they disagree,
    # treat the dir as ambiguous (operator-staged bytes that don't
    # match the canonical recipe).
    name_catalog = catalog_for_model(model_name)
    if name_catalog and name_catalog is not layout_catalog:
        return ()
    return layout_catalog


def _read_voice_manifest(model_dir: str) -> tuple[str, ...]:
    """Read ``voices.txt`` from ``model_dir`` and return the ordered
    list of speaker names. Returns an empty tuple when the sidecar
    is missing. Trims trailing whitespace and skips blank lines so a
    crash-truncated final newline doesn't shift speaker ids.
    """
    sidecar = os.path.join(model_dir, "voices.txt")
    if not os.path.exists(sidecar):
        return ()
    with open(sidecar, encoding="utf-8") as f:
        return tuple(line.strip() for line in f if line.strip())


def _read_artifact_version(model_dir: str) -> str:
    """Read the ``VERSION`` sidecar (or empty string)."""
    version_path = os.path.join(model_dir, "VERSION")
    if not os.path.isfile(version_path):
        return ""
    try:
        with open(version_path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


@dataclass(frozen=True)
class ResolvedVoiceCatalog:
    """Engine-side projection of a TTS voice catalog.

    Single-source-of-truth for both the synthesis path
    (``_voice_to_sid``) and the listing path
    (``list_speech_voices``). Position in ``voices`` matches
    sherpa-onnx speaker id; ``source`` is provenance — either
    ``"voices_txt"`` (read from the prepared artifact's sidecar),
    ``"static_recipe"`` (no sidecar yet, manifest came from the
    recipe), or ``""`` (no catalog known).
    """

    voices: tuple[str, ...]
    source: str = ""
    artifact_version: str = ""


def resolve_voice_catalog(
    model_name: str,
    *,
    prepared_model_dir: Optional[str] = None,
    static_recipe_manifest: tuple[str, ...] = (),
    static_recipe_artifact_version: str = "",
) -> ResolvedVoiceCatalog:
    """Single resolver shared by synthesis, validation, and listing.

    Resolution order:

      1. ``voices.txt`` sidecar in ``prepared_model_dir`` —
         authoritative for the artifact actually on disk.
      2. ``VERSION`` + layout fallback under ``prepared_model_dir``
         (handles sidecar-less prepared dirs from before the
         manifest patch).
      3. ``static_recipe_manifest`` — used when no prepared dir
         exists yet (the listing path can preview the catalog
         without forcing a download).
      4. Model-id legacy fallback (``catalog_for_model``), only
         when no other signal is available.

    Returns an empty catalog (``voices=()``) when none of the above
    yields a result; callers translate that into a strict refusal
    for the explicit-voice path or a "no catalog known" listing.
    """
    if prepared_model_dir:
        sidecar = _read_voice_manifest(prepared_model_dir)
        if sidecar:
            return ResolvedVoiceCatalog(
                voices=sidecar,
                source="voices_txt",
                artifact_version=_read_artifact_version(prepared_model_dir),
            )
        layout = fallback_catalog_for_artifact(model_name, prepared_model_dir)
        if layout:
            return ResolvedVoiceCatalog(
                voices=layout,
                source="voices_txt",  # derived from artifact-on-disk signals
                artifact_version=_read_artifact_version(prepared_model_dir),
            )
        # Prepared dir exists but is ambiguous (layout + model id
        # disagree, or layout is unrecognized). Refuse to fall
        # through to the model-id catalog or the recipe manifest:
        # an old pre-manifest dir under the same model id would
        # otherwise inherit the WRONG catalog and re-introduce the
        # silent sid-aliasing bug.
        return ResolvedVoiceCatalog(voices=(), source="", artifact_version="")

    if static_recipe_manifest:
        return ResolvedVoiceCatalog(
            voices=static_recipe_manifest,
            source="static_recipe",
            artifact_version=static_recipe_artifact_version,
        )

    name_fallback = catalog_for_model(model_name)
    if name_fallback:
        return ResolvedVoiceCatalog(
            voices=name_fallback,
            source="static_recipe",
            artifact_version="",
        )

    return ResolvedVoiceCatalog(voices=(), source="", artifact_version="")


def _build_kokoro_model_config(sherpa_onnx: Any, model_dir: str) -> Any:
    """Construct ``OfflineTtsKokoroModelConfig`` for a Kokoro artifact.

    Wires three shape-dependent knobs:

      - ``data_dir``: espeak-ng phoneme tables. Required keyword on
        sherpa-onnx 1.13.0's config (omitting it raises TypeError),
        and required *value* whenever the bundle ships
        ``espeak-ng-data/`` — both v0.19 AND v1.0 do, and upstream's
        own ``--kokoro-data-dir`` invocation passes it for v1.0
        alongside the lexicon files. Leaving it empty when the
        directory exists risks broken / OOV phonemization for any
        language whose phoneme set isn't covered by the lexicon.
      - ``lexicon`` + ``dict_dir``: v1.0+ adds Chinese segmentation
        (jieba dict) and per-language lexicons that replace espeak
        for English/Chinese specifically. These are additive on top
        of espeak rather than a replacement for the binary itself.

    Detection is by directory contents, not model id, so a future
    v1.1/v2 bundle with the same shape works without code changes.
    """
    espeak_dir = os.path.join(model_dir, "espeak-ng-data")
    dict_dir = os.path.join(model_dir, "dict")
    lexicon_files = [
        os.path.join(model_dir, name)
        for name in ("lexicon-us-en.txt", "lexicon-gb-en.txt", "lexicon-zh.txt")
        if os.path.isfile(os.path.join(model_dir, name))
    ]
    has_v1_layout = os.path.isdir(dict_dir) and bool(lexicon_files)
    has_espeak = os.path.isdir(espeak_dir)

    if not has_v1_layout and not has_espeak:
        raise OctomilError(
            code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
            message=(
                f"sherpa-onnx Kokoro artifact at {model_dir!r} has neither an "
                f"espeak-ng-data/ directory (v0.19 layout) nor dict/ + "
                f"lexicon-*.txt files (v1.0+ layout). The bundle is incomplete "
                f"or from an unsupported Kokoro release."
            ),
        )

    # ``data_dir`` MUST be present on the config kwargs (sherpa-onnx
    # 1.13.0 raises TypeError otherwise). Set the espeak path
    # whenever it exists, regardless of whether the v1.0 lexicon
    # path is also active — they coexist in upstream's invocation.
    base_kwargs: dict[str, Any] = {
        "model": os.path.join(model_dir, "model.onnx"),
        "voices": os.path.join(model_dir, "voices.bin"),
        "tokens": os.path.join(model_dir, "tokens.txt"),
        "data_dir": espeak_dir if has_espeak else "",
    }

    if has_v1_layout:
        base_kwargs["lexicon"] = ",".join(lexicon_files)
        base_kwargs["dict_dir"] = dict_dir

    return sherpa_onnx.OfflineTtsKokoroModelConfig(**base_kwargs)


# Required files in a prepared PocketTTS artifact directory. Names
# match upstream's ``sherpa-onnx-pocket-tts-*`` bundles and the
# ``OfflineTtsPocketModelConfig`` keyword arguments. We validate
# layout up front so a bad artifact fails *here* with a precise
# message instead of inside sherpa-onnx with an opaque ONNX error.
_POCKET_REQUIRED_FILES: tuple[tuple[str, str], ...] = (
    ("text_conditioner", "text_conditioner.onnx"),
    ("encoder", "encoder.onnx"),
    ("lm_flow", "lm_flow.int8.onnx"),
    ("decoder", "decoder.int8.onnx"),
    ("lm_main", "lm_main.int8.onnx"),
    ("vocab_json", "vocab.json"),
    ("token_scores_json", "token_scores.json"),
)


def _build_pocket_model_config(sherpa_onnx: Any, model_dir: str) -> Any:
    """Construct ``OfflineTtsPocketModelConfig`` for a PocketTTS artifact.

    Maps the prepared-artifact layout to the seven keyword arguments
    the sherpa-onnx 1.13 binding exposes::

        OfflineTtsPocketModelConfig(
            lm_flow=...,           # lm_flow.int8.onnx
            lm_main=...,           # lm_main.int8.onnx
            encoder=...,           # encoder.onnx
            decoder=...,           # decoder.int8.onnx
            text_conditioner=...,  # text_conditioner.onnx
            vocab_json=...,        # vocab.json
            token_scores_json=..., # token_scores.json
            voice_embedding_cache_capacity=50,  # default
        )

    Validates each required file exists and raises an actionable
    ``RUNTIME_UNAVAILABLE`` if not — a partial bundle would otherwise
    crash inside sherpa-onnx with an opaque ONNX session error.
    """
    missing: list[str] = []
    paths: dict[str, str] = {}
    for kw, filename in _POCKET_REQUIRED_FILES:
        path = os.path.join(model_dir, filename)
        if not os.path.isfile(path):
            missing.append(filename)
            continue
        paths[kw] = path

    if missing:
        raise OctomilError(
            code=OctomilErrorCode.RUNTIME_UNAVAILABLE,
            message=(
                f"sherpa-onnx PocketTTS artifact at {model_dir!r} is missing "
                f"required files: {', '.join(missing)}. Re-run "
                f"client.prepare(model='pocket-tts-int8', capability='tts') "
                f"or check the artifact tarball."
            ),
        )

    return sherpa_onnx.OfflineTtsPocketModelConfig(**paths)


def _load_reference_samples(reference_audio: str) -> tuple[Any, int]:
    """Load reference WAV from a local path; return (samples_float32, sr).

    PocketTTS / ZipVoice need ``prompt_samples`` (float32 in [-1, 1])
    plus their native ``sample_rate`` for the voice-cloning generate
    call. The stdlib ``wave`` module covers the common-case path
    (16-bit PCM mono) without a numpy dependency at the engine
    boundary; we still import numpy lazily so the conversion is
    vectorised when available.

    The planner is responsible for materializing the reference audio
    locally (URL-based references are pre-fetched). Callers receive
    a precise error when the path doesn't exist or the WAV is
    malformed.
    """
    import struct
    import wave

    if not reference_audio:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                "speaker_profile_missing_reference: PocketTTS requires "
                "reference_audio in the planner profile, but the resolved "
                "speaker has none. Check the app's tts_speakers map."
            ),
        )
    if not os.path.isfile(reference_audio):
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                f"reference_audio_not_found: {reference_audio!r} does not "
                f"exist on disk. Planner must materialize reference audio "
                f"locally before synthesis (URL-based references must be "
                f"prepared via the standard artifact pipeline)."
            ),
        )

    try:
        with wave.open(reference_audio, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except wave.Error as exc:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                f"reference_audio_format_unsupported: {reference_audio!r} is "
                f"not a valid WAV file ({exc}). PocketTTS reference audio "
                f"must be PCM WAV (16-bit mono recommended)."
            ),
        ) from exc

    if sample_width != 2:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                f"reference_audio_format_unsupported: {reference_audio!r} has "
                f"sample_width={sample_width}. PocketTTS reference audio must "
                f"be 16-bit PCM WAV."
            ),
        )

    n_samples = len(raw) // sample_width
    pcm_int16 = struct.unpack(f"<{n_samples}h", raw)
    if n_channels > 1:
        # Downmix interleaved stereo to mono — sherpa-onnx wants a
        # mono float32 buffer; better to do this here than crash in
        # the ONNX session.
        mixed: list[int] = []
        for i in range(0, n_samples, n_channels):
            chunk = pcm_int16[i : i + n_channels]
            mixed.append(sum(chunk) // n_channels)
        pcm_int16 = tuple(mixed)

    try:
        import numpy as np  # type: ignore[import-untyped]

        samples = np.asarray(pcm_int16, dtype=np.float32) / 32768.0
        return samples, framerate
    except ImportError:
        # Pure-Python fallback for environments without numpy
        # (Ren'Py / stripped PyInstaller bundles where Pocket isn't
        # used anyway, but the load path stays portable).
        return [s / 32768.0 for s in pcm_int16], framerate


def _has_sherpa_onnx() -> bool:
    """Check if the sherpa_onnx package is importable."""
    try:
        import sherpa_onnx  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def _get_sherpa_version() -> str:
    """Return sherpa_onnx version string, or empty if unavailable."""
    try:
        import sherpa_onnx  # type: ignore[import-untyped]

        return getattr(sherpa_onnx, "__version__", "unknown")
    except ImportError:
        return ""


def is_sherpa_tts_model(model_name: str) -> bool:
    """Check if a model name refers to a sherpa-onnx TTS model.

    Means "known model id," not "installed and runnable." For runnable
    detection, the kernel asks PrepareManager whether a prepared
    artifact dir exists for ``(model, capability='tts')`` — there is no
    legacy "is staged" path.
    """
    return model_name.lower() in _SHERPA_TTS_MODELS


def is_sherpa_tts_runtime_available(model_name: str) -> bool:
    """Return True when the *engine* is loadable for ``model_name``, even if
    the artifact has not been downloaded yet.

    Pairs with PrepareManager: a planner ``sdk_runtime`` candidate with
    ``prepare_required=True`` (or a static-recipe fallback with the same
    shape) is a valid local route as long as sherpa-onnx is importable
    and the model id is recognized — PrepareManager materializes the
    bytes before backend load, and the backend reads from the prepared
    artifact dir threaded in via ``model_dir=``.
    """
    return _has_sherpa_onnx() and is_sherpa_tts_model(model_name)


class SherpaTtsEngine(EnginePlugin):
    """Text-to-speech engine using sherpa-onnx."""

    @property
    def name(self) -> str:
        return "sherpa-onnx"

    @property
    def display_name(self) -> str:
        return "sherpa-onnx (Text-to-Speech)"

    @property
    def priority(self) -> int:
        return 36  # Sits next to whisper.cpp (35).

    def detect(self) -> bool:
        return _has_sherpa_onnx()

    def detect_info(self) -> str:
        version = _get_sherpa_version()
        if not version:
            return ""
        models = ", ".join(sorted(_SHERPA_TTS_MODELS.keys()))
        return f"sherpa_onnx {version}; tts models: {models}"

    def supports_model(self, model_name: str) -> bool:
        return is_sherpa_tts_model(model_name)

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        """Benchmark by synthesizing a short reference utterance.

        For TTS, ``tokens_per_second`` is repurposed as
        ``audio_seconds_per_second`` (real-time factor).
        """
        if not _has_sherpa_onnx():
            return BenchmarkResult(engine_name=self.name, error="sherpa_onnx not available")

        if not is_sherpa_tts_model(model_name):
            return BenchmarkResult(
                engine_name=self.name,
                error=f"Unsupported model: {model_name}",
            )

        try:
            backend = _SherpaTtsBackend(model_name)
            backend.load_model(model_name)

            reference = "Octomil benchmark synthesis check."

            start = time.monotonic()
            result = backend.synthesize(reference)
            elapsed = time.monotonic() - start

            audio_duration_s = result["duration_ms"] / 1000.0
            realtime_factor = audio_duration_s / elapsed if elapsed > 0 else 0.0

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=realtime_factor,
                metadata={
                    "method": "synthesize",
                    "audio_seconds_per_second": realtime_factor,
                    "audio_duration_s": round(audio_duration_s, 3),
                    "elapsed_s": round(elapsed, 3),
                    "model": model_name,
                    "sample_chars": len(reference),
                },
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        return _SherpaTtsBackend(model_name, **kwargs)


class _SherpaTtsBackend:
    """Text-to-speech backend using sherpa-onnx.

    Unlike LLM backends, this does NOT implement ``generate()`` or
    ``generate_stream()``. Instead it provides ``synthesize(text, voice, speed)``
    returning audio bytes plus metadata. The serve layer adds a dedicated
    ``/v1/audio/speech`` endpoint that mirrors OpenAI ``audio.speech.create``.
    """

    name = "sherpa-onnx"

    # Opts the kernel into ``speaker_profile=`` kwarg dispatch. The
    # bridge in ``octomil.execution.kernel._backend_synthesize_kwargs``
    # only sets this kwarg when the flag is True, so legacy backends
    # without the flag stay on the original ``(text, voice, speed)``
    # signature unchanged.
    accepts_speaker_profile = True

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._kwargs = kwargs
        # Optional caller-supplied model directory. When set, this short-
        # circuits the env/home lookup and is used verbatim, e.g. when the
        # PrepareManager has materialized the artifact under
        # ``<cache>/artifacts/<artifact_id>/`` and tells the backend exactly
        # where to load from.
        self._injected_model_dir: str | None = kwargs.get("model_dir")
        self._tts: Any = None
        self._sample_rate: int = 24000
        self._family: str = _model_family(model_name)
        self._default_voice: str = _default_voice(model_name)

    def load_model(self, model_name: str) -> None:
        """Load a sherpa-onnx TTS model from the configured model directory.

        Branches on model family because Kokoro and VITS/Piper expect
        different OfflineTtsModelConfig shapes:
          - kokoro: OfflineTtsKokoroModelConfig(model, voices, tokens,
            data_dir, [lexicon, dict_dir])
          - vits:   OfflineTtsVitsModelConfig(model, tokens, data_dir)

        See :func:`_build_kokoro_model_config` for the Kokoro
        sub-branching: ``data_dir`` is wired to ``espeak-ng-data/``
        whenever that directory exists (both v0.19 and v1.0 ship it
        and upstream's invocation passes it for both), and v1.0+
        additionally wires ``lexicon`` + ``dict_dir`` for Chinese
        segmentation. Detection is by directory contents, NOT model
        id, so a future v1.1/v2 with the same shape works without
        code changes.
        """
        self._model_name = model_name
        if not is_sherpa_tts_model(model_name):
            raise ValueError(
                f"Unknown sherpa-onnx TTS model '{model_name}'. Available: {', '.join(sorted(_SHERPA_TTS_MODELS))}"
            )

        import sherpa_onnx  # type: ignore[import-untyped]

        model_dir = self._resolve_model_dir(model_name)
        family = _model_family(model_name)
        num_threads = int(self._kwargs.get("num_threads", 2))
        provider = self._kwargs.get("provider", "cpu")

        if family == "kokoro":
            inner_model_config = sherpa_onnx.OfflineTtsModelConfig(
                kokoro=_build_kokoro_model_config(sherpa_onnx, model_dir),
                num_threads=num_threads,
                provider=provider,
            )
        elif family == "vits":
            inner_model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=os.path.join(model_dir, "model.onnx"),
                    tokens=os.path.join(model_dir, "tokens.txt"),
                    data_dir=os.path.join(model_dir, "espeak-ng-data"),
                ),
                num_threads=num_threads,
                provider=provider,
            )
        elif family == "pocket":
            inner_model_config = sherpa_onnx.OfflineTtsModelConfig(
                pocket=_build_pocket_model_config(sherpa_onnx, model_dir),
                num_threads=num_threads,
                provider=provider,
            )
        else:
            raise ValueError(f"Unsupported sherpa-onnx TTS family '{family}' for model '{model_name}'.")

        # max_num_sentences=1 forces sherpa-onnx to invoke the
        # per-chunk callback once per sentence instead of batching the
        # entire utterance into a single callback. This is what makes
        # the streaming path actually streaming for multi-sentence
        # input — without it, Kokoro emits exactly one chunk regardless
        # of input length and the SDK's "realtime" advertisement is a
        # lie. Single-sentence input still produces one chunk; the SDK
        # downgrades the advertised capability to ``final_chunk`` for
        # those cases.
        config = sherpa_onnx.OfflineTtsConfig(model=inner_model_config, max_num_sentences=1)
        logger.info("Loading sherpa-onnx %s TTS: %s from %s", family, model_name, model_dir)
        self._tts = sherpa_onnx.OfflineTts(config)
        self._sample_rate = self._tts.sample_rate
        self._family = family
        self._default_voice = _default_voice(model_name)
        logger.info(
            "sherpa-onnx TTS loaded: %s (family=%s, sample_rate=%d)",
            model_name,
            family,
            self._sample_rate,
        )

    def _resolve_model_dir(self, model_name: str) -> str:
        """Return the on-disk directory for a sherpa-onnx model.

        The only supported source is the ``model_dir`` kwarg passed
        to ``create_backend`` — i.e. the artifact dir
        :class:`PrepareManager` materialized for the request. PR D
        cut over the legacy ``OCTOMIL_SHERPA_MODELS_DIR`` /
        ``~/.octomil/models/sherpa/<model>/`` resolution; callers
        who hand-staged bytes in the legacy layout must either run
        ``client.prepare(model, capability='tts')`` or invoke the
        kernel through a planner candidate that triggers prepare.
        """
        if self._injected_model_dir:
            return self._injected_model_dir
        raise RuntimeError(
            f"sherpa-onnx TTS backend for {model_name!r} was constructed without "
            "a prepared model_dir. Run client.prepare(model, capability='tts') "
            "(or call the kernel through a planner candidate that triggers "
            "prepare) before loading the backend; the legacy "
            "OCTOMIL_SHERPA_MODELS_DIR / ~/.octomil/models/sherpa fallback was "
            "removed in 4.11.0."
        )

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        *,
        speaker_profile: Any = None,
    ) -> dict[str, Any]:
        """Synthesize speech from text and return audio bytes + metadata.

        Returns::

            {
                "audio_bytes": bytes,         # WAV (PCM 16-bit mono)
                "content_type": "audio/wav",
                "format": "wav",
                "sample_rate": 24000,
                "duration_ms": 1234,
                "voice": "af_bella",
                "model": "kokoro-82m",
            }

        ``voice`` defaults to the model's default if not provided.
        ``speed`` is a multiplier; 1.0 is default, 0.5 half-speed, 2.0 double.

        ``speaker_profile`` (optional) is a
        :class:`octomil.execution.tts_speaker_resolver.ResolvedTtsSpeaker`.
        Native-voice engines (Kokoro / Piper) ignore everything except
        ``native_voice``; PocketTTS uses ``reference_audio`` +
        ``reference_sample_rate`` to clone, ignoring ``voice`` entirely.
        """
        if not text.strip():
            raise ValueError("text must not be empty")
        if speed <= 0:
            raise ValueError("speed must be positive")

        if self._tts is None:
            self.load_model(self._model_name)
        assert self._tts is not None

        if self._family == "pocket":
            samples, sample_rate, voice_name = self._pocket_generate(
                text=text,
                speed=speed,
                speaker_profile=speaker_profile,
            )
        else:
            # Native-voice engines (Kokoro / Piper). Distinguish
            # "caller passed an explicit voice" from "fall back to the
            # model's documented default": the latter is a safe
            # ``sid=0`` for catalog-less models (single-speaker Piper
            # bundles), the former must enforce the catalog.
            caller_voice = (voice or "").strip()
            explicit = bool(caller_voice)
            voice_name = caller_voice or (self._default_voice or "").strip()
            sid = self._voice_to_sid(voice_name, explicit=explicit)
            audio = self._tts.generate(text, sid=sid, speed=speed)
            samples = list(audio.samples)
            sample_rate = audio.sample_rate or self._sample_rate

        wav_bytes = _samples_to_wav(samples, sample_rate)
        duration_ms = int(round(1000 * len(samples) / sample_rate)) if sample_rate else 0

        return {
            "audio_bytes": wav_bytes,
            "content_type": "audio/wav",
            "format": "wav",
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
            "voice": voice_name,
            "model": self._model_name,
        }

    def _pocket_generate(
        self,
        *,
        text: str,
        speed: float,
        speaker_profile: Any,
    ) -> tuple[list[float], int, str]:
        """PocketTTS voice-cloning generate path.

        Pocket's ``OfflineTts.generate`` voice-cloning overload is::

            generate(text, prompt_text, prompt_samples, sample_rate,
                     speed=1.0, num_steps=4, callback=None)

        We pull ``prompt_samples`` and ``sample_rate`` off the
        :class:`ResolvedTtsSpeaker`'s ``reference_audio`` /
        ``reference_sample_rate`` (loading the WAV via stdlib), and
        ``prompt_text`` from the planner profile metadata when
        present (transcription of the reference clip improves the
        clone). Returns the same ``(samples, sample_rate,
        voice_name)`` triple the native-voice path produces so the
        caller's WAV-wrapping is uniform.
        """
        if speaker_profile is None or not getattr(speaker_profile, "reference_audio", None):
            raise OctomilError(
                code=OctomilErrorCode.INVALID_INPUT,
                message=(
                    "speaker_profile_missing_reference: PocketTTS requires "
                    "a planner-supplied speaker profile carrying "
                    "reference_audio. Pass speaker= and ensure the app's "
                    "tts_speakers map publishes one."
                ),
            )

        prompt_samples, file_sr = _load_reference_samples(speaker_profile.reference_audio)
        # Profile may declare a sample_rate; if it disagrees with the
        # WAV header, trust the WAV header and surface a warning so
        # the planner can fix the mismatch.
        declared_sr = speaker_profile.reference_sample_rate or file_sr
        if declared_sr != file_sr:
            logger.warning(
                "PocketTTS reference sample-rate mismatch: profile=%d, wav=%d. Using wav header.",
                declared_sr,
                file_sr,
            )
        sample_rate_in = file_sr

        # Optional reference transcription. Pocket accepts an empty
        # string when one isn't available, so we don't enforce it.
        metadata = getattr(speaker_profile, "metadata", None) or {}
        prompt_text = str(metadata.get("reference_text", ""))

        audio = self._tts.generate(
            text,
            prompt_text,
            prompt_samples,
            sample_rate_in,
            speed=speed,
            num_steps=int(metadata.get("num_steps", 4)),
        )
        samples = list(audio.samples)
        out_sr = audio.sample_rate or self._sample_rate
        speaker_label = getattr(speaker_profile, "speaker", None) or ""
        return samples, out_sr, speaker_label

    def _voice_to_sid(self, voice: str, *, explicit: bool = True) -> int:
        """Map a voice name to a sherpa-onnx speaker id.

        Resolution rules:

          - Empty voice string → ``sid=0`` intentionally. The
            backend's first speaker is the contracted default.
          - ``voices.txt`` sidecar in the prepared artifact dir is
            the authoritative ordered catalog for THIS artifact.
            Position in the file == speaker id.
          - When the sidecar is missing, fall back to a per-model
            legacy catalog (``catalog_for_model``).
          - On a miss:
              * ``explicit=True`` (caller supplied a voice string)
                raises ``voice_not_supported_for_model`` so they get
                a clear error instead of sherpa-onnx silently
                aliasing the request to ``sid=0``.
              * ``explicit=False`` (we resolved the default voice
                because the caller passed nothing) returns ``sid=0``.
                Single-speaker bundles like Piper have no catalog
                and shouldn't reject the default voice path.
        """
        if not voice:
            return 0

        # Single shared resolver: same path the listing API
        # (``ExecutionKernel.list_speech_voices``) and the kernel
        # preflight call. Closes the loop — a voice that listing
        # advertises will resolve here and vice versa.
        model_dir = self._resolve_model_dir(self._model_name)
        resolved = resolve_voice_catalog(self._model_name, prepared_model_dir=model_dir)
        manifest = resolved.voices

        if not manifest:
            # No catalog available. For an explicit caller-supplied
            # voice, refuse loudly; for a defaulted lookup, fall
            # back to the bundle's first speaker.
            if explicit:
                raise OctomilError(
                    code=OctomilErrorCode.INVALID_INPUT,
                    message=(
                        f"voice_not_supported_for_model: model {self._model_name!r} "
                        f"has no declared voice catalog (no voices.txt sidecar, no "
                        f"built-in fallback). Pass voice=None to use the default "
                        f"speaker, or run client.prepare(model, capability='tts') "
                        f"to materialize the artifact's voice manifest."
                    ),
                )
            return 0

        target = voice.strip().lower()
        for idx, name in enumerate(manifest):
            if name.lower() == target:
                return idx

        if not explicit:
            # The model's documented default voice isn't in the
            # artifact's catalog. Don't crash an otherwise-valid
            # request; the bundle's first speaker is the safe
            # default and the explicit-voice path stays strict.
            return 0

        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                f"voice_not_supported_for_model: voice {voice!r} is not in "
                f"the speaker catalog for model {self._model_name!r}. "
                f"Supported voices: {', '.join(manifest)}."
            ),
        )

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def streaming_capability(self, text: str) -> "TtsStreamingCapability":
        """Honest streaming capability for *this* input on *this* engine.

        sherpa-onnx Kokoro / Piper invokes the per-chunk callback at
        sentence boundaries — there is no sub-sentence (frame /
        phoneme) streaming today. So:

          - Multi-sentence text -> ``sentence_chunk``.
          - Single-sentence text -> ``final_chunk`` (only one chunk
            will arrive regardless of what the engine claims).

        ``verified=False`` because the actual chunk count is only
        known after synthesis runs. The kernel's stream wrapper
        flips ``verified=True`` on completion if the run matched.
        """
        from octomil.audio.streaming import TtsStreamingCapability

        if _count_sentences(text) > 1:
            return TtsStreamingCapability.sentence(verified=False)
        return TtsStreamingCapability.final_only(verified=False)

    def validate_voice(self, voice: str | None) -> tuple[int, str]:
        """Resolve a caller-supplied voice synchronously.

        Returns ``(sid, resolved_label)``. Raises ``OctomilError`` with
        ``voice_not_supported_for_model`` when an explicit voice is
        unsupported. Public surface so the kernel and HTTP route can
        validate *before* a stream's first event / response status is
        committed — without this check, an unsupported voice would only
        surface mid-stream after the consumer started iterating.

        Default-label resolution: when ``voice`` is empty/None and the
        model has a manifest, the label returned is ``manifest[0]``
        (the actual sid=0 speaker). Catalog-less models (Piper) fall
        back to ``self._default_voice``.

        Pocket has no native voice catalog — its "voices" are
        reference-audio profiles owned by the planner. The kernel
        calls this with ``voice=None`` for Pocket requests (the
        resolver leaves ``native_voice`` ``None`` for reference-audio
        profiles), and we return ``(0, "")`` so synthesis_stream's
        synchronous validation step never trips on a missing native
        catalog. Speaker validation for Pocket happens at the
        ``speaker_profile`` check inside ``synthesize`` /
        ``synthesize_stream``.
        """
        if self._family == "pocket":
            # No native voice catalog. The kernel hands us ``None``
            # for Pocket reference-audio profiles; an explicit voice
            # string isn't meaningful here, so accept it as-is.
            return 0, (voice or "")

        explicit = (voice or "").strip()
        if explicit:
            sid = self._voice_to_sid(explicit, explicit=True)
            return sid, explicit
        # Default-label resolution only needs the prepared model_dir
        # — no loaded sherpa OfflineTts required. Reading voices.txt
        # is a file-system op, not an ONNX op. Loading the model
        # here would force callers (and tests) to have sherpa-onnx
        # importable just to learn the default voice's label.
        manifest = resolve_voice_catalog(
            self._model_name, prepared_model_dir=self._resolve_model_dir(self._model_name)
        ).voices
        if manifest:
            return 0, manifest[0]
        return 0, self._default_voice or ""

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        *,
        speaker_profile: Any = None,
        chunk_max_queue: int = 16,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield PCM s16le chunk dicts as samples are produced.

        Each yielded chunk has shape::

            {"pcm_s16le": <bytes>, "num_samples": <int>}

        The synthesis runs on a worker thread; sherpa-onnx's ``generate``
        callback pushes chunks into a bounded sync->async bridge so the
        event loop stays responsive and an unresponsive consumer applies
        real backpressure to the producer (the worker thread blocks
        when the bridge is full).

        Stopping iteration (``async generator .aclose()``, exception in
        consumer, etc.) sets a cancellation flag that causes the next
        callback invocation to return ``0`` — sherpa-onnx interprets
        that as "stop synthesis" — and the worker thread exits cleanly.
        """
        if not text.strip():
            raise ValueError("text must not be empty")
        if speed <= 0:
            raise ValueError("speed must be positive")

        if self._tts is None:
            self.load_model(self._model_name)
        assert self._tts is not None

        # Family-specific generate-args resolution. Pocket needs
        # reference samples; Kokoro/Piper just need a sid. Both
        # resolution paths are synchronous so an unsupported speaker
        # / voice raises BEFORE the worker thread spins up.
        if self._family == "pocket":
            if speaker_profile is None or not getattr(speaker_profile, "reference_audio", None):
                raise OctomilError(
                    code=OctomilErrorCode.INVALID_INPUT,
                    message=(
                        "speaker_profile_missing_reference: PocketTTS streaming "
                        "requires a planner-supplied speaker profile with "
                        "reference_audio. Pass speaker= and ensure the app's "
                        "tts_speakers map publishes one."
                    ),
                )
            prompt_samples, prompt_sr = _load_reference_samples(speaker_profile.reference_audio)
            metadata = getattr(speaker_profile, "metadata", None) or {}
            prompt_text = str(metadata.get("reference_text", ""))
            num_steps = int(metadata.get("num_steps", 4))
            sid = None
        else:
            sid, _resolved_label = self.validate_voice(voice)
            prompt_samples = None
            prompt_sr = 0
            prompt_text = ""
            num_steps = 0

        bridge = _StreamBridge(maxsize=chunk_max_queue)
        loop = asyncio.get_running_loop()
        sample_rate = self._tts.sample_rate or self._sample_rate

        def _worker() -> None:
            """Run sherpa generate on a thread; push chunks to bridge."""
            try:

                def _callback(samples_f32: Any, _progress: float) -> int:
                    if bridge.is_cancelled():
                        return 0
                    pcm = _float32_to_pcm_s16le_bytes(samples_f32)
                    n = int(getattr(samples_f32, "size", len(samples_f32)))
                    accepted = bridge.put(pcm, n)
                    return 1 if accepted else 0

                if self._family == "pocket":
                    self._tts.generate(
                        text,
                        prompt_text,
                        prompt_samples,
                        prompt_sr,
                        speed=speed,
                        num_steps=num_steps,
                        callback=_callback,
                    )
                else:
                    self._tts.generate(text, sid=sid, speed=speed, callback=_callback)
                bridge.close(error=None)
            except BaseException as exc:  # noqa: BLE001 — re-raised on consumer side
                bridge.close(error=exc)

        worker = threading.Thread(
            target=_worker,
            name=f"sherpa-tts-stream-{self._model_name}",
            daemon=True,
        )
        worker.start()

        try:
            while True:
                item = await loop.run_in_executor(None, bridge.get)
                if item is _STREAM_SENTINEL_DONE:
                    bridge.raise_if_error()
                    return
                pcm_bytes, num_samples = item
                yield {
                    "pcm_s16le": pcm_bytes,
                    "num_samples": num_samples,
                    "sample_rate": sample_rate,
                }
        finally:
            bridge.cancel()
            worker.join(timeout=5.0)


_STREAM_SENTINEL_DONE = object()


class _StreamBridge:
    """Bounded thread-safe bridge between the sherpa worker thread and
    the asyncio consumer.

    Hand-rolled instead of :class:`queue.Queue` /
    ``loop.call_soon_threadsafe(queue.put_nowait, ...)`` because we
    need (1) real backpressure: ``put`` blocks the producer thread
    when the bridge is full so memory cannot grow unboundedly, and
    (2) cooperative cancellation: ``put`` returns ``False`` when
    cancelled so the sherpa callback can return 0 and stop synthesis.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = max(1, int(maxsize))
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        self._buf: deque[tuple[bytes, int]] = deque()
        self._closed = False
        self._cancelled = False
        self._error: BaseException | None = None

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def put(self, pcm: bytes, num_samples: int) -> bool:
        with self._not_full:
            while len(self._buf) >= self._maxsize and not self._cancelled and not self._closed:
                self._not_full.wait(timeout=0.5)
            if self._cancelled or self._closed:
                return False
            self._buf.append((pcm, num_samples))
            self._not_empty.notify()
            return True

    def get(self) -> Any:
        with self._not_empty:
            while not self._buf and not self._closed and not self._cancelled:
                self._not_empty.wait(timeout=0.5)
            if self._buf:
                item = self._buf.popleft()
                self._not_full.notify()
                return item
            return _STREAM_SENTINEL_DONE

    def close(self, error: BaseException | None) -> None:
        with self._lock:
            self._closed = True
            self._error = error
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            self._closed = True
            self._buf.clear()
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def raise_if_error(self) -> None:
        with self._lock:
            err = self._error
        if err is not None:
            raise err


_SENTENCE_TERMINATORS = re.compile(r"[.!?。！？]+\s+\S")


def _count_sentences(text: str) -> int:
    """Cheap sentence count for streaming-capability advertisement.

    Mirrors what sherpa-onnx's ``max_num_sentences=1`` will actually
    split on (period / exclamation / question + whitespace). Not a
    full ICU-grade sentence segmenter — we just need a yes/no on
    "is multi-sentence." A trailing terminator without following
    text is still one sentence; the regex requires a non-space
    character after the whitespace so ``"Hello."`` reports 1, not 2.
    """
    if not text or not text.strip():
        return 0
    matches = _SENTENCE_TERMINATORS.findall(text)
    return 1 + len(matches)


def _float32_to_pcm_s16le_bytes(samples: Any) -> bytes:
    """Convert a numpy.ndarray[float32] of samples in [-1, 1] to PCM int16 LE.

    Uses ``numpy`` when available (sherpa-onnx already imports it) for the
    vectorized fast path. Falls back to a struct-based loop for the test
    fakes that pass plain Python iterables.
    """
    try:
        import numpy as np  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover — sherpa-onnx requires numpy
        np = None  # type: ignore[assignment]

    if np is not None and hasattr(samples, "dtype"):
        clipped = np.clip(samples, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2", copy=False)
        return pcm16.tobytes()
    pcm = bytearray()
    for s in samples:
        clipped = max(-1.0, min(1.0, float(s)))
        pcm += struct.pack("<h", int(clipped * 32767.0))
    return bytes(pcm)


def _samples_to_wav(samples: list[float], sample_rate: int) -> bytes:
    """Encode float samples in [-1, 1] as a WAV byte string (PCM 16-bit mono).

    Built by hand with :mod:`struct` rather than the stdlib :mod:`wave`
    module. ``wave`` transitively imports :mod:`audioop`, which is
    *removed* in stripped embedded Pythons (Ren'Py, PyInstaller
    ``--exclude-module audioop``, custom Bazel/Buck toolchains) — and
    the import is at MODULE LOAD time, so even reaching this function
    fails before we can touch a single byte. By formatting the
    RIFF/WAVE/fmt /data chunk headers ourselves we keep the engine
    importable everywhere ``sherpa_onnx`` itself is.

    Format details (see http://soundfile.sapp.org/doc/WaveFormat/):

      - RIFF header: ``b"RIFF" + <total size - 8> + b"WAVE"``
      - fmt  chunk:  ``b"fmt " + 16 + PCM(1) + channels + sample_rate
                       + byte_rate + block_align + bits_per_sample``
      - data chunk:  ``b"data" + <pcm size> + <pcm bytes>``

    Mono / 16-bit / little-endian — same shape ``wave.open(..., "wb")``
    used to produce, byte-identical for typical inputs.
    """
    pcm = bytearray()
    for s in samples:
        clipped = max(-1.0, min(1.0, s))
        pcm += struct.pack("<h", int(clipped * 32767.0))

    n_channels = 1
    sample_width = 2  # bytes per sample (PCM16)
    byte_rate = sample_rate * n_channels * sample_width
    block_align = n_channels * sample_width
    bits_per_sample = sample_width * 8
    pcm_size = len(pcm)
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,  # subchunk1 size for PCM
        1,  # AudioFormat = PCM
        n_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    data_header = struct.pack("<4sI", b"data", pcm_size)
    riff_payload_size = 4 + len(fmt_chunk) + len(data_header) + pcm_size
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_payload_size, b"WAVE")

    buf = BytesIO()
    buf.write(riff_header)
    buf.write(fmt_chunk)
    buf.write(data_header)
    buf.write(pcm)
    return buf.getvalue()
