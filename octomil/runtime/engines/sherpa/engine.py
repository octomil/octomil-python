"""sherpa-onnx engine plugin -- on-device text-to-speech via sherpa-onnx.

sherpa-onnx (k2-fsa) ships VITS/Piper/Kokoro TTS models packaged as ONNX.
This plugin wraps the sherpa-onnx Python bindings so TTS models register
with the octomil engine registry under the canonical ``sherpa-onnx``
executor id.

Unlike LLM engines, TTS does NOT use ``generate()`` / ``generate_stream()``.
Instead, the backend exposes :meth:`_SherpaTtsBackend.synthesize` for
non-streaming requests and :meth:`_SherpaTtsBackend.synthesize_stream`
for the typed PCM event stream consumed by
``client.audio.speech.stream(...)``. Both flow through the same
:class:`OfflineTts` instance; the streaming path uses sherpa's
``callback`` argument which is invoked on each generated audio chunk.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
import threading
import time
from collections import deque
from io import BytesIO
from typing import Any, AsyncIterator

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.runtime.core.base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Supported TTS models -- name -> (family, default voice).
# family selects the sherpa-onnx config path:
#   "kokoro" -> OfflineTtsKokoroModelConfig (model + voices.bin + tokens + data_dir)
#   "vits"   -> OfflineTtsVitsModelConfig   (Piper-style: model.onnx + tokens + data_dir)
# Voice catalogs are model-specific; the second tuple element is the default
# voice the backend uses when the request does not specify one.
_SHERPA_TTS_MODELS: dict[str, tuple[str, str]] = {
    "kokoro-82m": ("kokoro", "af_bella"),
    "piper-en-amy": ("vits", "amy"),
    "piper-en-ryan": ("vits", "ryan"),
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

# kokoro-en-v0_19 — the bundle currently shipped under the
# ``kokoro-82m`` static recipe.
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

# Per-model legacy fallback catalog. Used ONLY when no voices.txt
# sidecar is present. Keep tightly scoped: an unknown model id
# falls through to "fail loudly" so callers can't accidentally
# inherit some other artifact's catalog.
_LEGACY_KOKORO_FALLBACK_CATALOGS: dict[str, tuple[str, ...]] = {
    "kokoro-82m": _KOKORO_EN_V0_19_VOICES,
    "kokoro-en-v0_19": _KOKORO_EN_V0_19_VOICES,
}

# Back-compat alias. Old import path
# ``octomil.runtime.engines.sherpa._KOKORO_VOICES`` resolves to the
# active artifact's catalog so external callers keep working, but
# the canonical accessor is ``catalog_for_model(model_name)``.
_KOKORO_VOICES: tuple[str, ...] = _KOKORO_EN_V0_19_VOICES


def catalog_for_model(model_name: str) -> tuple[str, ...]:
    """Return the legacy fallback voice catalog for ``model_name``.

    Empty tuple when the model has no declared catalog. Callers that
    need authoritative ordering should read ``voices.txt`` from the
    prepared artifact directory; this helper is the *fallback* used
    only when the sidecar is missing.
    """
    return _LEGACY_KOKORO_FALLBACK_CATALOGS.get(model_name.lower(), ())


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
          - kokoro: OfflineTtsKokoroModelConfig(model, voices, tokens, data_dir)
          - vits:   OfflineTtsVitsModelConfig(model, tokens, data_dir)
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
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=os.path.join(model_dir, "model.onnx"),
                    voices=os.path.join(model_dir, "voices.bin"),
                    tokens=os.path.join(model_dir, "tokens.txt"),
                    data_dir=os.path.join(model_dir, "espeak-ng-data"),
                ),
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
        else:
            raise ValueError(f"Unsupported sherpa-onnx TTS family '{family}' for model '{model_name}'.")

        config = sherpa_onnx.OfflineTtsConfig(model=inner_model_config)
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
        """
        if not text.strip():
            raise ValueError("text must not be empty")
        if speed <= 0:
            raise ValueError("speed must be positive")

        if self._tts is None:
            self.load_model(self._model_name)
        assert self._tts is not None

        sid, reported_voice = self.validate_voice(voice)

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
            "voice": reported_voice,
            "model": self._model_name,
        }

    def _resolve_sid(self, explicit_voice: str) -> int:
        """Map a *caller-supplied* voice name to a sherpa-onnx speaker id.

        ``explicit_voice`` MUST be the empty string when the caller did
        not specify one — the prior implementation collapsed the
        per-model default label (``self._default_voice``) into the
        same parameter as the explicit choice, which made
        ``Piper -> default 'amy'`` indistinguishable from
        ``caller passed voice='amy'``. For Piper the default label is
        not in any voice manifest, so the lookup raised
        ``voice_not_supported_for_model`` even when the caller passed
        ``voice=None``.

        Resolution rules:

          - Empty ``explicit_voice`` → ``sid=0``. The backend's first
            speaker is the contracted default for every model,
            including catalog-less ones (Piper / single-speaker VITS).
          - ``voices.txt`` sidecar in the prepared artifact dir is
            authoritative for THIS artifact. Position == speaker id.
          - When no sidecar is present, fall back to a per-model
            legacy catalog (``catalog_for_model``).
          - When the caller passed an explicit voice the catalog
            doesn't recognize, raise ``voice_not_supported_for_model``
            instead of silently aliasing to ``sid=0``.
        """
        if not explicit_voice:
            return 0

        model_dir = self._resolve_model_dir(self._model_name)
        manifest = _read_voice_manifest(model_dir)
        if not manifest:
            manifest = catalog_for_model(self._model_name)

        if not manifest:
            # Single-speaker / unknown-catalog model and the caller
            # gave an explicit voice. Refuse rather than silently
            # alias; empty voice is already handled above.
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

        target = explicit_voice.strip().lower()
        for idx, name in enumerate(manifest):
            if name.lower() == target:
                return idx

        raise OctomilError(
            code=OctomilErrorCode.INVALID_INPUT,
            message=(
                f"voice_not_supported_for_model: voice {explicit_voice!r} is not in "
                f"the speaker catalog for model {self._model_name!r}. "
                f"Supported voices: {', '.join(manifest)}."
            ),
        )

    def _voice_to_sid(self, voice: str) -> int:
        """Backwards-compatible alias for ``_resolve_sid``.

        Kept for any external callers that imported the older name.
        Prefer :meth:`_resolve_sid`.
        """
        return self._resolve_sid((voice or "").strip())

    def validate_voice(self, voice: str | None) -> tuple[int, str]:
        """Resolve a caller-supplied voice synchronously.

        Returns ``(sid, resolved_label)``. Raises ``OctomilError`` with
        ``voice_not_supported_for_model`` when an explicit voice is
        unsupported. Public surface so the kernel and HTTP route can
        validate *before* a stream's first event / response status is
        committed — without this check, the async-generator setup in
        :meth:`synthesize_stream` only raises after the consumer has
        already started iterating.

        Default-label resolution: when ``voice`` is empty/None and the
        model has a manifest, the label returned is ``manifest[0]``
        (the actual sid=0 speaker). This fixes the Kokoro drift where
        ``_default_voice='af_bella'`` was reported even though sid=0
        is ``af``. Catalog-less models (Piper) fall back to
        ``self._default_voice`` since there's no authoritative source.
        """
        explicit = (voice or "").strip()
        sid = self._resolve_sid(explicit)
        if explicit:
            return sid, explicit
        manifest = _read_voice_manifest(self._resolve_model_dir(self._model_name))
        if not manifest:
            manifest = catalog_for_model(self._model_name)
        if manifest:
            return 0, manifest[0]
        return 0, self._default_voice or ""

    def list_models(self) -> list[str]:
        return [self._model_name] if self._model_name else []

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    @property
    def supports_streaming(self) -> bool:
        """sherpa-onnx OfflineTts.generate exposes a per-chunk callback."""
        return True

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        *,
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

        Args:
            text: utterance to synthesize
            voice: voice id (catalog-validated by the caller; here we
                just translate name->sid)
            speed: 1.0 = natural pace
            chunk_max_queue: max number of audio chunks held in the
                bridge before the producer thread blocks (backpressure).
        """
        if not text.strip():
            raise ValueError("text must not be empty")
        if speed <= 0:
            raise ValueError("speed must be positive")

        if self._tts is None:
            self.load_model(self._model_name)
        assert self._tts is not None

        # validate_voice runs before the worker thread spins up, so
        # voice-not-supported is raised from the synchronous entry to
        # the async generator (i.e. before any consumer iteration)
        # rather than emerging as a mid-stream exception.
        sid, _resolved_label = self.validate_voice(voice)

        bridge = _StreamBridge(maxsize=chunk_max_queue)
        loop = asyncio.get_running_loop()
        sample_rate = self._tts.sample_rate or self._sample_rate

        def _worker() -> None:
            """Run sherpa generate on a thread; push chunks to bridge."""
            try:

                def _callback(samples_f32: Any, _progress: float) -> int:
                    # samples_f32 is a numpy.ndarray[float32] of length n.
                    # Convert to PCM int16 LE bytes inline so the GIL is
                    # released before re-entering sherpa.
                    if bridge.is_cancelled():
                        return 0
                    pcm = _float32_to_pcm_s16le_bytes(samples_f32)
                    n = int(getattr(samples_f32, "size", len(samples_f32)))
                    # Blocks if the bridge is full; cancellation breaks
                    # out and returns 0 so sherpa stops synthesis.
                    accepted = bridge.put(pcm, n)
                    return 1 if accepted else 0

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
                # run_in_executor on a small helper so the asyncio loop
                # is never blocked waiting for the worker.
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
            # Whether we exited cleanly, by exception, or by the consumer
            # closing the generator: cancel + drain so the worker thread
            # never leaks. join() is bounded because the worker checks
            # cancel between callback invocations.
            bridge.cancel()
            worker.join(timeout=5.0)


_STREAM_SENTINEL_DONE = object()


class _StreamBridge:
    """Bounded thread-safe bridge between the sherpa worker thread and
    the asyncio consumer.

    Why hand-rolled instead of :class:`queue.Queue` /
    ``loop.call_soon_threadsafe(queue.put_nowait, ...)``:

    * Real backpressure: ``put`` blocks the worker thread when the
      bridge is full so the producer cannot outrun a slow consumer
      and grow memory unboundedly.
    * Cooperative cancellation: ``put`` returns ``False`` when
      cancelled so the sherpa callback can return 0 and stop synthesis.
    * No coupling to a specific event loop from the producer side.
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
        """Block until there is room (or we're cancelled). Returns
        True if the chunk was accepted, False if cancelled."""
        with self._not_full:
            while len(self._buf) >= self._maxsize and not self._cancelled and not self._closed:
                self._not_full.wait(timeout=0.5)
            if self._cancelled or self._closed:
                return False
            self._buf.append((pcm, num_samples))
            self._not_empty.notify()
            return True

    def get(self) -> Any:
        """Block until a chunk is available or the stream is done.

        Returns ``_STREAM_SENTINEL_DONE`` on completion (with optional
        error preserved for ``raise_if_error``).
        """
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
