"""Streaming text-to-speech primitives for the Octomil SDK.

The canonical wire format is interleaved PCM int16 little-endian
(``pcm_s16le``). WAV is produced only at the end via the
:class:`PcmWavFinalizer` helper or as a single ``final_chunk`` event when
the underlying engine cannot stream.

Stream events arrive in this order::

    SpeechStreamStarted
    SpeechAudioChunk (zero or more, ``is_final=False``)
    SpeechAudioChunk (one, ``is_final=True``)        # optional sentinel
    SpeechStreamCompleted

On error a :class:`SpeechStreamError` event is yielded and the underlying
:class:`octomil.errors.OctomilError` is re-raised on the next iteration.
The HTTP wire format is intentionally simpler: metadata in response
headers, raw PCM body, completion via clean EOF.
"""

from __future__ import annotations

import asyncio
import os
import struct
import tempfile
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from octomil.audio.metrics import TtsMetricsCollector
    from octomil.audio.speech import SpeechResponse


# Canonical sample format the SDK exposes. Engines that produce float32
# samples must convert before yielding chunks.
SAMPLE_FORMAT_PCM_S16LE = "pcm_s16le"
SUPPORTED_STREAM_FORMATS = (SAMPLE_FORMAT_PCM_S16LE, "wav")


class TtsStreamingMode(str, Enum):
    """Honest streaming behaviour the engine actually delivers.

    The mode is *advertised* by :class:`TtsStreamingCapability` and
    *verified* on completion. If the engine advertises
    ``sentence_chunk`` but the synthesis run produces only one chunk,
    the completion event flags ``capability_verified=False`` so callers
    can stop trusting the advertised label.
    """

    #: Engine yields the entire utterance in a single audio chunk.
    #: No first-byte-latency benefit — equivalent to the non-streaming
    #: ``create()`` path with the streaming wire shape.
    FINAL_CHUNK = "final_chunk"

    #: Engine yields one chunk per sentence. Multi-sentence input gets
    #: TTFB benefit; single-sentence input degrades to ``final_chunk``
    #: (the SDK reports the actually-observed mode in completion).
    SENTENCE_CHUNK = "sentence_chunk"

    #: Engine yields sub-sentence chunks (frame / phoneme / waveform
    #: rolling buffer). True low-latency streaming. No production engine
    #: in-tree advertises this today; reserved for future backends
    #: (XTTS streaming, custom Kokoro patches, etc.).
    PROGRESSIVE = "progressive"


# v4.13 compatibility alias.
#
# v4.13 shipped this enum as ``StreamingMode``. Renaming it to
# ``TtsStreamingMode`` was correct (the SDK's other audio-stream
# types use the ``Tts`` prefix) but ``StreamingMode`` was already
# imported by application code. Keep both names exported so
# ``from octomil.audio.streaming import StreamingMode`` keeps
# working through the deprecation window. Do NOT remove this
# alias without bumping the SDK to a major version and recording
# the removal in the migration guide.
StreamingMode = TtsStreamingMode


class TtsStreamingGranularity(str, Enum):
    """The smallest audio unit the engine commits to per chunk."""

    UTTERANCE = "utterance"  # whole input -> one chunk
    SENTENCE = "sentence"  # per-sentence boundary
    FRAME = "frame"  # sub-sentence (target for progressive)


@dataclass(frozen=True)
class TtsStreamingCapability:
    """What the engine *claims* it can do for a given (model, input).

    ``verified`` is ``True`` once the SDK has observed the engine
    actually delivering the advertised cadence on this model+artifact.
    Until then it's an unverified declaration; the completion event
    will flag a downgrade if the run did not match.

    Construct via :func:`TtsStreamingCapability.final_only` /
    ``.sentence`` / ``.progressive`` for readability.
    """

    mode: TtsStreamingMode
    granularity: TtsStreamingGranularity
    verified: bool = False

    @classmethod
    def final_only(cls, *, verified: bool = False) -> "TtsStreamingCapability":
        return cls(
            mode=TtsStreamingMode.FINAL_CHUNK,
            granularity=TtsStreamingGranularity.UTTERANCE,
            verified=verified,
        )

    @classmethod
    def sentence(cls, *, verified: bool = False) -> "TtsStreamingCapability":
        return cls(
            mode=TtsStreamingMode.SENTENCE_CHUNK,
            granularity=TtsStreamingGranularity.SENTENCE,
            verified=verified,
        )

    @classmethod
    def progressive(cls, *, verified: bool = False) -> "TtsStreamingCapability":
        return cls(
            mode=TtsStreamingMode.PROGRESSIVE,
            granularity=TtsStreamingGranularity.FRAME,
            verified=verified,
        )


@dataclass(frozen=True)
class SpeechStreamStarted:
    """First event of every stream. Always arrives before audio chunks.

    ``streaming_capability`` is the authoritative source for expected
    cadence — read ``streaming_capability.mode`` to decide whether to
    start playback eagerly (``sentence_chunk`` / ``progressive``) or
    buffer (``final_chunk``).

    ``priority`` echoes the request's
    :class:`octomil.audio.scheduler.TtsRequestPriority`. Surfacing
    it on the started event lets observability tooling correlate
    setup_ms latency with the priority tier the SDK accepted —
    "this 850ms setup_ms was a SPECULATIVE call that queued behind
    a FOREGROUND" is a more useful signal than a raw number.
    """

    model: str
    voice: Optional[str]
    sample_rate: int
    channels: int
    sample_format: str
    streaming_capability: TtsStreamingCapability
    locality: str  # "on_device" | "cloud"
    engine: Optional[str] = None
    request_id: Optional[str] = None
    priority: Optional[str] = None  # TtsRequestPriority.value

    # v4.13 compatibility: applications that read
    # ``event.streaming_mode`` (the v4.13 spelling) keep working.
    # The canonical field is ``streaming_capability.mode``; this
    # property just forwards the underlying enum.
    @property
    def streaming_mode(self) -> TtsStreamingMode:
        return self.streaming_capability.mode


@dataclass(frozen=True)
class SpeechAudioChunk:
    """A single audio chunk."""

    data: bytes
    sample_index: int  # cumulative samples produced through end of this chunk
    timestamp_ms: int  # cumulative duration_ms through end of this chunk
    is_final: bool = False


@dataclass(frozen=True)
class SpeechStreamCompleted:
    """Final event of a successful stream.

    Honest metrics — every duration is named for the boundary it
    measures, so callers can attribute latency without guessing:

    * ``setup_ms`` — SDK call entry (``client.audio.speech.stream``)
      to ``SpeechStreamStarted`` emitted. Includes routing, voice
      validation, backend acquisition, scheduler queue time.
    * ``engine_first_chunk_ms`` — engine synthesis start to first
      PCM bytes pushed by the engine. The "raw engine TTFB."
    * ``e2e_first_chunk_ms`` — SDK call entry to first
      :class:`SpeechAudioChunk` observed by the consumer. The
      customer-visible TTFB.
    * ``total_latency_ms`` — SDK call entry to
      :class:`SpeechStreamCompleted`. Wall time end-to-end.

    ``observed_chunks`` lets callers verify the advertised
    capability against reality; ``capability_verified`` is False
    when the engine claimed a finer cadence than it delivered.
    """

    duration_ms: int
    total_samples: int
    sample_rate: int
    channels: int
    sample_format: str
    streaming_capability: TtsStreamingCapability
    setup_ms: float
    engine_first_chunk_ms: Optional[float]
    e2e_first_chunk_ms: Optional[float]
    total_latency_ms: float
    observed_chunks: int
    capability_verified: bool
    # Scheduler queue-wait time, in milliseconds. ``setup_ms``
    # already *includes* this number — split out so observability
    # tooling can attribute "this 850ms setup_ms was actually
    # 800ms queued behind a foreground" without subtracting from
    # the headline metric. ``0.0`` for the no-contention fast path.
    queued_ms: float = 0.0
    priority: Optional[str] = None  # TtsRequestPriority.value

    # v4.13 compatibility properties.
    #
    # v4.13 named the fields ``streaming_mode`` / ``latency_ms`` /
    # ``first_chunk_ms``. Renaming was correct — the new names
    # disambiguate which boundary each metric measures — but
    # existing code reading the old names would break without
    # mapping properties. These map onto the canonical fields:
    #
    #   - ``streaming_mode`` → ``streaming_capability.mode``
    #   - ``latency_ms``     → ``total_latency_ms``
    #   - ``first_chunk_ms`` → ``e2e_first_chunk_ms``
    #
    # Do NOT remove without bumping the SDK to a major version
    # and recording the removal in the migration guide.
    @property
    def streaming_mode(self) -> TtsStreamingMode:
        return self.streaming_capability.mode

    @property
    def latency_ms(self) -> float:
        return self.total_latency_ms

    @property
    def first_chunk_ms(self) -> Optional[float]:
        return self.e2e_first_chunk_ms


@dataclass(frozen=True)
class SpeechStreamError:
    """Mid-stream error event. The stream will re-raise on next iteration."""

    code: str  # OctomilErrorCode value
    message: str


SpeechStreamEvent = Union[
    SpeechStreamStarted,
    SpeechAudioChunk,
    SpeechStreamCompleted,
    SpeechStreamError,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pcm_s16le_to_wav_bytes(
    pcm: bytes,
    sample_rate: int,
    channels: int = 1,
) -> bytes:
    """Wrap raw PCM int16 little-endian samples in a valid RIFF/WAVE container.

    Hand-rolled with :mod:`struct` (mirrors the engine's ``_samples_to_wav``)
    so the helper is importable in stripped Python builds that lack
    :mod:`audioop`/:mod:`wave` (Ren'Py, PyInstaller bundles).
    """
    n_channels = channels
    sample_width = 2
    byte_rate = sample_rate * n_channels * sample_width
    block_align = n_channels * sample_width
    bits_per_sample = sample_width * 8
    pcm_size = len(pcm)
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,  # PCM
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


class PcmWavFinalizer:
    """Accumulate PCM int16 chunks and emit a finalized WAV.

    Cheap, pure-Python, no buffering of decoded samples — just a list of
    byte slices joined at finalize time.
    """

    def __init__(self, sample_rate: int, channels: int = 1) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunks: list[bytes] = []
        self._size = 0

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._chunks.append(chunk)
        self._size += len(chunk)

    @property
    def total_bytes(self) -> int:
        return self._size

    def pcm_bytes(self) -> bytes:
        return b"".join(self._chunks)

    def to_wav_bytes(self) -> bytes:
        return pcm_s16le_to_wav_bytes(self.pcm_bytes(), self._sample_rate, self._channels)


class ChunkAccumulator:
    """Collect events from a :class:`SpeechStream` into a final result.

    Returns ``(metadata, pcm_bytes, completed_event)``. Useful when an app
    wants both incremental playback AND a finalized buffer to cache or
    persist.
    """

    def __init__(self) -> None:
        self.started: Optional[SpeechStreamStarted] = None
        self.completed: Optional[SpeechStreamCompleted] = None
        self._chunks: list[bytes] = []

    def consume(self, event: SpeechStreamEvent) -> None:
        if isinstance(event, SpeechStreamStarted):
            self.started = event
        elif isinstance(event, SpeechAudioChunk):
            if event.data:
                self._chunks.append(event.data)
        elif isinstance(event, SpeechStreamCompleted):
            self.completed = event

    def pcm_bytes(self) -> bytes:
        return b"".join(self._chunks)

    def to_wav_bytes(self) -> bytes:
        if self.started is None:
            raise RuntimeError("ChunkAccumulator: no SpeechStreamStarted observed")
        return pcm_s16le_to_wav_bytes(
            self.pcm_bytes(),
            self.started.sample_rate,
            self.started.channels,
        )


class FileSpooler:
    """Stream PCM chunks to a temp file and atomically finalize a WAV.

    Helper-only — the core SDK stream is bytes-in-memory. This is the
    persistence/caching adapter for callers that want to write while
    receiving chunks (game caches, prefetch systems, debug dumps).

    Usage::

        spool = FileSpooler("voice.wav")
        async with client.audio.speech.stream(...) as s:
            async for ev in s:
                spool.consume(ev)
        path = spool.finalize()
    """

    def __init__(self, target_wav_path: str, *, dir: Optional[str] = None) -> None:
        self._target = os.path.abspath(target_wav_path)
        spool_dir = dir or os.path.dirname(self._target) or None
        fd, self._tmp_path = tempfile.mkstemp(
            prefix=".octomil-tts-",
            suffix=".pcm",
            dir=spool_dir,
        )
        self._fp = os.fdopen(fd, "wb")
        self._sample_rate: Optional[int] = None
        self._channels: int = 1
        self._size = 0
        self._closed = False
        self._finalized = False

    def consume(self, event: SpeechStreamEvent) -> None:
        if isinstance(event, SpeechStreamStarted):
            self._sample_rate = event.sample_rate
            self._channels = event.channels
        elif isinstance(event, SpeechAudioChunk):
            if event.data and not self._closed:
                self._fp.write(event.data)
                self._size += len(event.data)

    def abort(self) -> None:
        """Discard the spool without producing a WAV."""
        if self._closed:
            return
        self._closed = True
        try:
            self._fp.close()
        finally:
            try:
                os.unlink(self._tmp_path)
            except FileNotFoundError:
                pass

    def finalize(self) -> str:
        """Atomically rename the spool to a WAV at ``target_wav_path``.

        Reads the spooled PCM, writes WAV bytes to a sibling temp file,
        and renames into place so partial writes are never visible.
        """
        if self._finalized:
            return self._target
        if self._sample_rate is None:
            self.abort()
            raise RuntimeError("FileSpooler.finalize: no SpeechStreamStarted observed")

        self._fp.flush()
        self._fp.close()
        self._closed = True

        with open(self._tmp_path, "rb") as src:
            pcm = src.read()
        wav = pcm_s16le_to_wav_bytes(pcm, self._sample_rate, self._channels)

        # Sibling-file + rename for atomicity. dir=parent so rename is on
        # the same filesystem.
        parent = os.path.dirname(self._target) or "."
        fd, tmp_wav = tempfile.mkstemp(prefix=".octomil-tts-", suffix=".wav.tmp", dir=parent)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(wav)
            os.replace(tmp_wav, self._target)
        except BaseException:
            try:
                os.unlink(tmp_wav)
            except FileNotFoundError:
                pass
            raise
        finally:
            try:
                os.unlink(self._tmp_path)
            except FileNotFoundError:
                pass

        self._finalized = True
        return self._target


# ---------------------------------------------------------------------------
# SpeechStream
# ---------------------------------------------------------------------------


# A producer is an async iterator over events. It is responsible for:
#   * yielding SpeechStreamStarted as the very first event
#   * yielding zero or more SpeechAudioChunk events
#   * yielding exactly one SpeechStreamCompleted on success
#   * raising OctomilError (after optionally yielding SpeechStreamError) on failure
#   * honoring cancellation: when its caller calls .aclose() / cancel(),
#     stop synthesis and clean up worker threads/temp files
SpeechStreamProducer = AsyncIterator[SpeechStreamEvent]


class SpeechStream:
    """Async iterator over a stream of :class:`SpeechStreamEvent`.

    Created by :meth:`octomil.audio.speech.FacadeSpeech.stream`. Do not
    instantiate directly.

    Usage::

        async with client.audio.speech.stream(model=..., input=...) as stream:
            async for event in stream:
                if isinstance(event, SpeechAudioChunk):
                    play(event.data)

    The stream MUST be either fully consumed or closed. ``async with``
    handles that automatically; otherwise call ``await stream.aclose()``.
    """

    def __init__(
        self,
        producer: SpeechStreamProducer,
        *,
        on_cancel: Optional[Callable[[], Awaitable[None]]] = None,
        metrics_collector: Optional["TtsMetricsCollector"] = None,
    ) -> None:
        self._producer = producer
        self._on_cancel = on_cancel
        self._closed = False
        self._started_event: Optional[SpeechStreamStarted] = None
        # Optional observability collector. When provided, every event
        # is fed into it before being returned to the caller; the
        # collector emits typed events (started / first_audio_chunk /
        # completed / cancelled / failed) to its configured sink.
        # Passing ``None`` keeps the legacy no-op behavior.
        self._metrics = metrics_collector
        self._completed_observed = False

    def __aiter__(self) -> "SpeechStream":
        return self

    async def __anext__(self) -> SpeechStreamEvent:
        if self._closed:
            raise StopAsyncIteration
        try:
            event = await self._producer.__anext__()
        except StopAsyncIteration:
            self._closed = True
            # Stream ended cleanly. If we never observed a Completed
            # event, that's an honest cancellation (consumer broke
            # mid-iteration). The aclose path handles the reverse.
            if self._metrics is not None and not self._completed_observed:
                from octomil.audio.metrics import CancellationSource

                self._metrics.on_cancel(CancellationSource.CLIENT_ACLOSE)
            raise
        except BaseException as exc:
            # Producer raised. Re-route through metrics with the
            # bounded error_code taxonomy before re-raising.
            if self._metrics is not None and not self._completed_observed:
                self._metrics.on_fail(error_code=_map_error_to_taxonomy(exc))
                # Fall through so the caller still sees the exception.
            self._closed = True
            raise
        if isinstance(event, SpeechStreamStarted) and self._started_event is None:
            self._started_event = event
            if self._metrics is not None:
                self._metrics.on_started(event)
        elif isinstance(event, SpeechAudioChunk) and self._metrics is not None:
            self._metrics.on_chunk(event)
        elif isinstance(event, SpeechStreamCompleted):
            self._completed_observed = True
            if self._metrics is not None:
                self._metrics.on_completed(event)
        return event

    async def __aenter__(self) -> "SpeechStream":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Cancel synthesis and release resources. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Fire ``cancelled`` BEFORE we tear down the producer so the
        # metric reflects the user-initiated abort, not whatever the
        # producer emits during teardown.
        if self._metrics is not None and not self._completed_observed:
            from octomil.audio.metrics import CancellationSource

            self._metrics.on_cancel(CancellationSource.CLIENT_ACLOSE)
        if self._on_cancel is not None:
            try:
                await self._on_cancel()
            except Exception:
                pass
        aclose = getattr(self._producer, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass

    async def cancel(self) -> None:
        """Alias for :meth:`aclose`."""
        await self.aclose()

    @property
    def started(self) -> Optional[SpeechStreamStarted]:
        """The :class:`SpeechStreamStarted` event, once observed."""
        return self._started_event

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    async def collect(self) -> "tuple[SpeechStreamStarted, bytes, SpeechStreamCompleted]":
        """Drain the stream and return ``(started, pcm_bytes, completed)``.

        Raises if the stream errored or never produced started/completed
        events.
        """
        acc = ChunkAccumulator()
        async for event in self:
            acc.consume(event)
        if acc.started is None or acc.completed is None:
            raise RuntimeError("SpeechStream.collect: stream ended without started/completed events")
        return acc.started, acc.pcm_bytes(), acc.completed

    async def to_speech_response(self) -> "SpeechResponse":
        """Drain the stream and return a non-streaming :class:`SpeechResponse`.

        The returned response carries the finalized WAV bytes. Useful as a
        compatibility shim for callers that want streaming first-byte
        latency but a single object at the end.
        """
        from octomil.audio.speech import SpeechResponse, SpeechRoute  # noqa: WPS433 (cycle)

        started, pcm, completed = await self.collect()
        wav_bytes = pcm_s16le_to_wav_bytes(pcm, started.sample_rate, started.channels)
        return SpeechResponse(
            audio_bytes=wav_bytes,
            content_type="audio/wav",
            format="wav",
            model=started.model,
            provider=None,
            voice=started.voice,
            sample_rate=started.sample_rate,
            duration_ms=completed.duration_ms,
            latency_ms=completed.total_latency_ms,
            route=SpeechRoute(
                locality=started.locality,
                engine=started.engine,
                policy=None,
                fallback_used=False,
            ),
            billed_units=None,
            unit_kind=None,
        )


def _map_error_to_taxonomy(exc: BaseException) -> str:
    """Project an arbitrary exception onto the bounded
    :class:`octomil.audio.metrics.TtsErrorCode` set. Free-form error
    strings would explode Prometheus cardinality."""
    from octomil.audio.metrics import TtsErrorCode  # local import to avoid cycle

    code = getattr(exc, "code", None)
    code_value = getattr(code, "value", None) or (code if isinstance(code, str) else None)
    msg = (str(exc) or "").lower()
    if isinstance(exc, asyncio.CancelledError):
        return TtsErrorCode.CANCELLED.value
    if code_value:
        if "digest" in code_value or "checksum" in code_value:
            return TtsErrorCode.ARTIFACT_DIGEST_MISMATCH.value
        if "artifact" in code_value or "missing" in code_value:
            return TtsErrorCode.ARTIFACT_MISSING.value
        if "runtime" in code_value or "load" in code_value:
            return TtsErrorCode.RUNTIME_LOAD_FAILED.value
        if "engine" in code_value or "unavailable" in code_value:
            return TtsErrorCode.ENGINE_UNAVAILABLE.value
        if "deadline" in code_value or "timeout" in code_value:
            return TtsErrorCode.DEADLINE_EXCEEDED.value
    if "synthes" in msg:
        return TtsErrorCode.SYNTHESIS_FAILED.value
    return TtsErrorCode.UNKNOWN.value


__all__ = [
    "SAMPLE_FORMAT_PCM_S16LE",
    "SUPPORTED_STREAM_FORMATS",
    "TtsStreamingMode",
    "StreamingMode",  # v4.13 alias — see TtsStreamingMode docstring
    "TtsStreamingGranularity",
    "TtsStreamingCapability",
    "SpeechStreamStarted",
    "SpeechAudioChunk",
    "SpeechStreamCompleted",
    "SpeechStreamError",
    "SpeechStreamEvent",
    "SpeechStream",
    "SpeechStreamProducer",
    "ChunkAccumulator",
    "FileSpooler",
    "PcmWavFinalizer",
    "pcm_s16le_to_wav_bytes",
]
