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
    from octomil.audio.speech import SpeechResponse


# Canonical sample format the SDK exposes. Engines that produce float32
# samples must convert before yielding chunks.
SAMPLE_FORMAT_PCM_S16LE = "pcm_s16le"
SUPPORTED_STREAM_FORMATS = (SAMPLE_FORMAT_PCM_S16LE, "wav")


class StreamingMode(str, Enum):
    """Whether the producer is actually streaming or faking it."""

    #: Audio chunks arrive incrementally as samples are produced.
    REALTIME = "realtime"
    #: A single chunk containing the full audio, emitted at completion.
    #: Clients should NOT assume low first-byte latency.
    FINAL_CHUNK = "final_chunk"


@dataclass(frozen=True)
class SpeechStreamStarted:
    """First event of every stream. Always arrives before audio chunks."""

    model: str
    voice: Optional[str]
    sample_rate: int
    channels: int
    sample_format: str
    streaming_mode: StreamingMode
    locality: str  # "on_device" | "cloud"
    engine: Optional[str] = None
    request_id: Optional[str] = None


@dataclass(frozen=True)
class SpeechAudioChunk:
    """A single audio chunk."""

    data: bytes
    sample_index: int  # cumulative samples produced through end of this chunk
    timestamp_ms: int  # cumulative duration_ms through end of this chunk
    is_final: bool = False


@dataclass(frozen=True)
class SpeechStreamCompleted:
    """Final event of a successful stream."""

    duration_ms: int
    total_samples: int
    sample_rate: int
    channels: int
    sample_format: str
    streaming_mode: StreamingMode
    latency_ms: float = 0.0
    first_chunk_ms: Optional[float] = None  # latency to first audio chunk


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
    ) -> None:
        self._producer = producer
        self._on_cancel = on_cancel
        self._closed = False
        self._started_event: Optional[SpeechStreamStarted] = None

    def __aiter__(self) -> "SpeechStream":
        return self

    async def __anext__(self) -> SpeechStreamEvent:
        if self._closed:
            raise StopAsyncIteration
        try:
            event = await self._producer.__anext__()
        except StopAsyncIteration:
            self._closed = True
            raise
        if isinstance(event, SpeechStreamStarted) and self._started_event is None:
            self._started_event = event
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
            latency_ms=completed.latency_ms,
            route=SpeechRoute(
                locality=started.locality,
                engine=started.engine,
                policy=None,
                fallback_used=False,
            ),
            billed_units=None,
            unit_kind=None,
        )


__all__ = [
    "SAMPLE_FORMAT_PCM_S16LE",
    "SUPPORTED_STREAM_FORMATS",
    "StreamingMode",
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
