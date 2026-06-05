"""AudioTranscriptions — speech-to-text API."""

from __future__ import annotations

import asyncio
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from octomil._generated.message_role import MessageRole
from octomil._generated.model_capability import ModelCapability
from octomil.audio.types import (
    ChunkDiagnostics,
    TranscriptionPartial,
    TranscriptionResult,
    TranscriptionSegment,
)
from octomil.model_ref import ModelRef, ModelRefFactory
from octomil.runtime.core.model_runtime import ModelRuntime
from octomil.runtime.core.types import (
    GenerationConfig,
    RuntimeContentPart,
    RuntimeMessage,
    RuntimeRequest,
    RuntimeResponse,
    SttOptions,
)

# Sample rate the native ``audio.stt.stream`` session expects. The runtime
# resamples internally, but the SDK advertises 16 kHz mono float32 PCM as
# the canonical streaming input — same contract the batch path uses.
_STREAM_SAMPLE_RATE_HZ = 16000

# poll_event timeout (ms) per drain iteration. Bounded so a wedged runtime
# surfaces via the overall deadline rather than blocking forever; matches
# the batch backend's 200ms poll cadence.
_STREAM_POLL_TIMEOUT_MS = 200


@runtime_checkable
class StreamSession(Protocol):
    """Minimal session contract ``stream_transcribe`` drives.

    Structural mirror of :class:`octomil.runtime.native.loader.NativeSession`
    — only the four methods the streaming loop touches. Keeping it a
    Protocol means the facade never imports the native cffi module at
    call time and tests can inject a pure-Python fake.
    """

    def send_audio(self, samples: bytes, *, sample_rate: int, channels: int = 1) -> None: ...

    def end_input(self) -> int: ...

    def poll_event(self, timeout_ms: int = 0) -> object: ...

    def close(self) -> None: ...


# A factory yields a ready-to-feed :class:`StreamSession` for a resolved
# model reference. Injected for tests; the default opens a native
# ``audio.stt.stream`` session against the resolved runtime.
StreamSessionFactory = Callable[[ModelRef], StreamSession]


class _OwnedStreamSession:
    """Keep the native backend alive for as long as its session is alive."""

    def __init__(self, session: StreamSession, owner: object) -> None:
        self._session = session
        self._owner = owner

    def send_audio(self, samples: bytes, *, sample_rate: int, channels: int = 1) -> None:
        self._session.send_audio(samples, sample_rate=sample_rate, channels=channels)

    def end_input(self) -> int:
        return self._session.end_input()

    def poll_event(self, timeout_ms: int = 0) -> object:
        return self._session.poll_event(timeout_ms=timeout_ms)

    def close(self) -> None:
        try:
            self._session.close()
        finally:
            close_owner = getattr(self._owner, "close", None)
            if callable(close_owner):
                close_owner()


class AudioTranscriptions:
    """Audio transcription API.

    Wraps the underlying audio runtime to provide speech-to-text.

    Usage::

        result = await client.audio.transcriptions.create(
            audio=audio_bytes
        )
        print(result.text)
    """

    def __init__(
        self,
        runtime_resolver: Callable[[ModelRef], Optional[ModelRuntime]],
        *,
        stream_session_factory: Optional[StreamSessionFactory] = None,
    ) -> None:
        self._runtime_resolver = runtime_resolver
        # Optional injection seam for tests / alternate transports. When
        # ``None`` the default factory opens a native ``audio.stt.stream``
        # session against the resolved runtime (lazy native import).
        self._stream_session_factory = stream_session_factory

    async def create(
        self,
        audio: bytes,
        *,
        model: Optional[ModelRef] = None,
        language: Optional[str] = None,
        response_format: Optional[str] = None,
        chunk_window_ms: Optional[int] = None,
        chunk_overlap_ms: Optional[int] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Raw audio data (WAV, MP3, etc.).
            model: Model reference. Defaults to transcription capability.
            language: Optional language hint (BCP 47 code, e.g. "en").
            response_format: Optional output format hint.
            chunk_window_ms: Optional fixed decode-window size for chunked
                transcription (native path only). ``None`` (default) runs
                a single full-buffer decode — byte-identical to the
                pre-v0.1.27 behaviour.
            chunk_overlap_ms: Optional overlap between consecutive decode
                windows. Ignored unless ``chunk_window_ms`` is set.

        Returns:
            TranscriptionResult with the transcribed text. On the native
            path the result also carries ``segments`` (with per-segment
            ``avg_logprob`` / ``no_speech_prob``), ``duration_ms``, and —
            when ``chunk_window_ms`` is set — ``chunk_diagnostics``.
        """
        ref = model or ModelRefFactory.capability(ModelCapability.TRANSCRIPTION)
        runtime = self._runtime_resolver(ref)
        if runtime is None:
            raise RuntimeError("No runtime available for transcription model")

        parts = [RuntimeContentPart.audio_part(audio, "audio/wav")]
        if language:
            parts.append(RuntimeContentPart.text_part(language))
        stt_options: Optional[SttOptions] = None
        if chunk_window_ms is not None or chunk_overlap_ms is not None:
            stt_options = SttOptions(
                chunk_window_ms=chunk_window_ms,
                chunk_overlap_ms=chunk_overlap_ms,
            )
        request = RuntimeRequest(
            messages=[RuntimeMessage(role=MessageRole.USER, parts=parts)],
            generation_config=GenerationConfig(max_tokens=0, temperature=0.0),
            stt_options=stt_options,
        )
        response = await runtime.run(request)
        return self._project_result(response, language)

    @staticmethod
    def _project_result(response: RuntimeResponse, language: Optional[str]) -> TranscriptionResult:
        """Project a ``RuntimeResponse`` onto the public result.

        Segments and chunk diagnostics are only populated on the native
        path; legacy / cloud runtimes leave the carriers as ``None`` so
        the public result keeps its empty-segments / ``None``-diagnostics
        defaults without raising.
        """
        segments: list[TranscriptionSegment] = list(response.stt_segments or [])
        diagnostics = response.stt_chunk_diagnostics
        if diagnostics is not None and not isinstance(diagnostics, ChunkDiagnostics):
            # Defensive: a runtime that hands back a None / unexpected
            # shape must not poison the public result. Only project a
            # real ``ChunkDiagnostics``; anything else is treated as
            # "no diagnostics" rather than surfaced verbatim.
            diagnostics = None
        return TranscriptionResult(
            text=response.text,
            language=language,
            segments=segments,
            duration_ms=int(response.stt_duration_ms or 0),
            chunk_diagnostics=diagnostics,
        )

    async def stream(
        self,
        audio: bytes,
        *,
        model: Optional[ModelRef] = None,
    ) -> list[TranscriptionSegment]:
        """Stream transcription segments.

        Args:
            audio: Raw audio data.
            model: Model reference. Defaults to transcription capability.

        Returns:
            List of transcription segments.
        """
        ref = model or ModelRefFactory.capability(ModelCapability.TRANSCRIPTION)
        runtime = self._runtime_resolver(ref)
        if runtime is None:
            raise RuntimeError("No runtime available for transcription model")

        request = RuntimeRequest(
            messages=[
                RuntimeMessage(
                    role=MessageRole.USER,
                    parts=[RuntimeContentPart.audio_part(audio, "audio/wav")],
                )
            ],
            generation_config=GenerationConfig(max_tokens=0, temperature=0.0),
        )
        segments: list[TranscriptionSegment] = []
        async for chunk in runtime.stream(request):
            if chunk.text:
                segments.append(TranscriptionSegment(text=chunk.text))
        return segments

    async def stream_transcribe(
        self,
        audio_chunks: Union[Iterable[bytes], AsyncIterator[bytes]],
        *,
        model: Optional[ModelRef] = None,
        language: Optional[str] = None,
        sample_rate_hz: int = _STREAM_SAMPLE_RATE_HZ,
        deadline_ms: int = 60_000,
    ) -> AsyncGenerator[Union[TranscriptionPartial, TranscriptionSegment], None]:
        """Stream transcription of incrementally-arriving audio.

        Opens an ``audio.stt.stream`` session, feeds each block of
        ``audio_chunks`` (interleaved mono float32 PCM) via
        ``send_audio``, and yields provisional
        :class:`~octomil.audio.types.TranscriptionPartial` revisions as
        the runtime decodes. Once the input iterable is exhausted it calls
        ``end_input`` exactly once and then yields the committed
        :class:`~octomil.audio.types.TranscriptionSegment` finals.

        Ordering contract (preserved end-to-end):

        * Partials are yielded in non-decreasing ``revision_id`` order; a
          newer ``revision_id`` supersedes every earlier partial. Stale
          partials (a ``revision_id`` not greater than the highest already
          seen) are dropped, never yielded.
        * Every partial is provisional. The final ``TranscriptionSegment``
          values yielded after ``end_input`` are authoritative — callers
          should discard speculative partial text once finals arrive.

        Args:
            audio_chunks: Sync iterable or async iterator of raw float32
                PCM byte blocks. Each block is forwarded as one
                ``send_audio`` call.
            model: Model reference. Defaults to the transcription capability.
            language: Optional language hint (BCP 47 code, e.g. ``"en"``).
            sample_rate_hz: Sample rate of the supplied PCM. Defaults to 16 kHz.
            deadline_ms: Overall wall-clock budget for draining finals after
                ``end_input``. Exceeding it stops the drain.

        Yields:
            ``TranscriptionPartial`` while audio is in flight, then
            ``TranscriptionSegment`` finals after ``end_input``.
        """
        ref = model or ModelRefFactory.capability(ModelCapability.TRANSCRIPTION)
        factory = self._stream_session_factory or self._default_stream_session_factory(language)
        session = factory(ref)

        highest_revision = 0
        end_input_calls = 0
        try:
            # Feed phase: push each audio block, draining any provisional
            # partials the runtime emits between sends so callers see them
            # as early as possible.
            async for block in _as_async_iter(audio_chunks):
                if block:
                    session.send_audio(block, sample_rate=sample_rate_hz, channels=1)
                for partial in self._drain_partials(session):
                    if partial.revision_id > highest_revision:
                        highest_revision = partial.revision_id
                        yield partial

            # Finalize input exactly once, then drain to completion: late
            # partials first (still subject to revision supersession), then
            # the authoritative final segments.
            session.end_input()
            end_input_calls += 1

            for event in self._drain_until_completed(session, deadline_ms=deadline_ms):
                if isinstance(event, TranscriptionPartial):
                    if event.revision_id > highest_revision:
                        highest_revision = event.revision_id
                        yield event
                else:
                    # TranscriptionSegment — committed, authoritative.
                    yield event
        finally:
            session.close()
        # ``end_input`` is called exactly once on the normal path. The
        # assertion documents the invariant for readers; the loop above
        # guarantees it structurally.
        assert end_input_calls <= 1

    def _default_stream_session_factory(self, language: Optional[str]) -> StreamSessionFactory:
        """Build the native ``audio.stt.stream`` session factory.

        Warms a :class:`~octomil.runtime.native.stt_backend.NativeSttBackend`
        for the resolved model and opens a streaming session through it, so
        the native runtime + model acquisition stays in the native module.
        Imported lazily so the public ``octomil.audio`` surface stays
        cffi-free unless streaming is actually used.
        """

        def _factory(ref: ModelRef) -> StreamSession:
            from octomil.runtime.native.stt_backend import NativeSttBackend

            model_name = getattr(ref, "model_id", None) or getattr(ref, "name", "") or ""
            backend = NativeSttBackend()
            backend.load_model(str(model_name))
            session = backend.open_stream_session(language=language)
            return _OwnedStreamSession(session, backend)

        return _factory

    @staticmethod
    def _drain_partials(session: StreamSession) -> "list[TranscriptionPartial]":
        """Non-blocking drain of any pending provisional partials.

        Uses ``timeout_ms=0`` so the feed loop never stalls waiting on a
        partial that has not been produced yet.
        """
        from octomil.runtime.native import loader as _L

        out: list[TranscriptionPartial] = []
        while True:
            ev = session.poll_event(timeout_ms=0)
            ev_type = getattr(ev, "type", _L.OCT_EVENT_NONE)
            if ev_type == _L.OCT_EVENT_NONE:
                break
            if ev_type == _L.OCT_EVENT_TRANSCRIPT_PARTIAL:
                out.append(_partial_from_event(ev))
        return out

    @staticmethod
    def _drain_until_completed(
        session: StreamSession,
        *,
        deadline_ms: int,
    ) -> "list[Union[TranscriptionPartial, TranscriptionSegment]]":
        """Drain finals (and any late partials) until SESSION_COMPLETED.

        Returns events in the order the runtime emits them so the caller's
        ordering contract is preserved. Bounded by ``deadline_ms``.
        """
        import time

        from octomil.runtime.native import loader as _L

        out: list[Union[TranscriptionPartial, TranscriptionSegment]] = []
        deadline = time.monotonic() + (deadline_ms / 1000.0)
        while time.monotonic() < deadline:
            ev = session.poll_event(timeout_ms=_STREAM_POLL_TIMEOUT_MS)
            ev_type = getattr(ev, "type", _L.OCT_EVENT_NONE)
            if ev_type == _L.OCT_EVENT_NONE:
                continue
            if ev_type in (_L.OCT_EVENT_SESSION_STARTED, _L.OCT_EVENT_TRANSCRIPT_FINAL):
                continue
            if ev_type == _L.OCT_EVENT_TRANSCRIPT_PARTIAL:
                out.append(_partial_from_event(ev))
                continue
            if ev_type == _L.OCT_EVENT_TRANSCRIPT_SEGMENT:
                out.append(_segment_from_event(ev))
                continue
            if ev_type == _L.OCT_EVENT_SESSION_COMPLETED:
                break
        return out


async def _as_async_iter(
    source: Union[Iterable[bytes], AsyncIterator[bytes]],
) -> AsyncIterator[bytes]:
    """Normalize a sync iterable or async iterator into an async iterator.

    Lets ``stream_transcribe`` accept both a plain ``list[bytes]`` (the
    common batched-PCM case and what tests feed) and a live async source
    (mic capture, socket) without branching at the call site.
    """
    if hasattr(source, "__aiter__"):
        async for item in source:  # type: ignore[union-attr]
            yield item
        return
    for item in source:  # type: ignore[union-attr]
        yield item
        # Cooperative yield so a large pre-buffered list does not starve
        # the event loop / consumer between sends.
        await asyncio.sleep(0)


def _partial_from_event(ev: object) -> TranscriptionPartial:
    """Project a decoded ``OCT_EVENT_TRANSCRIPT_PARTIAL`` event onto the
    public :class:`TranscriptionPartial`. Reads only the ``partial_*`` /
    ``text`` attributes the loader's ``NativeEvent`` exposes (also satisfied
    by test fakes)."""
    return TranscriptionPartial(
        text=getattr(ev, "text", "") or "",
        revision_id=int(getattr(ev, "partial_revision_id", 0)),
        is_stable=bool(getattr(ev, "partial_is_stable", False)),
        start_ms=int(getattr(ev, "partial_start_ms", 0)),
        end_ms=int(getattr(ev, "partial_end_ms", 0)),
        stable_prefix_bytes=int(getattr(ev, "partial_stable_prefix_bytes", 0)),
    )


def _segment_from_event(ev: object) -> TranscriptionSegment:
    """Project a decoded ``OCT_EVENT_TRANSCRIPT_SEGMENT`` event onto the
    public :class:`TranscriptionSegment` (with WU-1 per-segment decode
    diagnostics). These finals are authoritative."""
    return TranscriptionSegment(
        text=getattr(ev, "text", "") or "",
        start_ms=int(getattr(ev, "segment_start_ms", 0)),
        end_ms=int(getattr(ev, "segment_end_ms", 0)),
        avg_logprob=float(getattr(ev, "segment_avg_logprob", 0.0)),
        no_speech_prob=float(getattr(ev, "segment_no_speech_prob", 0.0)),
        source_window_index=int(getattr(ev, "segment_source_window_index", 0)),
        source_window_start_ms=int(getattr(ev, "segment_source_window_start_ms", 0)),
        source_window_end_ms=int(getattr(ev, "segment_source_window_end_ms", 0)),
        partial_revision_start=int(getattr(ev, "segment_partial_revision_start", 0)),
        partial_revision_end=int(getattr(ev, "segment_partial_revision_end", 0)),
        source_kind=int(getattr(ev, "segment_source_kind", 0)),
        vad_active=bool(getattr(ev, "segment_vad_active", False)),
        no_speech_decision=bool(getattr(ev, "segment_no_speech_decision", False)),
    )
