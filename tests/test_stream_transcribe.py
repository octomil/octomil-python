"""WU-3 — public async ``stream_transcribe`` streaming STT API.

Pure-Python (no dylib / cffi): a fake :class:`StreamSession` emits the
canonical realtime sequence::

    SESSION_STARTED
      -> TRANSCRIPT_PARTIAL (rev 1)
      -> TRANSCRIPT_PARTIAL (rev 2)
      -> TRANSCRIPT_PARTIAL (rev 3)
    end_input()
      -> TRANSCRIPT_SEGMENT (final, authoritative)
      -> TRANSCRIPT_SEGMENT (final, authoritative)
    SESSION_COMPLETED

The tests lock the WU-3 acceptance: ordering preserved, stale partials
superseded by ``revision_id``, final segments authoritative, and
``end_input`` called exactly once.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from octomil.audio.transcriptions import AudioTranscriptions
from octomil.audio.types import TranscriptionPartial, TranscriptionSegment
from octomil.model_ref import ModelRefFactory
from octomil.runtime.native import loader as L

# ---------------------------------------------------------------------------
# Fake event + session (structural match of NativeEvent / NativeSession).
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Minimal stand-in for ``loader.NativeEvent`` — only the attributes
    the streaming loop reads. Defaults mirror the loader's zero-state."""

    def __init__(self, ev_type: int, **fields: Any) -> None:
        self.type = ev_type
        self.text = fields.get("text", "")
        self.partial_revision_id = fields.get("partial_revision_id", 0)
        self.partial_is_stable = fields.get("partial_is_stable", False)
        self.partial_start_ms = fields.get("partial_start_ms", 0)
        self.partial_end_ms = fields.get("partial_end_ms", 0)
        self.partial_stable_prefix_bytes = fields.get("partial_stable_prefix_bytes", 0)
        self.segment_start_ms = fields.get("segment_start_ms", 0)
        self.segment_end_ms = fields.get("segment_end_ms", 0)
        self.segment_avg_logprob = fields.get("segment_avg_logprob", 0.0)
        self.segment_no_speech_prob = fields.get("segment_no_speech_prob", 0.0)
        self.segment_source_window_index = fields.get("segment_source_window_index", 0)
        self.segment_source_window_start_ms = fields.get("segment_source_window_start_ms", 0)
        self.segment_source_window_end_ms = fields.get("segment_source_window_end_ms", 0)
        self.segment_partial_revision_start = fields.get("segment_partial_revision_start", 0)
        self.segment_partial_revision_end = fields.get("segment_partial_revision_end", 0)
        self.segment_source_kind = fields.get("segment_source_kind", L.OCT_TRANSCRIPT_SOURCE_NORMAL)
        self.segment_vad_active = fields.get("segment_vad_active", False)
        self.segment_no_speech_decision = fields.get("segment_no_speech_decision", False)


def _none() -> _FakeEvent:
    return _FakeEvent(L.OCT_EVENT_NONE)


class _FakeStreamSession:
    """Scripted session: ``send_audio`` releases queued pre-``end_input``
    events; ``end_input`` releases the post-finalize events. ``poll_event``
    pops the next ready event (or NONE if the queue is momentarily empty)."""

    def __init__(self, pre: list[_FakeEvent], post: list[_FakeEvent]) -> None:
        self._pre = list(pre)
        self._post = list(post)
        self._ready: list[_FakeEvent] = []
        self.audio_blocks: list[bytes] = []
        self.end_input_calls = 0
        self.closed = False
        self._finalized = False

    def send_audio(self, samples: bytes, *, sample_rate: int, channels: int = 1) -> None:
        self.audio_blocks.append(samples)
        # Release the next pre-finalize event so the feed loop's
        # non-blocking drain can observe it.
        if self._pre:
            self._ready.append(self._pre.pop(0))

    def end_input(self) -> int:
        self.end_input_calls += 1
        if not self._finalized:
            # Flush any remaining pre-events first, then all post-events,
            # preserving runtime emission order.
            self._ready.extend(self._pre)
            self._pre.clear()
            self._ready.extend(self._post)
            self._post.clear()
            self._finalized = True
        return L.OCT_STATUS_OK

    def poll_event(self, timeout_ms: int = 0) -> _FakeEvent:
        if self._ready:
            return self._ready.pop(0)
        return _none()

    def close(self) -> None:
        self.closed = True


def _backend(session: _FakeStreamSession) -> AudioTranscriptions:
    """Build an ``AudioTranscriptions`` whose stream factory returns the
    given fake session. The runtime resolver is never hit on the streaming
    path (factory is injected)."""
    return AudioTranscriptions(
        runtime_resolver=lambda ref: None,
        stream_session_factory=lambda ref: session,
    )


def _scenario_session() -> _FakeStreamSession:
    pre = [
        _FakeEvent(L.OCT_EVENT_SESSION_STARTED),
        _FakeEvent(L.OCT_EVENT_TRANSCRIPT_PARTIAL, text="he", partial_revision_id=1),
        _FakeEvent(L.OCT_EVENT_TRANSCRIPT_PARTIAL, text="hello", partial_revision_id=2),
        _FakeEvent(
            L.OCT_EVENT_TRANSCRIPT_PARTIAL,
            text="hello world",
            partial_revision_id=3,
            partial_is_stable=True,
        ),
    ]
    post = [
        _FakeEvent(
            L.OCT_EVENT_TRANSCRIPT_SEGMENT,
            text="hello world",
            segment_start_ms=0,
            segment_end_ms=1000,
            segment_avg_logprob=-0.2,
            segment_no_speech_prob=0.01,
            segment_source_window_index=2,
            segment_source_window_start_ms=0,
            segment_source_window_end_ms=30000,
            segment_partial_revision_start=1,
            segment_partial_revision_end=3,
            segment_source_kind=L.OCT_TRANSCRIPT_SOURCE_TAIL_RECOVERY,
            segment_vad_active=True,
            segment_no_speech_decision=False,
        ),
        _FakeEvent(
            L.OCT_EVENT_TRANSCRIPT_SEGMENT,
            text="goodbye",
            segment_start_ms=1000,
            segment_end_ms=1800,
        ),
        _FakeEvent(L.OCT_EVENT_TRANSCRIPT_FINAL, text="hello world goodbye"),
        _FakeEvent(L.OCT_EVENT_SESSION_COMPLETED),
    ]
    return _FakeStreamSession(pre, post)


async def _collect(transcriptions: AudioTranscriptions, chunks: list[bytes]) -> list[Any]:
    out: list[Any] = []
    async for ev in transcriptions.stream_transcribe(chunks):
        out.append(ev)
    return out


# Four audio blocks → one per scripted pre-event so each partial surfaces.
_FOUR_BLOCKS = [b"\x00\x00\x00\x00"] * 4


@pytest.mark.asyncio
async def test_ordering_partials_then_finals() -> None:
    """Partials are yielded before finals, in source order, then the
    authoritative segments follow."""
    sess = _scenario_session()
    events = await _collect(_backend(sess), _FOUR_BLOCKS)

    kinds = [type(e).__name__ for e in events]
    # All partials precede all segments.
    last_partial = max(i for i, e in enumerate(events) if isinstance(e, TranscriptionPartial))
    first_seg = min(i for i, e in enumerate(events) if isinstance(e, TranscriptionSegment))
    assert last_partial < first_seg, kinds

    partials = [e for e in events if isinstance(e, TranscriptionPartial)]
    assert [p.text for p in partials] == ["he", "hello", "hello world"]
    assert [p.revision_id for p in partials] == [1, 2, 3]


@pytest.mark.asyncio
async def test_revision_supersession_drops_stale() -> None:
    """A partial whose revision_id does not exceed the highest already
    seen is dropped (never yielded)."""
    pre = [
        _FakeEvent(L.OCT_EVENT_SESSION_STARTED),
        _FakeEvent(L.OCT_EVENT_TRANSCRIPT_PARTIAL, text="a", partial_revision_id=1),
        _FakeEvent(L.OCT_EVENT_TRANSCRIPT_PARTIAL, text="ab", partial_revision_id=3),
        # Stale: rev 2 arrives after rev 3 — must be discarded.
        _FakeEvent(L.OCT_EVENT_TRANSCRIPT_PARTIAL, text="STALE", partial_revision_id=2),
    ]
    post = [
        _FakeEvent(L.OCT_EVENT_TRANSCRIPT_SEGMENT, text="abc"),
        _FakeEvent(L.OCT_EVENT_SESSION_COMPLETED),
    ]
    sess = _FakeStreamSession(pre, post)
    events = await _collect(_backend(sess), _FOUR_BLOCKS)

    partials = [e for e in events if isinstance(e, TranscriptionPartial)]
    assert [p.revision_id for p in partials] == [1, 3]
    assert all(p.text != "STALE" for p in partials)


@pytest.mark.asyncio
async def test_finals_authoritative_and_diagnostics() -> None:
    """Final segments carry the committed text + WU-1 per-segment decode
    diagnostics, distinct from any provisional partial."""
    sess = _scenario_session()
    events = await _collect(_backend(sess), _FOUR_BLOCKS)

    segs = [e for e in events if isinstance(e, TranscriptionSegment)]
    assert [s.text for s in segs] == ["hello world", "goodbye"]
    assert segs[0].start_ms == 0 and segs[0].end_ms == 1000
    assert segs[0].avg_logprob == pytest.approx(-0.2)
    assert segs[0].no_speech_prob == pytest.approx(0.01)
    assert segs[0].source_window_index == 2
    assert segs[0].source_window_start_ms == 0
    assert segs[0].source_window_end_ms == 30000
    assert segs[0].partial_revision_start == 1
    assert segs[0].partial_revision_end == 3
    assert segs[0].source_kind == L.OCT_TRANSCRIPT_SOURCE_TAIL_RECOVERY
    assert segs[0].vad_active is True
    assert segs[0].no_speech_decision is False


@pytest.mark.asyncio
async def test_end_input_called_once_and_session_closed() -> None:
    """``end_input`` is invoked exactly once and the session is always
    closed (the finally path)."""
    sess = _scenario_session()
    await _collect(_backend(sess), _FOUR_BLOCKS)
    assert sess.end_input_calls == 1
    assert sess.closed is True
    assert sess.audio_blocks == _FOUR_BLOCKS


@pytest.mark.asyncio
async def test_session_closed_on_consumer_break() -> None:
    """If the consumer abandons the generator early, the session is still
    closed via the generator's finally on aclose."""
    sess = _scenario_session()
    agen = _backend(sess).stream_transcribe(_FOUR_BLOCKS)
    first = await agen.__anext__()
    assert isinstance(first, TranscriptionPartial)
    await agen.aclose()
    assert sess.closed is True


@pytest.mark.asyncio
async def test_async_audio_source_supported() -> None:
    """``stream_transcribe`` accepts an async iterator of PCM blocks, not
    just a sync list."""

    async def _agen():
        for _ in range(4):
            yield b"\x00\x00\x00\x00"

    sess = _scenario_session()
    events = await _collect_async(_backend(sess), _agen())
    partials = [e for e in events if isinstance(e, TranscriptionPartial)]
    assert [p.revision_id for p in partials] == [1, 2, 3]
    assert sess.end_input_calls == 1


async def _collect_async(transcriptions: AudioTranscriptions, source: Any) -> list[Any]:
    out: list[Any] = []
    async for ev in transcriptions.stream_transcribe(source):
        out.append(ev)
    return out


def test_default_factory_keeps_native_backend_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public factory must keep the NativeSttBackend alive until
    session.close(); otherwise the parent runtime closes under the session."""

    class _LifetimeSession:
        def __init__(self) -> None:
            self.invalidated = False
            self.closed = False

        def send_audio(self, samples: bytes, *, sample_rate: int, channels: int = 1) -> None:
            if self.invalidated:
                raise AssertionError("session invalidated by backend lifetime bug")

        def end_input(self) -> int:
            return L.OCT_STATUS_OK

        def poll_event(self, timeout_ms: int = 0) -> _FakeEvent:
            return _none()

        def close(self) -> None:
            self.closed = True

    class _LifetimeBackend:
        last: "_LifetimeBackend | None" = None

        def __init__(self) -> None:
            self.session = _LifetimeSession()
            self.closed = False
            self.loaded_model = ""
            _LifetimeBackend.last = self

        def load_model(self, model_name: str) -> None:
            self.loaded_model = model_name

        def open_stream_session(self, *, language: str | None = None) -> _LifetimeSession:
            return self.session

        def close(self) -> None:
            self.closed = True
            self.session.invalidated = True

        def __del__(self) -> None:
            self.close()

    import octomil.runtime.native.stt_backend as stt_backend

    monkeypatch.setattr(stt_backend, "NativeSttBackend", _LifetimeBackend)
    tx = AudioTranscriptions(runtime_resolver=lambda ref: None)
    factory = tx._default_stream_session_factory(language="ja")
    session = factory(ModelRefFactory.id("whisper-medium"))
    gc.collect()

    backend = _LifetimeBackend.last
    assert backend is not None
    assert backend.loaded_model == "whisper-medium"
    session.send_audio(b"\x00\x00\x00\x00", sample_rate=16000)
    assert backend.closed is False
    session.close()
    assert backend.closed is True
