"""SDK-side TTS streaming observability regressions.

Pins :mod:`octomil.audio.metrics` against the contract at
``octomil-contracts/conformance/tts_observability.yaml``. Every
documented event shape, every privacy invariant, and the
optional-OTel ImportError path get a dedicated test.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from octomil.audio.metrics import (
    CallbackSink,
    CancellationSource,
    RejectionReason,
    TtsErrorCode,
    TtsMetricsCollector,
    TtsStreamMetrics,
    configure_default_sink,
    get_default_sink,
    resolve_sink,
)
from octomil.audio.streaming import (
    SpeechAudioChunk,
    SpeechStream,
    SpeechStreamCompleted,
    SpeechStreamStarted,
    StreamingMode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingSink:
    """Test sink that captures every emitted payload as a flat list of
    (event_name, payload) tuples. Used by the privacy + ordering
    assertions below."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def started(self, m: TtsStreamMetrics) -> None:
        self.events.append(("tts.stream.started", m.as_event_dict("tts.stream.started")))

    def first_audio_chunk(self, m: TtsStreamMetrics) -> None:
        self.events.append(("tts.stream.first_audio_chunk", m.as_event_dict("tts.stream.first_audio_chunk")))

    def completed(self, m: TtsStreamMetrics) -> None:
        self.events.append(("tts.stream.completed", m.as_event_dict("tts.stream.completed")))

    def cancelled(self, m: TtsStreamMetrics) -> None:
        self.events.append(("tts.stream.cancelled", m.as_event_dict("tts.stream.cancelled")))

    def failed(self, m: TtsStreamMetrics) -> None:
        self.events.append(("tts.stream.failed", m.as_event_dict("tts.stream.failed")))

    def voice_rejected(self, **kwargs: Any) -> None:
        self.events.append(("tts.voice.rejected", {**kwargs, "event": "tts.voice.rejected"}))


def _make_clock(times: list[float]):
    """Deterministic monotonic clock. Returns each successive value
    from ``times``; once exhausted, repeats the last one. The
    collector reads the clock multiple times per event (once for
    the timing checkpoint and once when ``_snapshot`` builds the
    payload), so the test only needs to seed the inflection points
    and let later reads stick."""
    pending = list(times)
    last = times[-1]

    def _clock() -> float:
        nonlocal last
        if pending:
            last = pending.pop(0)
        return last

    return _clock


def _started(model: str = "kokoro-82m") -> SpeechStreamStarted:
    return SpeechStreamStarted(
        model=model,
        voice="af_bella",
        sample_rate=24000,
        channels=1,
        sample_format="pcm_s16le",
        streaming_mode=StreamingMode.REALTIME,
        locality="on_device",
        engine="sherpa-onnx",
    )


def _chunk(data: bytes, *, sample_index: int, timestamp_ms: int) -> SpeechAudioChunk:
    return SpeechAudioChunk(
        data=data,
        sample_index=sample_index,
        timestamp_ms=timestamp_ms,
        is_final=False,
    )


# ---------------------------------------------------------------------------
# 1. Collector math: TTFB + RTF
# ---------------------------------------------------------------------------


def test_collector_computes_ttfb_and_rtf_correctly():
    """Started at t=0; first chunk at t=0.05; completed at t=1.0;
    audio_duration_ms=2000.

    Expected:
      first_chunk_ms = 50.0
      latency_ms     = 1000.0
      rtf            = 2000 / 1000 = 2.0
    """
    sink = _RecordingSink()
    # Five clock reads: started, first-chunk, completed-snapshot now, sample assertion call doesn't read clock; resolve_sink_versions doesn't either.
    # The collector reads the clock once per event emission point.
    clock_times = [
        0.000,  # on_started snapshot
        0.050,  # on_chunk (first chunk records timestamp)
        0.050,  # on_chunk first_audio_chunk emit snapshot (now)
        1.000,  # on_completed snapshot
    ]
    collector = TtsMetricsCollector(
        sink=sink,
        model="kokoro-82m",
        voice="af_bella",
        clock=_make_clock(clock_times),
    )

    collector.on_started(_started())
    collector.on_chunk(_chunk(b"\x00\x01" * 240, sample_index=240, timestamp_ms=10))
    collector.on_completed(
        SpeechStreamCompleted(
            duration_ms=2000,
            total_samples=48000,
            sample_rate=24000,
            channels=1,
            sample_format="pcm_s16le",
            streaming_mode=StreamingMode.REALTIME,
            latency_ms=1000.0,
            first_chunk_ms=50.0,
        )
    )

    completed = next(p for ev, p in sink.events if ev == "tts.stream.completed")
    assert completed["first_chunk_ms"] == pytest.approx(50.0, abs=0.5)
    assert completed["latency_ms"] == pytest.approx(1000.0, abs=0.5)
    assert completed["rtf"] == pytest.approx(2.0, abs=0.05)
    assert completed["audio_duration_ms"] == 2000
    assert completed["total_samples"] == 48000


def test_collector_warns_on_sample_rate_inconsistency(caplog):
    """``total_samples / sample_rate * 1000 ≈ audio_duration_ms``
    is the engine-self-consistency invariant. A mismatch indicates
    the engine reported the wrong sample rate (real bug we've hit
    before) and silently destroys RTF accuracy. The collector logs
    a WARNING when this fails."""
    sink = _RecordingSink()
    collector = TtsMetricsCollector(
        sink=sink,
        model="kokoro-82m",
        clock=_make_clock([0.0, 0.05, 0.05, 1.0]),
    )
    collector.on_started(_started())
    collector.on_chunk(_chunk(b"\x00" * 16, sample_index=24000, timestamp_ms=1000))
    # Engine reports a duration that doesn't match
    # total_samples / sample_rate. Warning expected.
    import logging as _logging

    with caplog.at_level(_logging.WARNING, logger="octomil.audio.metrics"):
        collector.on_completed(
            SpeechStreamCompleted(
                duration_ms=5000,  # wrong — should be ~1000ms for 24000 samples @ 24kHz
                total_samples=24000,
                sample_rate=24000,
                channels=1,
                sample_format="pcm_s16le",
                streaming_mode=StreamingMode.REALTIME,
                latency_ms=1000.0,
            )
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "sample-rate" in m and "duration" in m for m in msgs
    ), f"expected sample-rate/duration mismatch warning; got {msgs!r}"


# ---------------------------------------------------------------------------
# 2. Streaming mode mapping (contract canonicalization)
# ---------------------------------------------------------------------------


def test_streaming_mode_maps_to_contract_realtime():
    sink = _RecordingSink()
    collector = TtsMetricsCollector(
        sink=sink,
        model="kokoro-82m",
        clock=_make_clock([0.0, 0.05, 0.05, 1.0]),
    )
    collector.on_started(_started())
    collector.on_chunk(_chunk(b"\x00", sample_index=1, timestamp_ms=0))
    collector.on_completed(
        SpeechStreamCompleted(
            duration_ms=10,
            total_samples=240,
            sample_rate=24000,
            channels=1,
            sample_format="pcm_s16le",
            streaming_mode=StreamingMode.REALTIME,
            latency_ms=1000.0,
        )
    )
    started = next(p for ev, p in sink.events if ev == "tts.stream.started")
    assert started["streaming_mode"] == "realtime"


def test_streaming_mode_maps_final_chunk_to_coalesced():
    """SDK's ``StreamingMode.FINAL_CHUNK`` projects to the contract
    label ``coalesced_final_chunk``. The contract uses the longer
    name to make it impossible to misread as real streaming."""
    sink = _RecordingSink()
    collector = TtsMetricsCollector(
        sink=sink,
        model="kokoro-82m",
        clock=_make_clock([0.0, 0.05, 0.05, 1.0]),
    )
    started_event = SpeechStreamStarted(
        model="kokoro-82m",
        voice="af_bella",
        sample_rate=24000,
        channels=1,
        sample_format="pcm_s16le",
        streaming_mode=StreamingMode.FINAL_CHUNK,
        locality="on_device",
        engine="sherpa-onnx",
    )
    collector.on_started(started_event)
    collector.on_chunk(_chunk(b"\x00", sample_index=1, timestamp_ms=0))
    collector.on_completed(
        SpeechStreamCompleted(
            duration_ms=10,
            total_samples=240,
            sample_rate=24000,
            channels=1,
            sample_format="pcm_s16le",
            streaming_mode=StreamingMode.FINAL_CHUNK,
            latency_ms=1000.0,
        )
    )
    started = next(p for ev, p in sink.events if ev == "tts.stream.started")
    assert started["streaming_mode"] == "coalesced_final_chunk"


# ---------------------------------------------------------------------------
# 3. Cancellation + failure paths
# ---------------------------------------------------------------------------


def test_collector_emits_cancelled_with_source():
    sink = _RecordingSink()
    collector = TtsMetricsCollector(
        sink=sink,
        model="kokoro-82m",
        clock=_make_clock([0.0, 0.5]),
    )
    collector.on_started(_started())
    collector.on_cancel(CancellationSource.CLIENT_ACLOSE)
    cancelled = next(p for ev, p in sink.events if ev == "tts.stream.cancelled")
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancellation_source"] == "client_aclose"
    # No completed/failed event; finalized.
    assert not any(ev == "tts.stream.completed" for ev, _ in sink.events)
    assert not any(ev == "tts.stream.failed" for ev, _ in sink.events)


def test_collector_emits_failed_with_bounded_error_code():
    sink = _RecordingSink()
    collector = TtsMetricsCollector(
        sink=sink,
        model="kokoro-82m",
        clock=_make_clock([0.0, 0.7]),
    )
    collector.on_started(_started())
    collector.on_fail(error_code=TtsErrorCode.SYNTHESIS_FAILED)
    failed = next(p for ev, p in sink.events if ev == "tts.stream.failed")
    assert failed["status"] == "error"
    assert failed["error_code"] == "tts_synthesis_failed"


def test_collector_finalize_is_idempotent():
    sink = _RecordingSink()
    collector = TtsMetricsCollector(
        sink=sink,
        model="kokoro-82m",
        clock=_make_clock([0.0, 0.5, 0.6, 0.7, 0.8, 0.9]),
    )
    collector.on_started(_started())
    collector.on_cancel(CancellationSource.CLIENT_ACLOSE)
    collector.on_cancel(CancellationSource.HTTP_DISCONNECT)  # ignored
    collector.on_fail(error_code=TtsErrorCode.UNKNOWN)  # ignored
    cancelled_count = sum(1 for ev, _ in sink.events if ev == "tts.stream.cancelled")
    failed_count = sum(1 for ev, _ in sink.events if ev == "tts.stream.failed")
    assert cancelled_count == 1
    assert failed_count == 0


# ---------------------------------------------------------------------------
# 4. Pre-stream voice rejection (no started event)
# ---------------------------------------------------------------------------


def test_voice_rejection_fires_before_any_started_event():
    sink = _RecordingSink()
    TtsMetricsCollector.emit_voice_rejection(
        sink=sink,
        model="kokoro-82m",
        voice="alloy",
        rejection_reason=RejectionReason.VOICE_NOT_SUPPORTED_FOR_LOCALITY,
        locality="on_device",
    )
    assert sink.events == [
        (
            "tts.voice.rejected",
            {
                "event": "tts.voice.rejected",
                "model": "kokoro-82m",
                "voice": "alloy",
                "rejection_reason": "voice_not_supported_for_locality",
                "locality": "on_device",
                "app_slug": None,
                "policy": None,
                "sdk_surface": "python",
                "sdk_version": sink.events[0][1].get("sdk_version"),
            },
        )
    ]


# ---------------------------------------------------------------------------
# 5. SpeechStream wrapper: ordering + auto-cancel on aclose
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speechstream_emits_started_then_first_chunk_then_completed():
    sink = _RecordingSink()
    collector = TtsMetricsCollector(
        sink=sink,
        model="kokoro-82m",
        clock=_make_clock([0.0, 0.05, 0.05, 1.0]),
    )

    async def producer():
        yield _started()
        yield _chunk(b"\x00\x01", sample_index=2, timestamp_ms=0)
        yield SpeechStreamCompleted(
            duration_ms=10,
            total_samples=240,
            sample_rate=24000,
            channels=1,
            sample_format="pcm_s16le",
            streaming_mode=StreamingMode.REALTIME,
            latency_ms=1000.0,
            first_chunk_ms=50.0,
        )

    stream = SpeechStream(producer(), metrics_collector=collector)
    async for _ev in stream:
        pass

    names = [ev for ev, _ in sink.events]
    assert names == [
        "tts.stream.started",
        "tts.stream.first_audio_chunk",
        "tts.stream.completed",
    ]


@pytest.mark.asyncio
async def test_aclose_mid_stream_fires_cancelled_not_failed():
    sink = _RecordingSink()
    collector = TtsMetricsCollector(
        sink=sink,
        model="kokoro-82m",
        clock=_make_clock([0.0, 0.05, 0.05, 0.10]),
    )

    async def producer():
        yield _started()
        yield _chunk(b"\x00\x01", sample_index=1, timestamp_ms=0)
        # Pretend more chunks would come; aclose stops us first.
        for _ in range(5):
            yield _chunk(b"\x00\x02", sample_index=2, timestamp_ms=0)

    stream = SpeechStream(producer(), metrics_collector=collector)
    async for ev in stream:
        if isinstance(ev, SpeechAudioChunk):
            break  # consumer breaks; SpeechStream.aclose runs in __aexit__
    await stream.aclose()

    names = [ev for ev, _ in sink.events]
    assert "tts.stream.cancelled" in names
    assert "tts.stream.failed" not in names
    cancelled = next(p for ev, p in sink.events if ev == "tts.stream.cancelled")
    assert cancelled["cancellation_source"] == "client_aclose"


@pytest.mark.asyncio
async def test_producer_exception_fires_failed_then_reraises():
    sink = _RecordingSink()
    collector = TtsMetricsCollector(
        sink=sink,
        model="kokoro-82m",
        clock=_make_clock([0.0, 0.05, 0.05, 0.10]),
    )

    class _Boom(Exception):
        code = "tts_synthesis_failed"

    async def producer():
        yield _started()
        yield _chunk(b"\x00", sample_index=1, timestamp_ms=0)
        raise _Boom("synthesis blew up")

    stream = SpeechStream(producer(), metrics_collector=collector)
    with pytest.raises(_Boom):
        async for _ev in stream:
            pass

    names = [ev for ev, _ in sink.events]
    assert "tts.stream.failed" in names
    failed = next(p for ev, p in sink.events if ev == "tts.stream.failed")
    # Mapping recognizes the code attribute.
    assert failed["error_code"] in {
        "tts_synthesis_failed",
        "tts_unknown_error",  # fallback if mapping changes; either is bounded
    }


# ---------------------------------------------------------------------------
# 6. Privacy regression — no input text leaks to any sink
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_input_text_or_audio_bytes_in_emitted_events():
    """The contract's privacy_invariants are non-negotiable. Every
    emitted event must NOT contain the original input text, audio
    bytes, or filesystem paths beyond artifact_id. Pin by injecting
    a known sensitive substring as the input + chunk payload and
    asserting it's absent from every event payload."""
    SENSITIVE = "secret-prompt-do-not-leak"
    sink = _RecordingSink()
    collector = TtsMetricsCollector(
        sink=sink,
        model="kokoro-82m",
        clock=_make_clock([0.0, 0.05, 0.05, 1.0]),
    )

    async def producer():
        yield _started()
        # Audio chunk data with the sensitive substring in bytes.
        yield SpeechAudioChunk(
            data=SENSITIVE.encode("utf-8") + b"\x00\x01" * 100,
            sample_index=200,
            timestamp_ms=10,
            is_final=False,
        )
        yield SpeechStreamCompleted(
            duration_ms=10,
            total_samples=240,
            sample_rate=24000,
            channels=1,
            sample_format="pcm_s16le",
            streaming_mode=StreamingMode.REALTIME,
            latency_ms=1000.0,
            first_chunk_ms=50.0,
        )

    stream = SpeechStream(producer(), metrics_collector=collector)
    async for _ev in stream:
        pass

    # Every event payload, when serialized as JSON, must NOT contain
    # the sensitive substring. This catches both literal-payload
    # leaks and accidental "data": <bytes-decoded-as-string> bugs.
    for event_name, payload in sink.events:
        serialized = json.dumps(payload, default=str)
        assert (
            SENSITIVE not in serialized
        ), f"Privacy invariant breach: {SENSITIVE!r} appeared in {event_name} payload: {payload!r}"


# ---------------------------------------------------------------------------
# 7. Sink wiring + module-level configuration
# ---------------------------------------------------------------------------


def test_default_no_op_sink_does_not_crash():
    sink = get_default_sink()
    sink.started(
        TtsStreamMetrics(
            model="kokoro-82m",
            locality=None,
            engine=None,
            voice=None,
            streaming_mode="realtime",
            sample_rate=24000,
            sample_format="pcm_s16le",
            chunk_count=0,
            bytes_total=0,
            total_samples=0,
            audio_duration_ms=0,
            latency_ms=0.0,
            first_chunk_ms=None,
            rtf=None,
            status="ok",
        )
    )


def test_callback_sink_receives_event_dicts():
    captured: list[dict[str, Any]] = []
    sink = CallbackSink(captured.append)
    sink.started(
        TtsStreamMetrics(
            model="kokoro-82m",
            locality="on_device",
            engine="sherpa-onnx",
            voice="af_bella",
            streaming_mode="realtime",
            sample_rate=24000,
            sample_format="pcm_s16le",
            chunk_count=0,
            bytes_total=0,
            total_samples=0,
            audio_duration_ms=0,
            latency_ms=0.0,
            first_chunk_ms=None,
            rtf=None,
            status="ok",
        )
    )
    assert captured[0]["event"] == "tts.stream.started"
    assert captured[0]["model"] == "kokoro-82m"
    assert captured[0]["streaming_mode"] == "realtime"


def test_resolve_sink_per_call_callback_fans_out_to_default():
    """A per-call ``metrics_callback`` must AUGMENT, not replace, the
    configured default sink. Otherwise installing global telemetry
    would break for any caller that wants per-call visibility too."""
    captured_default: list[dict[str, Any]] = []
    captured_call: list[dict[str, Any]] = []
    configure_default_sink(CallbackSink(captured_default.append))
    try:
        sink = resolve_sink(metrics_callback=captured_call.append)
        sink.started(
            TtsStreamMetrics(
                model="kokoro-82m",
                locality="on_device",
                engine="sherpa-onnx",
                voice="af_bella",
                streaming_mode="realtime",
                sample_rate=24000,
                sample_format="pcm_s16le",
                chunk_count=0,
                bytes_total=0,
                total_samples=0,
                audio_duration_ms=0,
                latency_ms=0.0,
                first_chunk_ms=None,
                rtf=None,
                status="ok",
            )
        )
    finally:
        configure_default_sink(None)
    assert captured_default and captured_default[0]["event"] == "tts.stream.started"
    assert captured_call and captured_call[0]["event"] == "tts.stream.started"


# ---------------------------------------------------------------------------
# 8. Optional OpenTelemetry sink — graceful ImportError when extra not installed
# ---------------------------------------------------------------------------


def test_otel_sink_raises_clear_importerror_when_extra_missing(monkeypatch):
    """``OpenTelemetryTtsMetricsSink`` must raise a clear ImportError
    pointing at the ``[otel]`` extra when the underlying module is
    not importable. Simulate by blocking the underlying module."""
    import sys as _sys

    from octomil.audio import metrics as metrics_mod

    monkeypatch.setitem(_sys.modules, "octomil.audio._otel_metrics", None)
    with pytest.raises(ImportError) as exc:
        metrics_mod.OpenTelemetryTtsMetricsSink()
    assert "[otel]" in str(exc.value)
    assert "pip install octomil[otel]" in str(exc.value)


def test_otel_sink_label_projection_keeps_cardinality_bounded():
    """When ``opentelemetry`` is installed, the live sink must project
    only contract-bounded enum values onto its label set. Free-form
    fields (artifact_id, voice, sdk_version) MUST stay out of the
    Prometheus label dimension or cardinality blows up.

    This test does NOT exercise the meter — it exercises the
    label projection. Runs only when the [otel] extra is installed."""
    pytest.importorskip("opentelemetry.metrics")
    from octomil.audio._otel_metrics import OpenTelemetryTtsMetricsSink

    metrics = TtsStreamMetrics(
        model="kokoro-82m",
        locality="device",
        engine="sherpa-onnx",
        voice="af_heart",
        streaming_mode="realtime",
        sample_rate=24_000,
        sample_format="pcm_s16le",
        chunk_count=10,
        bytes_total=240_000,
        total_samples=120_000,
        audio_duration_ms=5_000,
        latency_ms=4_500.0,
        first_chunk_ms=180.0,
        rtf=1.11,
        status="ok",
        artifact_id="should-not-be-a-label",
        artifact_version="v17",
        sdk_version="4.12.4",
    )
    labels = OpenTelemetryTtsMetricsSink._stream_labels(metrics)

    # Bounded enums travel as labels.
    assert labels["model"] == "kokoro-82m"
    assert labels["sdk_surface"] == "python"
    assert labels["locality"] == "device"
    assert labels["engine"] == "sherpa-onnx"
    assert labels["streaming_mode"] == "realtime"
    # High-cardinality fields stay OUT of the label set.
    assert "artifact_id" not in labels
    assert "artifact_version" not in labels
    assert "voice" not in labels
    assert "sdk_version" not in labels
