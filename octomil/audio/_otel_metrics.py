"""Optional OpenTelemetry sink for the TTS observability collector.

This module is only imported when a caller actually constructs
:func:`octomil.audio.metrics.OpenTelemetryTtsMetricsSink`. Keeping the
``opentelemetry`` import behind that lazy entry point preserves the
core SDK's "import octomil" zero-cost contract — same posture as the
sqlite3 and audioop fixes in 4.12.x. Stripped Python builds (Ren'Py /
PyInstaller without ``opentelemetry``) don't pay for what they don't use.

Instruments map 1:1 onto the Prometheus histograms / counters defined
in ``conformance/tts_observability.yaml``. Only contract-bounded enum
values appear as metric labels — ``error_code``, ``cancellation_source``,
``streaming_mode``, ``rejection_reason`` are all bounded by enums in
:mod:`octomil.audio.metrics`. Free-form fields (artifact_id, voice,
sdk_version) are intentionally NOT label-promoted to keep cardinality
finite; they ride on event logs / traces instead.
"""

from __future__ import annotations

from typing import Any, Optional

from opentelemetry import metrics as _otel_metrics

from octomil.audio.metrics import TtsStreamMetrics

_INSTRUMENTATION_NAME = "octomil.audio.tts.streaming"
_INSTRUMENTATION_VERSION = "1"


class OpenTelemetryTtsMetricsSink:
    """Emits the contract's ten Prometheus instruments as OTel meter
    instruments. Works with any OTLP-compatible collector and with the
    Prometheus exporter."""

    def __init__(
        self,
        meter_provider: Optional[Any] = None,
        *,
        meter_name: str = _INSTRUMENTATION_NAME,
    ) -> None:
        if meter_provider is None:
            meter = _otel_metrics.get_meter(meter_name, _INSTRUMENTATION_VERSION)
        else:
            meter = meter_provider.get_meter(meter_name, _INSTRUMENTATION_VERSION)
        self._meter = meter

        self._started_total = meter.create_counter(
            "tts.stream.started",
            description="TTS streams started.",
            unit="1",
        )
        self._completed_total = meter.create_counter(
            "tts.stream.completed",
            description="TTS streams that completed cleanly.",
            unit="1",
        )
        self._cancelled_total = meter.create_counter(
            "tts.stream.cancelled",
            description="TTS streams cancelled (consumer aclose, http disconnect, deadline, server shutdown).",
            unit="1",
        )
        self._failed_total = meter.create_counter(
            "tts.stream.failed",
            description="TTS streams that failed mid-stream with a bounded error_code.",
            unit="1",
        )
        self._voice_rejected_total = meter.create_counter(
            "tts.voice.rejected",
            description="Pre-stream voice/format/policy rejections.",
            unit="1",
        )
        self._first_chunk_ms = meter.create_histogram(
            "tts.stream.first_chunk_ms",
            description="Time from stream start to first audio chunk (TTFB).",
            unit="ms",
        )
        self._latency_ms = meter.create_histogram(
            "tts.stream.latency_ms",
            description="Total wall-clock duration of the stream.",
            unit="ms",
        )
        self._audio_duration_ms = meter.create_histogram(
            "tts.stream.audio_duration_ms",
            description="Total audio duration produced by the stream.",
            unit="ms",
        )
        self._rtf = meter.create_histogram(
            "tts.stream.rtf",
            description="Real-time factor: audio_duration_ms / latency_ms. Higher is better.",
            unit="1",
        )
        self._chunk_count = meter.create_histogram(
            "tts.stream.chunk_count",
            description="Number of audio chunks emitted per stream.",
            unit="1",
        )

    # -----------------------------------------------------------------
    # Label projection
    # -----------------------------------------------------------------

    @staticmethod
    def _stream_labels(metrics: TtsStreamMetrics) -> dict[str, str]:
        """Project a metrics rollup onto the bounded label set. Only
        contract-bounded enums travel as labels; free-form fields stay
        out of the label set so Prometheus cardinality stays finite."""
        labels: dict[str, str] = {
            "model": metrics.model,
            "sdk_surface": metrics.sdk_surface,
        }
        if metrics.locality is not None:
            labels["locality"] = metrics.locality
        if metrics.engine is not None:
            labels["engine"] = metrics.engine
        if metrics.streaming_mode is not None:
            labels["streaming_mode"] = metrics.streaming_mode
        return labels

    # -----------------------------------------------------------------
    # Sink methods (TtsMetricsSink protocol)
    # -----------------------------------------------------------------

    def started(self, metrics: TtsStreamMetrics) -> None:
        self._started_total.add(1, self._stream_labels(metrics))

    def first_audio_chunk(self, metrics: TtsStreamMetrics) -> None:
        if metrics.first_chunk_ms is not None:
            self._first_chunk_ms.record(float(metrics.first_chunk_ms), self._stream_labels(metrics))

    def completed(self, metrics: TtsStreamMetrics) -> None:
        labels = self._stream_labels(metrics)
        self._completed_total.add(1, labels)
        self._latency_ms.record(float(metrics.latency_ms), labels)
        self._audio_duration_ms.record(float(metrics.audio_duration_ms), labels)
        self._chunk_count.record(int(metrics.chunk_count), labels)
        if metrics.rtf is not None:
            self._rtf.record(float(metrics.rtf), labels)
        if metrics.first_chunk_ms is not None:
            # ``completed`` re-records first_chunk_ms so dashboards that
            # filter on ``status="ok"`` can pivot on TTFB without
            # joining the started → completed pair.
            self._first_chunk_ms.record(float(metrics.first_chunk_ms), labels)

    def cancelled(self, metrics: TtsStreamMetrics) -> None:
        labels = self._stream_labels(metrics)
        if metrics.cancellation_source is not None:
            labels["cancellation_source"] = metrics.cancellation_source
        self._cancelled_total.add(1, labels)
        self._latency_ms.record(float(metrics.latency_ms), labels)
        self._chunk_count.record(int(metrics.chunk_count), labels)

    def failed(self, metrics: TtsStreamMetrics) -> None:
        labels = self._stream_labels(metrics)
        if metrics.error_code is not None:
            labels["error_code"] = metrics.error_code
        self._failed_total.add(1, labels)
        self._latency_ms.record(float(metrics.latency_ms), labels)

    def voice_rejected(
        self,
        *,
        model: str,
        voice: Optional[str],
        rejection_reason: str,
        locality: Optional[str] = None,
        app_slug: Optional[str] = None,
        policy: Optional[str] = None,
        sdk_surface: str = "python",
        sdk_version: Optional[str] = None,
    ) -> None:
        labels: dict[str, str] = {
            "model": model,
            "rejection_reason": rejection_reason,
            "sdk_surface": sdk_surface,
        }
        if locality is not None:
            labels["locality"] = locality
        self._voice_rejected_total.add(1, labels)
