"""TTS streaming observability collector + sinks.

Implements the SDK side of
``octomil-contracts/conformance/tts_observability.yaml``. Six event
shapes:

  - ``tts.stream.started``
  - ``tts.stream.first_audio_chunk``
  - ``tts.stream.completed``
  - ``tts.stream.cancelled``
  - ``tts.stream.failed``
  - ``tts.voice.rejected``

A :class:`TtsMetricsCollector` is automatically attached to every
:class:`octomil.audio.streaming.SpeechStream`. It accumulates per-stream
state from the events the stream is already producing (start time,
chunk count, first-chunk time, total bytes, total samples,
:class:`StreamingMode`) and dispatches typed events to the configured
sinks on stream completion / cancellation / failure / pre-stream
rejection.

Privacy invariants (non-negotiable, per the contract):
the emitted event payloads MUST NOT contain the original ``input``
text, audio bytes, filesystem paths beyond the artifact key, the
user's prompt, or PII beyond ``device_id`` / ``org_id``. The
collector itself never accepts those fields, and the test suite
asserts a known sensitive substring is absent from every event
emitted across every sink.

OpenTelemetry is an optional extra (``pip install octomil[otel]``).
The core SDK keeps working in stripped Python builds (Ren'Py /
PyInstaller without ``opentelemetry``) — same lazy-import posture as
the sqlite3 fix in 4.12.x.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from octomil.audio.streaming import (
        SpeechAudioChunk,
        SpeechStreamCompleted,
        SpeechStreamStarted,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounded enums (from the contract)
# ---------------------------------------------------------------------------


class CancellationSource(str, Enum):
    """Why a stream ended early. Bounded so cancel-rate alerts can
    distinguish user vs infra cancellations."""

    CLIENT_ACLOSE = "client_aclose"
    HTTP_DISCONNECT = "http_disconnect"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    SERVER_SHUTDOWN = "server_shutdown"


class TtsErrorCode(str, Enum):
    """Bounded error_code taxonomy. Free-form error strings are a
    Prometheus cardinality footgun; this enum is the only thing
    that may travel as a metric label."""

    ENGINE_UNAVAILABLE = "tts_engine_unavailable"
    RUNTIME_LOAD_FAILED = "tts_runtime_load_failed"
    SYNTHESIS_FAILED = "tts_synthesis_failed"
    ARTIFACT_MISSING = "tts_artifact_missing"
    ARTIFACT_DIGEST_MISMATCH = "tts_artifact_digest_mismatch"
    CANCELLED = "tts_cancelled"
    CLIENT_DISCONNECT = "tts_client_disconnect"
    DEADLINE_EXCEEDED = "tts_deadline_exceeded"
    UNKNOWN = "tts_unknown_error"


class RejectionReason(str, Enum):
    """Why a pre-stream rejection fired. Distinct from synthesis
    errors so the dashboards can separate "we never started" from
    "synthesis broke mid-stream"."""

    VOICE_NOT_SUPPORTED_FOR_LOCALITY = "voice_not_supported_for_locality"
    FORMAT_UNSUPPORTED = "format_unsupported"
    APP_POLICY_REFUSED = "app_policy_refused"
    MODEL_UNKNOWN = "model_unknown"


# Mapping from the streaming module's TtsStreamingMode enum values
# to the contract's wire vocabulary
# (``octomil-contracts/conformance/tts_observability.yaml::field_semantics.streaming_mode.enum``).
#
# Contract vocabulary (only two values):
#   - ``realtime`` — engine emits PCM as it synthesizes (any
#     progressive cadence: per-sentence chunks, per-frame chunks,
#     or anything finer than a single coalesced final chunk).
#   - ``coalesced_final_chunk`` — engine batched and the SDK wraps
#     the completed buffer as a single trailing chunk.
#
# SDK ``TtsStreamingMode`` is a richer taxonomy than the contract:
#   - ``final_chunk``     → ``coalesced_final_chunk``
#   - ``sentence_chunk``  → ``realtime`` under the observability
#     contract (the consumer gets TTFB benefit on multi-sentence
#     input). NOT progressive in the engine-cadence sense —
#     progressive denotes sub-sentence PCM, which only
#     ``TtsStreamingMode.PROGRESSIVE`` advertises.
#   - ``progressive``     → ``realtime`` (sub-sentence cadence —
#     the engine streams PCM as samples are produced).
#
# Anything not listed here projects to ``coalesced_final_chunk``
# (the safe default — never claim realtime cadence we can't prove).
_STREAMING_MODE_TO_CONTRACT: dict[str, str] = {
    "final_chunk": "coalesced_final_chunk",
    "sentence_chunk": "realtime",
    "progressive": "realtime",
}


# ---------------------------------------------------------------------------
# Event payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TtsStreamMetrics:
    """Privacy-safe rollup metrics for one TTS streaming attempt.

    Field names match the contract YAML; this is the payload every
    sink receives via ``completed`` / ``cancelled`` / ``failed``.
    Fields are deliberately limited to the contract's allowlist —
    no input text, no audio bytes, no fs paths, no prompts.
    """

    model: str
    locality: Optional[str]
    engine: Optional[str]
    voice: Optional[str]
    streaming_mode: Optional[str]  # contract-canonical: realtime | coalesced_final_chunk
    sample_rate: Optional[int]
    sample_format: Optional[str]
    chunk_count: int
    bytes_total: int
    total_samples: int
    audio_duration_ms: int
    latency_ms: float
    first_chunk_ms: Optional[float]
    rtf: Optional[float]
    status: str  # "ok" | "cancelled" | "error"
    error_code: Optional[str] = None
    cancellation_source: Optional[str] = None
    artifact_id: Optional[str] = None
    artifact_version: Optional[str] = None
    engine_version: Optional[str] = None
    policy: Optional[str] = None
    app_slug: Optional[str] = None
    sdk_surface: str = "python"
    sdk_version: Optional[str] = None

    def as_event_dict(self, event: str) -> dict[str, Any]:
        """Project to the contract-shaped event dict (drops ``None``s
        and the rollup-only ``status``/``error_code``/``cancellation_source``
        when they don't apply to the named event)."""
        base: dict[str, Any] = {
            "event": event,
            "model": self.model,
            "sdk_surface": self.sdk_surface,
        }
        for key, val in (
            ("voice", self.voice),
            ("locality", self.locality),
            ("engine", self.engine),
            ("policy", self.policy),
            ("streaming_mode", self.streaming_mode),
            ("sample_rate", self.sample_rate),
            ("sample_format", self.sample_format),
            ("chunk_count", self.chunk_count),
            ("bytes_total", self.bytes_total),
            ("total_samples", self.total_samples),
            ("audio_duration_ms", self.audio_duration_ms),
            ("latency_ms", self.latency_ms),
            ("first_chunk_ms", self.first_chunk_ms),
            ("rtf", self.rtf),
            ("status", self.status),
            ("error_code", self.error_code),
            ("cancellation_source", self.cancellation_source),
            ("artifact_id", self.artifact_id),
            ("artifact_version", self.artifact_version),
            ("engine_version", self.engine_version),
            ("app_slug", self.app_slug),
            ("sdk_version", self.sdk_version),
        ):
            if val is not None:
                base[key] = val
        return base


# ---------------------------------------------------------------------------
# Sink protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TtsMetricsSink(Protocol):
    """Six methods, one per contract event shape. All synchronous —
    sinks that need async work should schedule it themselves."""

    def started(self, metrics: TtsStreamMetrics) -> None: ...
    def first_audio_chunk(self, metrics: TtsStreamMetrics) -> None: ...
    def completed(self, metrics: TtsStreamMetrics) -> None: ...
    def cancelled(self, metrics: TtsStreamMetrics) -> None: ...
    def failed(self, metrics: TtsStreamMetrics) -> None: ...
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
    ) -> None: ...


class _NoOpSink:
    """Default sink. Drops every event."""

    def started(self, metrics: TtsStreamMetrics) -> None:
        pass

    def first_audio_chunk(self, metrics: TtsStreamMetrics) -> None:
        pass

    def completed(self, metrics: TtsStreamMetrics) -> None:
        pass

    def cancelled(self, metrics: TtsStreamMetrics) -> None:
        pass

    def failed(self, metrics: TtsStreamMetrics) -> None:
        pass

    def voice_rejected(self, **kwargs: Any) -> None:
        pass


class CallbackSink:
    """Single-callback sink. The callback receives the event-shaped
    dict (per ``TtsStreamMetrics.as_event_dict``) for every event.
    Useful for embedding apps that want one place to ship metrics
    onward (e.g. into their own logger / metrics platform)."""

    def __init__(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._callback = callback

    def _safe(self, event: str, payload: dict[str, Any]) -> None:
        try:
            self._callback(payload)
        except Exception:
            logger.exception("octomil.audio.metrics.CallbackSink: %s callback raised", event)

    def started(self, metrics: TtsStreamMetrics) -> None:
        self._safe("tts.stream.started", metrics.as_event_dict("tts.stream.started"))

    def first_audio_chunk(self, metrics: TtsStreamMetrics) -> None:
        self._safe(
            "tts.stream.first_audio_chunk",
            metrics.as_event_dict("tts.stream.first_audio_chunk"),
        )

    def completed(self, metrics: TtsStreamMetrics) -> None:
        self._safe("tts.stream.completed", metrics.as_event_dict("tts.stream.completed"))

    def cancelled(self, metrics: TtsStreamMetrics) -> None:
        self._safe("tts.stream.cancelled", metrics.as_event_dict("tts.stream.cancelled"))

    def failed(self, metrics: TtsStreamMetrics) -> None:
        self._safe("tts.stream.failed", metrics.as_event_dict("tts.stream.failed"))

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
        payload: dict[str, Any] = {
            "event": "tts.voice.rejected",
            "model": model,
            "rejection_reason": rejection_reason,
            "sdk_surface": sdk_surface,
        }
        for key, val in (
            ("voice", voice),
            ("locality", locality),
            ("app_slug", app_slug),
            ("policy", policy),
            ("sdk_version", sdk_version),
        ):
            if val is not None:
                payload[key] = val
        self._safe("tts.voice.rejected", payload)


class _TelemetryReporterSink:
    """Routes events through the existing :class:`TelemetryReporter`
    so callers that already opted into telemetry get TTS streaming
    visibility for free. Reuses whichever transport the reporter
    is already configured with (otel, http, in-memory test)."""

    def __init__(self, reporter: Any) -> None:
        self._reporter = reporter

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        emit = getattr(self._reporter, "track_event", None) or getattr(self._reporter, "track", None)
        if emit is None:
            return
        try:
            emit(event, payload)  # type: ignore[misc]
        except Exception:
            logger.exception("TelemetryReporter sink: %s emit failed", event)

    def started(self, metrics: TtsStreamMetrics) -> None:
        self._emit("tts.stream.started", metrics.as_event_dict("tts.stream.started"))

    def first_audio_chunk(self, metrics: TtsStreamMetrics) -> None:
        self._emit(
            "tts.stream.first_audio_chunk",
            metrics.as_event_dict("tts.stream.first_audio_chunk"),
        )

    def completed(self, metrics: TtsStreamMetrics) -> None:
        self._emit("tts.stream.completed", metrics.as_event_dict("tts.stream.completed"))

    def cancelled(self, metrics: TtsStreamMetrics) -> None:
        self._emit("tts.stream.cancelled", metrics.as_event_dict("tts.stream.cancelled"))

    def failed(self, metrics: TtsStreamMetrics) -> None:
        self._emit("tts.stream.failed", metrics.as_event_dict("tts.stream.failed"))

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
        payload: dict[str, Any] = {
            "event": "tts.voice.rejected",
            "model": model,
            "rejection_reason": rejection_reason,
            "sdk_surface": sdk_surface,
        }
        for key, val in (
            ("voice", voice),
            ("locality", locality),
            ("app_slug", app_slug),
            ("policy", policy),
            ("sdk_version", sdk_version),
        ):
            if val is not None:
                payload[key] = val
        self._emit("tts.voice.rejected", payload)


class _FanOutSink:
    """Dispatches every event to multiple underlying sinks. Used to
    chain (callback + telemetry-reporter + otel) without making
    each call site know about all of them."""

    def __init__(self, sinks: list[Any]) -> None:
        self._sinks = list(sinks)

    def _fan(self, method: str, *args: Any, **kwargs: Any) -> None:
        for sink in self._sinks:
            handler = getattr(sink, method, None)
            if handler is None:
                continue
            try:
                handler(*args, **kwargs)
            except Exception:
                logger.exception("octomil.audio.metrics: %s sink raised on %s", sink, method)

    def started(self, metrics: TtsStreamMetrics) -> None:
        self._fan("started", metrics)

    def first_audio_chunk(self, metrics: TtsStreamMetrics) -> None:
        self._fan("first_audio_chunk", metrics)

    def completed(self, metrics: TtsStreamMetrics) -> None:
        self._fan("completed", metrics)

    def cancelled(self, metrics: TtsStreamMetrics) -> None:
        self._fan("cancelled", metrics)

    def failed(self, metrics: TtsStreamMetrics) -> None:
        self._fan("failed", metrics)

    def voice_rejected(self, **kwargs: Any) -> None:
        self._fan("voice_rejected", **kwargs)


# ---------------------------------------------------------------------------
# Optional OpenTelemetry sink
# ---------------------------------------------------------------------------


def OpenTelemetryTtsMetricsSink(*args: Any, **kwargs: Any):  # noqa: N802
    """Lazy entry point for the optional OpenTelemetry sink.

    The actual implementation lives in :mod:`octomil.audio._otel_metrics`
    and is imported on demand so the core SDK keeps working in
    stripped Python builds without ``opentelemetry`` installed.

    Raises :class:`ImportError` with a clear message pointing at
    ``pip install octomil[otel]`` when the extra is not installed.
    """
    try:
        from octomil.audio._otel_metrics import OpenTelemetryTtsMetricsSink as _Impl
    except ImportError as exc:  # pragma: no cover - import-error path
        raise ImportError(
            "OpenTelemetry TTS metrics sink requires the optional ``[otel]`` extra: "
            "``pip install octomil[otel]``. Underlying ImportError: " + repr(exc)
        ) from exc
    return _Impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


@dataclass
class _CollectorState:
    """Per-stream mutable state. Reset for every collector instance."""

    started_at_perf: float = 0.0
    first_chunk_at_perf: Optional[float] = None
    chunk_count: int = 0
    bytes_total: int = 0
    total_samples: int = 0
    audio_duration_ms: int = 0
    streaming_mode_label: Optional[str] = None
    started_emitted: bool = False
    completed_emitted: bool = False
    finalized: bool = False


class TtsMetricsCollector:
    """Per-stream metrics accumulator.

    The :class:`SpeechStream` wrapper feeds every event into the
    collector. The collector tracks the timing/byte/sample state and
    emits via the configured sink at well-defined moments:

      - first ``SpeechStreamStarted`` -> ``started`` (with
        ``streaming_mode`` mapped to the contract enum)
      - first ``SpeechAudioChunk`` after start -> ``first_audio_chunk``
      - ``SpeechStreamCompleted`` -> ``completed`` with rtf
      - explicit ``cancel(source=...)`` call -> ``cancelled``
      - explicit ``fail(error_code=...)`` call -> ``failed``

    ``voice_rejected`` is exposed as a class method so the synchronous
    rejection path can fire it without instantiating a per-stream
    collector.
    """

    def __init__(
        self,
        *,
        sink: TtsMetricsSink,
        model: str,
        voice: Optional[str] = None,
        locality: Optional[str] = None,
        engine: Optional[str] = None,
        policy: Optional[str] = None,
        artifact_id: Optional[str] = None,
        artifact_version: Optional[str] = None,
        engine_version: Optional[str] = None,
        app_slug: Optional[str] = None,
        sdk_surface: str = "python",
        sdk_version: Optional[str] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._sink = sink
        self._model = model
        self._voice = voice
        self._locality = locality
        self._engine = engine
        self._policy = policy
        self._artifact_id = artifact_id
        self._artifact_version = artifact_version
        self._engine_version = engine_version
        self._app_slug = app_slug
        self._sdk_surface = sdk_surface
        self._sdk_version = sdk_version or _resolve_sdk_version()
        self._clock = clock
        self._sample_rate: Optional[int] = None
        self._sample_format: Optional[str] = None
        self._state = _CollectorState()

    # ------------------------------------------------------------------
    # Event ingestion (called from SpeechStream wrapper)
    # ------------------------------------------------------------------

    def on_started(self, event: "SpeechStreamStarted") -> None:
        if self._state.started_emitted:
            return
        self._state.started_at_perf = self._clock()
        self._sample_rate = event.sample_rate
        self._sample_format = event.sample_format
        # Map the SDK enum to the contract-canonical label.
        # ``streaming_capability.mode`` is the canonical field; the
        # v4.13 ``streaming_mode`` shim is gone, so read the new
        # path directly.
        sm = event.streaming_capability.mode.value
        self._state.streaming_mode_label = _STREAMING_MODE_TO_CONTRACT.get(sm, "coalesced_final_chunk")
        # Locality / engine on the started event override the
        # collector's defaults (the kernel knows the resolved values).
        if event.locality:
            self._locality = event.locality
        if event.engine:
            self._engine = event.engine
        self._state.started_emitted = True
        try:
            self._sink.started(self._snapshot(status="ok"))
        except Exception:
            logger.exception("TtsMetricsCollector: started sink raised")

    def on_chunk(self, chunk: "SpeechAudioChunk") -> None:
        if not self._state.started_emitted:
            return
        if self._state.first_chunk_at_perf is None and chunk.data:
            self._state.first_chunk_at_perf = self._clock()
            try:
                self._sink.first_audio_chunk(self._snapshot(status="ok"))
            except Exception:
                logger.exception("TtsMetricsCollector: first_audio_chunk sink raised")
        self._state.chunk_count += 1
        self._state.bytes_total += len(chunk.data) if chunk.data else 0
        self._state.total_samples = max(self._state.total_samples, chunk.sample_index)
        self._state.audio_duration_ms = max(self._state.audio_duration_ms, chunk.timestamp_ms)

    def on_completed(self, event: "SpeechStreamCompleted") -> None:
        if self._state.finalized:
            return
        self._state.finalized = True
        self._state.completed_emitted = True
        # Prefer the engine-reported totals when present.
        if event.total_samples:
            self._state.total_samples = event.total_samples
        if event.duration_ms:
            self._state.audio_duration_ms = event.duration_ms
        if event.e2e_first_chunk_ms is not None:
            self._state.first_chunk_at_perf = self._state.started_at_perf + event.e2e_first_chunk_ms / 1000.0
        # streaming_mode might be revised on completion if the engine
        # adapter only knew the answer post-hoc.
        sm = event.streaming_capability.mode.value
        self._state.streaming_mode_label = _STREAMING_MODE_TO_CONTRACT.get(sm, self._state.streaming_mode_label)
        snapshot = self._snapshot(status="ok")
        # Sample-rate sanity assertion: total_samples / sample_rate * 1000
        # ≈ audio_duration_ms. Mismatches indicate engine bugs that
        # silently destroy RTF accuracy. Use a debug log + assertion
        # in DEBUG builds only.
        if (
            self._sample_rate
            and snapshot.total_samples
            and abs(snapshot.total_samples / max(self._sample_rate, 1) * 1000.0 - snapshot.audio_duration_ms) > 1.0
        ):
            logger.warning(
                "TtsMetricsCollector: sample-rate / duration mismatch "
                "(samples=%d rate=%d duration_ms=%d) — engine reported inconsistent timing",
                snapshot.total_samples,
                self._sample_rate,
                snapshot.audio_duration_ms,
            )
        try:
            self._sink.completed(snapshot)
        except Exception:
            logger.exception("TtsMetricsCollector: completed sink raised")

    def on_cancel(self, source: Union[CancellationSource, str]) -> None:
        if self._state.finalized:
            return
        self._state.finalized = True
        src = source.value if isinstance(source, CancellationSource) else str(source)
        try:
            self._sink.cancelled(self._snapshot(status="cancelled", cancellation_source=src))
        except Exception:
            logger.exception("TtsMetricsCollector: cancelled sink raised")

    def on_fail(
        self,
        *,
        error_code: Union[TtsErrorCode, str] = TtsErrorCode.UNKNOWN,
    ) -> None:
        if self._state.finalized:
            return
        self._state.finalized = True
        code = error_code.value if isinstance(error_code, TtsErrorCode) else str(error_code)
        try:
            self._sink.failed(self._snapshot(status="error", error_code=code))
        except Exception:
            logger.exception("TtsMetricsCollector: failed sink raised")

    # ------------------------------------------------------------------
    # Synchronous rejection (no stream ever started)
    # ------------------------------------------------------------------

    @classmethod
    def emit_voice_rejection(
        cls,
        *,
        sink: TtsMetricsSink,
        model: str,
        voice: Optional[str],
        rejection_reason: Union[RejectionReason, str],
        locality: Optional[str] = None,
        app_slug: Optional[str] = None,
        policy: Optional[str] = None,
        sdk_surface: str = "python",
        sdk_version: Optional[str] = None,
    ) -> None:
        reason = rejection_reason.value if isinstance(rejection_reason, RejectionReason) else str(rejection_reason)
        try:
            sink.voice_rejected(
                model=model,
                voice=voice,
                rejection_reason=reason,
                locality=locality,
                app_slug=app_slug,
                policy=policy,
                sdk_surface=sdk_surface,
                sdk_version=sdk_version or _resolve_sdk_version(),
            )
        except Exception:
            logger.exception("TtsMetricsCollector: voice_rejected sink raised")

    # ------------------------------------------------------------------
    # Internal: build the immutable rollup payload
    # ------------------------------------------------------------------

    def _snapshot(
        self,
        *,
        status: str,
        error_code: Optional[str] = None,
        cancellation_source: Optional[str] = None,
    ) -> TtsStreamMetrics:
        now = self._clock()
        latency_ms = (now - self._state.started_at_perf) * 1000.0 if self._state.started_emitted else 0.0
        first_chunk_ms: Optional[float]
        if self._state.first_chunk_at_perf is not None and self._state.started_emitted:
            first_chunk_ms = (self._state.first_chunk_at_perf - self._state.started_at_perf) * 1000.0
        else:
            first_chunk_ms = None
        rtf: Optional[float]
        if status == "ok" and latency_ms > 0 and self._state.audio_duration_ms > 0:
            rtf = self._state.audio_duration_ms / latency_ms
        else:
            rtf = None
        return TtsStreamMetrics(
            model=self._model,
            locality=self._locality,
            engine=self._engine,
            voice=self._voice,
            streaming_mode=self._state.streaming_mode_label,
            sample_rate=self._sample_rate,
            sample_format=self._sample_format,
            chunk_count=self._state.chunk_count,
            bytes_total=self._state.bytes_total,
            total_samples=self._state.total_samples,
            audio_duration_ms=self._state.audio_duration_ms,
            latency_ms=latency_ms,
            first_chunk_ms=first_chunk_ms,
            rtf=rtf,
            status=status,
            error_code=error_code,
            cancellation_source=cancellation_source,
            artifact_id=self._artifact_id,
            artifact_version=self._artifact_version,
            engine_version=self._engine_version,
            policy=self._policy,
            app_slug=self._app_slug,
            sdk_surface=self._sdk_surface,
            sdk_version=self._sdk_version,
        )


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------


_default_sink: TtsMetricsSink = _NoOpSink()


def configure_default_sink(sink: Optional[TtsMetricsSink]) -> None:
    """Install a process-wide default sink. Pass ``None`` to revert to
    the no-op sink. Per-stream overrides via the ``metrics_callback``
    kwarg always win."""
    global _default_sink
    _default_sink = sink if sink is not None else _NoOpSink()


def get_default_sink() -> TtsMetricsSink:
    return _default_sink


def resolve_sink(
    *,
    metrics_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    sink: Optional[TtsMetricsSink] = None,
) -> TtsMetricsSink:
    """Pick the sink for a single stream:

    - explicit ``sink=`` wins
    - otherwise wrap ``metrics_callback`` in a :class:`CallbackSink`
      and fan out to the configured default sink (so a per-call
      callback augments rather than replaces global telemetry)
    - otherwise use the default sink
    """
    if sink is not None:
        return sink
    default = get_default_sink()
    if metrics_callback is None:
        return default
    cb_sink = CallbackSink(metrics_callback)
    if isinstance(default, _NoOpSink):
        return cb_sink
    return _FanOutSink([cb_sink, default])


def _resolve_sdk_version() -> Optional[str]:
    try:
        from octomil import __version__ as v  # noqa: PLC0415

        return v
    except Exception:  # pragma: no cover - defensive
        return None


__all__ = [
    "CancellationSource",
    "TtsErrorCode",
    "RejectionReason",
    "TtsStreamMetrics",
    "TtsMetricsSink",
    "TtsMetricsCollector",
    "CallbackSink",
    "OpenTelemetryTtsMetricsSink",
    "configure_default_sink",
    "get_default_sink",
    "resolve_sink",
]
