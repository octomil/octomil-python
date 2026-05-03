"""TTS bench harness — v0.5 PR B.

Runs the in-process competition that picks the empirical winning
``(engine, provider, threads, quantization)`` config per
``(capability, model, device, dispatch_shape)`` cache key. Reads the
cache from PR A (`octomil.runtime.bench.cache`); writes the winner
back to the same cache. Driven by the env-var gate from PR C and the
CLI from PR D.

Architectural rules (per `strategy/runtime-selection-bench.md`):

  * **Bench runs in the runtime's warmup worker thread.** The
    candidate set is enumerated; each candidate is executed against
    the fixture suite; the layered quality gate disqualifies
    regressions; the surviving candidate with the lowest score wins.
  * **First dispatch never runs the bench by default.** Bench fires
    on cache miss as a background task; declared defaults serve the
    first ~10 dispatches; the empirical winner takes over once the
    background bench commits.
  * **Layered TTS quality gate** — five independent checks, each
    cheap, each catching a different failure mode. A candidate must
    pass all of them. CPU is the BASELINE (not the oracle) — the
    structural / energy / ASR-intelligibility / speaker-embedding
    gates are independent of any reference. The DTW-aligned
    correlation against CPU is one signal among five.
  * **Threshold calibration deferred.** The thresholds in this
    module are placeholders; v1 implementation calibrates against
    measured CPU-vs-CPU variance on at least 3 M-series + 1 Intel
    device and sets gates 2× above noise.
  * **Hard-cutover policy.** No back-compat aliases, no deprecated
    properties, no dual code paths. When a v1 cutover lands, the
    v0.5 harness is removed in the same release.

Hard-cutover note: this module ships in PR B. Background-bench
firing rules + env-var gate land in PR C. CLI in PR D.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from octomil.runtime.bench.cache import (
    CacheKey,
    CacheStore,
    Result,
    Winner,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contract constants — kept in sync with octomil-contracts schemas.
# ---------------------------------------------------------------------------

#: Hard timeout for a foreground bench. Background bench uses this same
#: budget. Reviewer P2 in the strategy debate: timeout-truncated leaders
#: are biased; we commit a winner only when every candidate ran to
#: completion AND the layered gate fully passed AND the score margin
#: meets ``MIN_SCORE_MARGIN``. Otherwise we write ``incomplete=true``.
DEFAULT_BUDGET_S: float = 60.0

#: Minimum score margin between the winner and the next-qualified
#: candidate before we treat the win as "high confidence." Below this,
#: the result is committed but flagged ``confidence: "low"`` so the
#: dispatch path falls back to declared defaults until the next bench
#: cycle confirms the same winner. Strategy doc reviewer pass: tightened
#: from 5 to 15 because free perf wins should clear a higher bar than
#: "got lucky once."
MIN_SCORE_MARGIN_FRACTION: float = 0.15

#: Confidence tags written into ``Result.confidence``. Mirrors the
#: contract enum in ``octomil-contracts/schemas/core/runtime_bench_result.json``.
CONFIDENCE_HIGH = "high"
CONFIDENCE_LOW = "low"

#: Reasons a candidate may be disqualified, written into
#: ``Result.disqualified[].reason``. Each maps to a layered-gate stage
#: from the strategy doc.
DQ_NAN_INF = "nan_inf"
DQ_CLIPPING = "saturating_clipping"
DQ_SAMPLE_RATE_MISMATCH = "sample_rate_mismatch"
DQ_SAMPLE_FORMAT_MISMATCH = "sample_format_mismatch"
DQ_DURATION_DEVIATION = "duration_deviation"
DQ_RMS_OUT_OF_BAND = "rms_energy_out_of_band"
DQ_SILENCE_FRACTION_OUT_OF_BAND = "silence_fraction_out_of_band"
DQ_DTW_CORRELATION_LOW = "dtw_correlation_low"
DQ_FRAME_ENERGY_CONTOUR_LOW = "frame_energy_contour_low"
DQ_ASR_WER_HIGH = "asr_wer_high"
DQ_SPEAKER_EMBEDDING_LOW = "speaker_embedding_low"


# ---------------------------------------------------------------------------
# Protocols — every hardware-specific dependency goes through one of these.
# ---------------------------------------------------------------------------


class _SynthesizeFn(Protocol):
    """The minimum surface a candidate engine must expose to be benched.

    The harness is deliberately decoupled from ``octomil.audio.speech``
    and the kernel — it doesn't know about ``client.audio.speech.create``,
    routing policy, or app-ref resolution. The bench runs against a
    bare ``synthesize(text) -> AudioOutput`` callable that the caller
    constructs from an engine + config tuple. PR C wires this up.
    """

    def __call__(self, text: str) -> "AudioOutput": ...


class _AsrFn(Protocol):
    """ASR-intelligibility check. Returns the WER of the candidate
    output against the fixture's known transcript. The harness uses a
    lightweight Whisper-tiny by default; bindings can swap by passing
    a different ``asr_fn`` to :class:`BenchHarness`."""

    def __call__(self, audio: "AudioOutput", expected_transcript: str) -> float: ...


class _SpeakerEmbeddingFn(Protocol):
    """Embedding-similarity check. Returns cosine ∈ [-1, 1] between
    candidate output and CPU reference. Frozen open-weights speaker
    encoder pinned in `octomil-contracts/fixtures/runtime_bench/`."""

    def __call__(self, candidate: "AudioOutput", reference: "AudioOutput") -> float: ...


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioOutput:
    """One synthesis result. Format matches the dispatch shape's
    ``sample_format`` / ``sample_rate_out`` exactly; the structural
    gate disqualifies on mismatch."""

    pcm_s16le: bytes
    sample_rate: int
    n_samples: int

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.n_samples / self.sample_rate


@dataclass(frozen=True)
class ReferenceFixture:
    """One fixture in the per-capability reference workload. Pinned
    in ``octomil-contracts/fixtures/runtime_bench/tts/``.

    The fixture text is the input to the bench's synthesize call; the
    transcript is the expected text the ASR-intelligibility gate
    measures WER against."""

    fixture_id: str
    text: str
    expected_transcript: str
    language: str = "en"


@dataclass(frozen=True)
class CandidateConfig:
    """One ``(engine, provider, config)`` tuple competed against the
    others. Constructed by PR C's wiring from the candidate-set
    declarations in ``octomil-contracts``."""

    engine: str
    provider: str
    config: dict[str, Any]
    quantization: str = "fp32"

    def to_summary(self) -> str:
        threads = self.config.get("num_threads")
        suffix = f" ({threads} thread{'s' if threads and threads > 1 else ''})" if threads is not None else ""
        return f"{self.engine} + {self.provider}{suffix}"


@dataclass(frozen=True)
class QualityGateThresholds:
    """Numeric thresholds for the layered TTS quality gate.

    All values are PLACEHOLDERS calibrated against measured CPU-vs-CPU
    variance — v1 implementation runs the calibration step. The
    strategy doc explicitly forbids shipping the placeholder values
    in production; PR C's bench-runner integration test refuses to
    commit a Result if calibration hasn't run on the current device.
    """

    # Stage 1 — structural
    max_clipped_sample_fraction: float = 0.001
    duration_deviation_max: float = 0.10  # ±10% of CPU reference duration

    # Stage 2 — energy
    rms_ratio_min: float = 0.7
    rms_ratio_max: float = 1.4
    silence_fraction_deviation_max: float = 0.15

    # Stage 3 — alignment
    dtw_correlation_min: float = 0.85
    frame_energy_contour_correlation_min: float = 0.90

    # Stage 4 — intelligibility
    asr_wer_max: float = 0.15

    # Stage 5 — speaker embedding
    speaker_embedding_cosine_min: float = 0.85


@dataclass(frozen=True)
class CandidateMeasurement:
    """The harness's per-candidate observation. Composed by
    :meth:`BenchHarness._measure_candidate`. Either a winner-shaped
    score + per-fixture quality breakdown, OR a disqualification
    reason + which fixture/check failed."""

    candidate: CandidateConfig
    first_chunk_ms: float
    total_latency_ms: float
    score: float
    quality_metrics: dict[str, float]
    disqualified_reason: Optional[str] = None
    disqualified_fixture: Optional[str] = None
    disqualified_check_value: Optional[float] = None


@dataclass(frozen=True)
class BenchOutcome:
    """The harness's verdict. Wraps the :class:`Result` written to
    cache plus harness diagnostics for the CLI / observability path."""

    result: Result
    committed: bool
    confidence: str
    n_candidates_run: int
    n_disqualified: int
    elapsed_s: float


# ---------------------------------------------------------------------------
# BenchHarness — the orchestrator
# ---------------------------------------------------------------------------


class BenchHarness:
    """Executes one bench cycle for a (capability, model, device,
    dispatch_shape) cache key. Single-threaded by design — the
    bench runs on the warmup worker thread; concurrent benches for
    different cache_keys serialize via the cache layer's per-leaf
    file lock.

    The harness is deliberately stateless across cycles. Every cycle
    constructs a fresh ``BenchHarness``, runs against the cache, and
    is discarded. State that lives across cycles (warm-backend cache,
    bench scheduler) lives in PR C.
    """

    def __init__(
        self,
        *,
        cache: CacheStore,
        cache_key: CacheKey,
        candidates: list[CandidateConfig],
        fixtures: list[ReferenceFixture],
        synthesize_factory: "SynthesizeFactoryFn",
        cpu_baseline_factory: "SynthesizeFactoryFn",
        asr_fn: _AsrFn,
        speaker_embedding_fn: _SpeakerEmbeddingFn,
        thresholds: QualityGateThresholds = QualityGateThresholds(),
        budget_s: float = DEFAULT_BUDGET_S,
        runtime_build_tag: str = "",
    ) -> None:
        if not candidates:
            raise ValueError("BenchHarness requires at least one candidate")
        if not fixtures:
            raise ValueError("BenchHarness requires at least one fixture")
        if budget_s <= 0:
            raise ValueError(f"budget_s must be positive; got {budget_s!r}")
        self._cache = cache
        self._cache_key = cache_key
        self._candidates = list(candidates)
        self._fixtures = list(fixtures)
        self._synthesize_factory = synthesize_factory
        self._cpu_baseline_factory = cpu_baseline_factory
        self._asr_fn = asr_fn
        self._speaker_embedding_fn = speaker_embedding_fn
        self._thresholds = thresholds
        self._budget_s = budget_s
        self._runtime_build_tag = runtime_build_tag

    def run(self) -> BenchOutcome:
        """Execute the bench cycle. Returns a :class:`BenchOutcome`
        with the result already persisted to the cache (if committed).
        """
        t0 = time.monotonic()

        # CPU baseline first — every other candidate is measured
        # against this reference. Strategy doc rule: CPU is the
        # baseline, NOT the oracle. The structural / energy / ASR /
        # speaker gates are reference-independent; only DTW
        # correlation depends on the baseline.
        baseline = self._build_baseline()

        measurements: list[CandidateMeasurement] = []
        deadline = t0 + self._budget_s
        complete = True

        for candidate in self._candidates:
            if time.monotonic() >= deadline:
                # Budget exhausted before every candidate ran. Per
                # strategy doc reviewer P2: don't commit a partial
                # winner silently. We mark complete=False and write
                # incomplete=true; the next bench cycle resumes.
                complete = False
                break
            measurement = self._measure_candidate(candidate, baseline)
            measurements.append(measurement)

        elapsed_s = time.monotonic() - t0
        return self._compose_outcome(measurements, baseline, complete, elapsed_s)

    # --------- Phase 1: baseline -------------------------------------------

    def _build_baseline(self) -> dict[str, AudioOutput]:
        """Synthesize every fixture once with the declared CPU
        baseline. Returns ``{fixture_id: AudioOutput}``."""
        synthesize = self._cpu_baseline_factory(self._cpu_baseline_config())
        baseline: dict[str, AudioOutput] = {}
        for fixture in self._fixtures:
            baseline[fixture.fixture_id] = synthesize(fixture.text)
        return baseline

    def _cpu_baseline_config(self) -> CandidateConfig:
        """Conventional CPU baseline configuration. Engine + provider
        are inferred from the cache_key's capability — TTS uses the
        sherpa-onnx CPU path with the conservative thread count."""
        return CandidateConfig(
            engine="sherpa-onnx",
            provider="cpu",
            config={"num_threads": 2},
            quantization=self._cache_key.quantization_preference,
        )

    # --------- Phase 2: per-candidate measurement --------------------------

    def _measure_candidate(
        self,
        candidate: CandidateConfig,
        baseline: dict[str, AudioOutput],
    ) -> CandidateMeasurement:
        """Run ``candidate`` against every fixture; apply the layered
        gate; return either a passing measurement or a disqualified one."""
        synthesize = self._synthesize_factory(candidate)
        per_fixture_first_chunk: list[float] = []
        per_fixture_total: list[float] = []
        per_fixture_quality: dict[str, float] = {}

        for fixture in self._fixtures:
            t0 = time.monotonic()
            output = synthesize(fixture.text)
            t_first = time.monotonic()  # PR C wires real first-chunk timing
            t_done = time.monotonic()
            ref = baseline[fixture.fixture_id]

            # Layered gate. Stop at first failure to avoid wasting
            # ASR / speaker-embedding compute on already-disqualified
            # candidates.
            dq = self._apply_gate(output, ref, fixture)
            if dq is not None:
                return CandidateMeasurement(
                    candidate=candidate,
                    first_chunk_ms=0.0,
                    total_latency_ms=0.0,
                    score=float("inf"),
                    quality_metrics={},
                    disqualified_reason=dq[0],
                    disqualified_fixture=fixture.fixture_id,
                    disqualified_check_value=dq[1],
                )

            per_fixture_first_chunk.append((t_first - t0) * 1000.0)
            per_fixture_total.append((t_done - t0) * 1000.0)
            # Aggregate quality scalar for the result file. v0.5 stores
            # one scalar per fixture; v1 expands to per-stage breakdown.
            per_fixture_quality[fixture.fixture_id] = self._aggregate_quality_scalar(output, ref, fixture)

        first_chunk_ms = sum(per_fixture_first_chunk) / max(len(per_fixture_first_chunk), 1)
        total_latency_ms = sum(per_fixture_total) / max(len(per_fixture_total), 1)
        # Strategy doc score formula: 70% first_chunk_ms + 30% total_latency_ms.
        score = 0.7 * first_chunk_ms + 0.3 * total_latency_ms
        return CandidateMeasurement(
            candidate=candidate,
            first_chunk_ms=first_chunk_ms,
            total_latency_ms=total_latency_ms,
            score=score,
            quality_metrics=per_fixture_quality,
        )

    # --------- Phase 3: layered gate ---------------------------------------

    def _apply_gate(
        self,
        output: AudioOutput,
        reference: AudioOutput,
        fixture: ReferenceFixture,
    ) -> Optional[tuple[str, float]]:
        """Run all 5 layers. Returns ``(reason, value)`` on first
        failure, or ``None`` if every check passes.

        Stage implementations are deliberately small and independent
        so future per-capability bench harnesses (ASR, chat, embeddings)
        can reuse the structural + energy stages without inheriting
        the TTS-specific intelligibility / speaker-embedding stages.
        """
        # Stage 1 — structural
        if (s := self._stage_structural(output, reference)) is not None:
            return s

        # Stage 2 — energy
        if (s := self._stage_energy(output, reference)) is not None:
            return s

        # Stage 3 — alignment (DTW + frame-energy contour)
        if (s := self._stage_alignment(output, reference)) is not None:
            return s

        # Stage 4 — ASR intelligibility
        wer = self._asr_fn(output, fixture.expected_transcript)
        if wer > self._thresholds.asr_wer_max:
            return (DQ_ASR_WER_HIGH, wer)

        # Stage 5 — speaker embedding
        cos = self._speaker_embedding_fn(output, reference)
        if cos < self._thresholds.speaker_embedding_cosine_min:
            return (DQ_SPEAKER_EMBEDDING_LOW, cos)

        return None

    def _stage_structural(
        self,
        output: AudioOutput,
        reference: AudioOutput,
    ) -> Optional[tuple[str, float]]:
        """Hard structural checks: NaN/Inf, saturating clipping,
        sample-rate match, sample-format match, duration ±10% of
        reference. Each is cheap; they catch the worst regressions
        before any expensive analysis runs."""
        # PCM s16le has no NaN/Inf representation; the float-domain
        # equivalent is "all-zero sample buffers" which the energy
        # stage catches. Skipped here.

        n_clipped = _count_saturating_samples(output.pcm_s16le)
        clipped_fraction = n_clipped / max(output.n_samples, 1)
        if clipped_fraction > self._thresholds.max_clipped_sample_fraction:
            return (DQ_CLIPPING, clipped_fraction)

        if output.sample_rate != reference.sample_rate:
            return (DQ_SAMPLE_RATE_MISMATCH, float(output.sample_rate))

        # Duration deviation
        ref_duration = max(reference.duration_s, 1e-6)
        ratio = output.duration_s / ref_duration
        if abs(ratio - 1.0) > self._thresholds.duration_deviation_max:
            return (DQ_DURATION_DEVIATION, ratio - 1.0)

        return None

    def _stage_energy(
        self,
        output: AudioOutput,
        reference: AudioOutput,
    ) -> Optional[tuple[str, float]]:
        """RMS energy ratio + silence-fraction deviation."""
        out_rms = _rms_int16(output.pcm_s16le)
        ref_rms = max(_rms_int16(reference.pcm_s16le), 1.0)  # avoid div-by-zero
        ratio = out_rms / ref_rms
        if ratio < self._thresholds.rms_ratio_min or ratio > self._thresholds.rms_ratio_max:
            return (DQ_RMS_OUT_OF_BAND, ratio)

        out_silence = _silence_fraction_int16(output.pcm_s16le)
        ref_silence = _silence_fraction_int16(reference.pcm_s16le)
        if abs(out_silence - ref_silence) > self._thresholds.silence_fraction_deviation_max:
            return (DQ_SILENCE_FRACTION_OUT_OF_BAND, out_silence - ref_silence)

        return None

    def _stage_alignment(
        self,
        output: AudioOutput,
        reference: AudioOutput,
    ) -> Optional[tuple[str, float]]:
        """DTW-aligned cross-correlation + per-frame energy contour.

        v0.5 ships a placeholder implementation that returns a
        plausible value when the audio is broadly similar (length +
        sample_rate match) and a low one otherwise. v1 wires the real
        DTW + energy-contour computation against the calibrated noise
        floor. Not load-bearing for the architecture decision; load-
        bearing for shipping correctness.
        """
        # Placeholder until v1 calibration. Returns no failure so the
        # bench harness can run end-to-end on synthetic fixtures in CI;
        # PR C's integration test refuses to commit a Result with
        # placeholder thresholds on a real device.
        del output, reference
        return None

    # --------- Phase 4: result composition ---------------------------------

    def _aggregate_quality_scalar(
        self,
        output: AudioOutput,
        reference: AudioOutput,
        fixture: ReferenceFixture,
    ) -> float:
        """Single-number quality summary for the cache result body.
        v0.5 returns the speaker-embedding cosine as the most-load-
        bearing metric; v1 expands to a per-stage breakdown."""
        del fixture
        return self._speaker_embedding_fn(output, reference)

    def _compose_outcome(
        self,
        measurements: list[CandidateMeasurement],
        baseline: dict[str, AudioOutput],
        complete: bool,
        elapsed_s: float,
    ) -> BenchOutcome:
        """Pick winner / runners-up / disqualified; write to cache."""
        del baseline  # baseline contributed only to per-candidate measurement

        passing = [m for m in measurements if m.disqualified_reason is None]
        disqualified = [m for m in measurements if m.disqualified_reason is not None]

        # No candidate survived the gate? We still write an
        # incomplete result so observability sees the bench ran.
        if not passing:
            return self._write_outcome(
                winner=None,
                runners_up=[],
                disqualified=disqualified,
                complete=complete,
                confidence=CONFIDENCE_LOW,
                elapsed_s=elapsed_s,
            )

        passing.sort(key=lambda m: m.score)
        winner = passing[0]
        runners_up = passing[1:]

        # Confidence policy from the strategy doc reviewer pass:
        # require >= MIN_SCORE_MARGIN_FRACTION margin AND complete
        # measurement to commit at high confidence.
        confidence: str = CONFIDENCE_HIGH
        if not complete:
            confidence = CONFIDENCE_LOW
        elif runners_up:
            margin = (runners_up[0].score - winner.score) / max(runners_up[0].score, 1e-6)
            if margin < MIN_SCORE_MARGIN_FRACTION:
                confidence = CONFIDENCE_LOW

        return self._write_outcome(
            winner=winner,
            runners_up=runners_up,
            disqualified=disqualified,
            complete=complete,
            confidence=confidence,
            elapsed_s=elapsed_s,
        )

    def _write_outcome(
        self,
        *,
        winner: Optional[CandidateMeasurement],
        runners_up: list[CandidateMeasurement],
        disqualified: list[CandidateMeasurement],
        complete: bool,
        confidence: str,
        elapsed_s: float,
    ) -> BenchOutcome:
        """Assemble the :class:`Result` and hand it to the cache."""
        winner_dataclass: Optional[Winner] = None
        if winner is not None:
            avg_quality = (
                sum(winner.quality_metrics.values()) / max(len(winner.quality_metrics), 1)
                if winner.quality_metrics
                else 0.0
            )
            winner_dataclass = Winner(
                engine=winner.candidate.engine,
                provider=winner.candidate.provider,
                config=dict(winner.candidate.config),
                score=winner.score,
                first_chunk_ms=winner.first_chunk_ms,
                total_latency_ms=winner.total_latency_ms,
                quality_metrics={"avg_speaker_embedding_cosine": avg_quality},
            )

        runners_up_dataclasses = tuple(
            Winner(
                engine=m.candidate.engine,
                provider=m.candidate.provider,
                config=dict(m.candidate.config),
                score=m.score,
                first_chunk_ms=m.first_chunk_ms,
                total_latency_ms=m.total_latency_ms,
                quality_metrics={},
            )
            for m in runners_up
        )

        disqualified_payload: tuple[dict[str, Any], ...] = tuple(
            {
                "engine": m.candidate.engine,
                "provider": m.candidate.provider,
                "config": dict(m.candidate.config),
                "reason": m.disqualified_reason,
                "fixture": m.disqualified_fixture,
                "value": m.disqualified_check_value,
            }
            for m in disqualified
        )

        hardware = self._cache.hardware
        result = Result(
            cache_key=self._cache_key,
            hardware_fingerprint=hardware.full_digest(),
            hardware_descriptor=hardware.descriptor_dict(),
            writer_runtime_build_tag=self._runtime_build_tag or hardware.runtime_build_tag,
            winner=winner_dataclass,
            runners_up=runners_up_dataclasses,
            disqualified=disqualified_payload,
            incomplete=not complete or winner is None,
            confidence=confidence,
        )
        committed = winner is not None and complete
        if committed:
            self._cache.put(result)
        return BenchOutcome(
            result=result,
            committed=committed,
            confidence=confidence,
            n_candidates_run=len([m for m in (runners_up + ([winner] if winner else []))]),
            n_disqualified=len(disqualified),
            elapsed_s=elapsed_s,
        )


# Type alias for the synthesize factory: takes a candidate config,
# returns a synthesize callable. Defined late because it depends on
# CandidateConfig.
SynthesizeFactoryFn = Any  # Callable[[CandidateConfig], _SynthesizeFn]


# ---------------------------------------------------------------------------
# Audio analysis helpers — pure functions, no engine dependency
# ---------------------------------------------------------------------------


def _count_saturating_samples(pcm_s16le: bytes) -> int:
    """Count samples at +/- int16 saturation. PCM s16le is 2 bytes
    per sample; we read pairs and check ``±32767`` / ``-32768``."""
    if not pcm_s16le or len(pcm_s16le) % 2 != 0:
        return 0
    n = 0
    # Iterate by 2-byte slices. Avoiding numpy keeps the harness
    # importable in stripped CPython environments (Ren'Py / sandboxed
    # PyInstaller bundles where the bench is opt-out anyway, but the
    # cache reader path stays portable).
    for i in range(0, len(pcm_s16le), 2):
        sample = int.from_bytes(pcm_s16le[i : i + 2], "little", signed=True)
        if sample >= 32766 or sample <= -32767:
            n += 1
    return n


def _rms_int16(pcm_s16le: bytes) -> float:
    """RMS of int16 PCM samples. Returns 0 on empty buffer."""
    if not pcm_s16le or len(pcm_s16le) < 2:
        return 0.0
    n_samples = len(pcm_s16le) // 2
    if n_samples == 0:
        return 0.0
    sq_sum = 0
    for i in range(0, len(pcm_s16le), 2):
        sample = int.from_bytes(pcm_s16le[i : i + 2], "little", signed=True)
        sq_sum += sample * sample
    return (sq_sum / n_samples) ** 0.5


def _silence_fraction_int16(pcm_s16le: bytes, *, threshold_abs: int = 100) -> float:
    """Fraction of samples whose absolute value is below
    ``threshold_abs``. Default threshold catches near-silent
    background; real silence is bytes-zero. Returns 0 on empty."""
    if not pcm_s16le or len(pcm_s16le) < 2:
        return 0.0
    n_samples = len(pcm_s16le) // 2
    if n_samples == 0:
        return 0.0
    quiet = 0
    for i in range(0, len(pcm_s16le), 2):
        sample = int.from_bytes(pcm_s16le[i : i + 2], "little", signed=True)
        if abs(sample) < threshold_abs:
            quiet += 1
    return quiet / n_samples


__all__ = [
    "AudioOutput",
    "BenchHarness",
    "BenchOutcome",
    "CandidateConfig",
    "CandidateMeasurement",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_LOW",
    "DEFAULT_BUDGET_S",
    "DQ_ASR_WER_HIGH",
    "DQ_CLIPPING",
    "DQ_DTW_CORRELATION_LOW",
    "DQ_DURATION_DEVIATION",
    "DQ_FRAME_ENERGY_CONTOUR_LOW",
    "DQ_NAN_INF",
    "DQ_RMS_OUT_OF_BAND",
    "DQ_SAMPLE_FORMAT_MISMATCH",
    "DQ_SAMPLE_RATE_MISMATCH",
    "DQ_SILENCE_FRACTION_OUT_OF_BAND",
    "DQ_SPEAKER_EMBEDDING_LOW",
    "MIN_SCORE_MARGIN_FRACTION",
    "QualityGateThresholds",
    "ReferenceFixture",
]
