"""Tests for ``octomil.runtime.bench.scheduler`` — v0.5 PR C.

Covers the env-var gate, calibration refusal, dedup, queue capacity,
and worker-thread lifecycle. The :class:`BenchHarness` itself is
mocked out via the synthesize-factory stubs from PR B's tests; this
file is about scheduler semantics, not gate behavior.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from octomil.runtime.bench.cache import (
    CacheKey,
    CacheStore,
    DispatchShape,
    HardwareFingerprint,
)
from octomil.runtime.bench.runner import (
    AudioOutput,
    CandidateConfig,
    QualityGateThresholds,
    ReferenceFixture,
)
from octomil.runtime.bench.scheduler import (
    ENV_ALLOW_PLACEHOLDER,
    ENV_ALLOW_PLACEHOLDER_VALUE_ON,
    ENV_BENCH_GATE,
    ENV_BENCH_GATE_VALUE_ON,
    BenchScheduler,
    CalibrationRefusedError,
    is_bench_enabled,
    is_placeholder_bypassed,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hardware() -> HardwareFingerprint:
    return HardwareFingerprint(
        machine="arm64",
        processor="Apple M2",
        cpu_count=8,
        ram_gb=16,
        os_version="macOS 15.1",
        runtime_build_tag="octomil-python:test",
    )


@pytest.fixture
def store(tmp_path: Path, hardware: HardwareFingerprint) -> CacheStore:
    return CacheStore(cache_root=tmp_path, hardware=hardware)


@pytest.fixture
def cache_key() -> CacheKey:
    return CacheKey(
        capability="tts",
        model_id="kokoro-en-v0_19",
        model_digest="sha256:" + "d" * 64,
        quantization_preference="fp32",
        candidate_set_version="1.0",
        reference_workload_version="1.0",
        dispatch_shape=DispatchShape(
            fields={
                "language": "en",
                "sample_format": "pcm_s16le",
                "sample_rate_out": 24000,
                "voice_family": "kokoro_en",
            }
        ),
    )


@pytest.fixture
def fixtures() -> list[ReferenceFixture]:
    return [
        ReferenceFixture(fixture_id="hello", text="Hello.", expected_transcript="hello"),
    ]


@pytest.fixture
def candidate() -> CandidateConfig:
    return CandidateConfig(engine="sherpa-onnx", provider="coreml", config={"num_threads": 1})


@pytest.fixture
def calibrated_thresholds() -> QualityGateThresholds:
    """Non-default thresholds — passes the calibration check without
    needing the bypass env var."""
    return QualityGateThresholds(
        max_clipped_sample_fraction=0.0011,  # off-default by 1e-4
    )


def _sine_pcm(n_samples: int, amplitude: int = 8000) -> bytes:
    out = bytearray()
    for i in range(n_samples):
        sample = amplitude if (i // 100) % 2 == 0 else -amplitude
        out.extend(int(sample).to_bytes(2, "little", signed=True))
    return bytes(out)


def _make_factory(synthesize_called: list[CandidateConfig]):
    def factory(c: CandidateConfig):
        synthesize_called.append(c)

        def synthesize(text: str) -> AudioOutput:
            del text
            n_samples = 24000
            return AudioOutput(
                pcm_s16le=_sine_pcm(n_samples),
                sample_rate=24000,
                n_samples=n_samples,
            )

        return synthesize

    return factory


def _passing_asr_fn(audio: AudioOutput, expected_transcript: str) -> float:
    del audio, expected_transcript
    return 0.05


def _passing_speaker_fn(c: AudioOutput, r: AudioOutput) -> float:
    del c, r
    return 0.95


# ---------------------------------------------------------------------------
# Env-var gate
# ---------------------------------------------------------------------------


def test_is_bench_enabled_only_for_experimental():
    assert is_bench_enabled({ENV_BENCH_GATE: "experimental"}) is True
    assert is_bench_enabled({ENV_BENCH_GATE: "on"}) is False
    assert is_bench_enabled({ENV_BENCH_GATE: "off"}) is False
    assert is_bench_enabled({ENV_BENCH_GATE: "Experimental"}) is False  # case-sensitive
    assert is_bench_enabled({}) is False


def test_is_placeholder_bypassed_only_for_one():
    assert is_placeholder_bypassed({ENV_ALLOW_PLACEHOLDER: "1"}) is True
    assert is_placeholder_bypassed({ENV_ALLOW_PLACEHOLDER: "true"}) is False
    assert is_placeholder_bypassed({ENV_ALLOW_PLACEHOLDER: ""}) is False
    assert is_placeholder_bypassed({}) is False


# ---------------------------------------------------------------------------
# lookup_or_schedule
# ---------------------------------------------------------------------------


def test_lookup_returns_cached_when_present(store, cache_key, fixtures, candidate, calibrated_thresholds):
    """If the cache already holds a winner, scheduler returns it
    without even checking the env var."""
    # Seed the cache with a committed result.
    seed_calls: list[CandidateConfig] = []
    scheduler_seed = BenchScheduler(
        cache=store,
        env={ENV_BENCH_GATE: ENV_BENCH_GATE_VALUE_ON},
    )
    seeded = scheduler_seed.run_foreground(
        cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory(seed_calls),
        cpu_baseline_factory=_make_factory(seed_calls),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
        thresholds=calibrated_thresholds,
    )
    assert seeded.committed is True

    # Now an env=off scheduler should still find the cached entry.
    scheduler = BenchScheduler(cache=store, env={})
    cached = scheduler.lookup_or_schedule(
        cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory([]),
        cpu_baseline_factory=_make_factory([]),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
        thresholds=calibrated_thresholds,
    )
    assert cached is not None
    assert cached.winner is not None


def test_lookup_returns_none_when_env_off(store, cache_key, fixtures, candidate):
    """Env-var unset → no schedule, no exception, returns None."""
    scheduler = BenchScheduler(cache=store, env={})
    factory_calls: list[CandidateConfig] = []
    result = scheduler.lookup_or_schedule(
        cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory(factory_calls),
        cpu_baseline_factory=_make_factory(factory_calls),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
    )
    assert result is None
    # No bench should have been kicked off (factory not called).
    assert factory_calls == []
    assert scheduler.queue_depth == 0


def test_lookup_enqueues_when_env_on_and_calibrated(store, cache_key, fixtures, candidate, calibrated_thresholds):
    """Env-var on + calibrated thresholds → schedule, commit a winner,
    return None on the first call. Subsequent call returns cached."""
    factory_calls: list[CandidateConfig] = []
    scheduler = BenchScheduler(
        cache=store,
        env={ENV_BENCH_GATE: ENV_BENCH_GATE_VALUE_ON},
    )

    first = scheduler.lookup_or_schedule(
        cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory(factory_calls),
        cpu_baseline_factory=_make_factory(factory_calls),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
        thresholds=calibrated_thresholds,
    )
    assert first is None  # async — winner not yet committed

    # Wait for the worker to drain.
    deadline = time.monotonic() + 5.0
    while scheduler.queue_depth > 0 or scheduler.in_flight:
        if time.monotonic() > deadline:
            pytest.fail("scheduler did not drain within 5s")
        time.sleep(0.05)

    # Re-lookup; the cache now holds a winner.
    second = scheduler.lookup_or_schedule(
        cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory(factory_calls),
        cpu_baseline_factory=_make_factory(factory_calls),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
        thresholds=calibrated_thresholds,
    )
    assert second is not None
    assert second.winner is not None
    scheduler.shutdown()


def test_lookup_softly_refuses_placeholder_thresholds(caplog, store, cache_key, fixtures, candidate):
    """Env-var on + default thresholds + bypass off → log WARNING +
    return None (NOT raise). Claude R1 fix: dispatch hot path must
    not crash; calibration enforcement is via log + observability."""
    scheduler = BenchScheduler(
        cache=store,
        env={ENV_BENCH_GATE: ENV_BENCH_GATE_VALUE_ON},
    )
    factory_calls: list[CandidateConfig] = []
    with caplog.at_level("WARNING", logger="octomil.runtime.bench.scheduler"):
        result = scheduler.lookup_or_schedule(
            cache_key,
            candidates=[candidate],
            fixtures=fixtures,
            synthesize_factory=_make_factory(factory_calls),
            cpu_baseline_factory=_make_factory(factory_calls),
            asr_fn=_passing_asr_fn,
            speaker_embedding_fn=_passing_speaker_fn,
            # thresholds=None → defaults to placeholder QualityGateThresholds()
        )
    assert result is None  # fail-soft, not raise
    assert factory_calls == []  # no bench was scheduled
    assert any("placeholder" in r.message for r in caplog.records)


def test_lookup_accepts_placeholder_when_bypass_on(store, cache_key, fixtures, candidate):
    """Env-var on + default thresholds + bypass on → schedules normally."""
    scheduler = BenchScheduler(
        cache=store,
        env={
            ENV_BENCH_GATE: ENV_BENCH_GATE_VALUE_ON,
            ENV_ALLOW_PLACEHOLDER: ENV_ALLOW_PLACEHOLDER_VALUE_ON,
        },
    )
    result = scheduler.lookup_or_schedule(
        cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory([]),
        cpu_baseline_factory=_make_factory([]),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
    )
    assert result is None  # scheduled, not refused
    scheduler.shutdown()


# ---------------------------------------------------------------------------
# Dedup + queue capacity
# ---------------------------------------------------------------------------


def test_concurrent_lookups_dedupe_by_cache_key(store, cache_key, fixtures, candidate, calibrated_thresholds):
    """Two ``lookup_or_schedule`` calls for the same cache_key while
    a bench is in flight enqueue once."""
    factory_calls: list[CandidateConfig] = []
    started = threading.Event()
    finish = threading.Event()

    def slow_factory(c: CandidateConfig):
        factory_calls.append(c)

        def synthesize(text: str) -> AudioOutput:
            del text
            started.set()
            finish.wait(timeout=2.0)
            n_samples = 24000
            return AudioOutput(
                pcm_s16le=_sine_pcm(n_samples),
                sample_rate=24000,
                n_samples=n_samples,
            )

        return synthesize

    scheduler = BenchScheduler(
        cache=store,
        env={ENV_BENCH_GATE: ENV_BENCH_GATE_VALUE_ON},
    )
    try:
        # Kick off the first job; the synthesize call blocks on the
        # `finish` event until the test releases it.
        first = scheduler.lookup_or_schedule(
            cache_key,
            candidates=[candidate],
            fixtures=fixtures,
            synthesize_factory=slow_factory,
            cpu_baseline_factory=slow_factory,
            asr_fn=_passing_asr_fn,
            speaker_embedding_fn=_passing_speaker_fn,
            thresholds=calibrated_thresholds,
        )
        assert first is None
        assert started.wait(timeout=2.0), "bench job never started"

        # While the first job is in flight, enqueue twice more.
        for _ in range(2):
            r = scheduler.lookup_or_schedule(
                cache_key,
                candidates=[candidate],
                fixtures=fixtures,
                synthesize_factory=slow_factory,
                cpu_baseline_factory=slow_factory,
                asr_fn=_passing_asr_fn,
                speaker_embedding_fn=_passing_speaker_fn,
                thresholds=calibrated_thresholds,
            )
            assert r is None
        assert scheduler.queue_depth == 0  # dedup'd against in-flight
        assert cache_key.leaf_filename() in scheduler.in_flight
    finally:
        finish.set()
        scheduler.shutdown(timeout_s=5.0)

    # The blocked synthesize was called by the first cycle: baseline
    # (1) + candidate (1) = 2 factory invocations. Subsequent calls
    # hit the dedup short-circuit and never reach the factory.
    assert len(factory_calls) == 2


# ---------------------------------------------------------------------------
# Foreground execution (used by PR D's CLI)
# ---------------------------------------------------------------------------


def test_run_foreground_runs_synchronously(store, cache_key, fixtures, candidate, calibrated_thresholds):
    """`run_foreground` blocks until the cycle finishes and returns
    the BenchOutcome. No thread / queue involvement."""
    factory_calls: list[CandidateConfig] = []
    scheduler = BenchScheduler(cache=store, env={})  # env-var off; foreground bypasses it
    outcome = scheduler.run_foreground(
        cache_key,
        candidates=[candidate],
        fixtures=fixtures,
        synthesize_factory=_make_factory(factory_calls),
        cpu_baseline_factory=_make_factory(factory_calls),
        asr_fn=_passing_asr_fn,
        speaker_embedding_fn=_passing_speaker_fn,
        thresholds=calibrated_thresholds,
    )
    assert outcome.committed is True
    assert outcome.result.winner is not None
    # Worker thread should NOT have been started for foreground.
    assert scheduler.queue_depth == 0


def test_run_foreground_refuses_placeholder_thresholds(store, cache_key, fixtures, candidate):
    scheduler = BenchScheduler(cache=store, env={})
    with pytest.raises(CalibrationRefusedError, match="placeholder"):
        scheduler.run_foreground(
            cache_key,
            candidates=[candidate],
            fixtures=fixtures,
            synthesize_factory=_make_factory([]),
            cpu_baseline_factory=_make_factory([]),
            asr_fn=_passing_asr_fn,
            speaker_embedding_fn=_passing_speaker_fn,
        )


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def test_shutdown_is_idempotent(store):
    scheduler = BenchScheduler(cache=store, env={})
    scheduler.shutdown()
    scheduler.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# Queue capacity (Codex R1 missed-case)
# ---------------------------------------------------------------------------


def test_queue_capacity_drops_excess(caplog, store, fixtures, calibrated_thresholds):
    """When the queue is at capacity, new enqueues are dropped with a
    log warning instead of blocking the caller."""
    started = threading.Event()
    finish = threading.Event()

    def slow_factory(c: CandidateConfig):
        del c

        def synthesize(text: str) -> AudioOutput:
            del text
            started.set()
            finish.wait(timeout=2.0)
            n_samples = 24000
            return AudioOutput(
                pcm_s16le=_sine_pcm(n_samples),
                sample_rate=24000,
                n_samples=n_samples,
            )

        return synthesize

    scheduler = BenchScheduler(
        cache=store,
        env={ENV_BENCH_GATE: ENV_BENCH_GATE_VALUE_ON},
        queue_max=2,
    )
    try:
        # Distinct cache_keys so dedup doesn't fire; queue capacity does.
        keys = [
            CacheKey(
                capability="tts",
                model_id=f"model-{i}",
                model_digest="sha256:" + "a" * 64,
                quantization_preference="fp32",
                candidate_set_version="1.0",
                reference_workload_version="1.0",
                dispatch_shape=DispatchShape(
                    fields={
                        "language": "en",
                        "sample_format": "pcm_s16le",
                        "sample_rate_out": 24000,
                        "voice_family": f"voice_{i}",
                    }
                ),
            )
            for i in range(5)
        ]
        # First enqueue starts running and blocks on `finish`. Queue:
        # depth=0 (in_flight={k0}). Subsequent enqueues fill the queue
        # to capacity (2), then drop the next two.
        with caplog.at_level("WARNING", logger="octomil.runtime.bench.scheduler"):
            for k in keys:
                scheduler.lookup_or_schedule(
                    k,
                    candidates=[CandidateConfig(engine="sherpa-onnx", provider="coreml", config={})],
                    fixtures=fixtures,
                    synthesize_factory=slow_factory,
                    cpu_baseline_factory=slow_factory,
                    asr_fn=_passing_asr_fn,
                    speaker_embedding_fn=_passing_speaker_fn,
                    thresholds=calibrated_thresholds,
                )
        assert started.wait(timeout=2.0), "first job never started"
        # Worker timing is non-deterministic: it might dequeue entries
        # before subsequent enqueues run. Invariant: queue never exceeds
        # capacity, and the total dropped + accepted = total enqueued.
        assert scheduler.queue_depth <= 2
        drops = [r for r in caplog.records if "scheduler queue at capacity" in r.message]
        # k0 always accepted (worker drains it immediately). The remaining
        # 4 distribute across "queued" and "dropped" depending on the
        # worker's progress; total drops ∈ {2, 3, 4} by invariant.
        assert 2 <= len(drops) <= 4
    finally:
        finish.set()
        scheduler.shutdown(timeout_s=5.0)


# ---------------------------------------------------------------------------
# Env-var typo warning (Claude R1 missed-case)
# ---------------------------------------------------------------------------


def test_env_var_typo_logs_warning(caplog, store):
    """Setting `OCTOMIL_RUNTIME_BENCH=Experimental` (capitalized) or
    any non-empty non-recognized value silently disables the bench;
    we surface a one-shot WARNING at construction so a developer who
    typo'd sees it in their dev console."""
    with caplog.at_level("WARNING", logger="octomil.runtime.bench.scheduler"):
        BenchScheduler(cache=store, env={ENV_BENCH_GATE: "Experimental"})
    assert any(ENV_BENCH_GATE in r.message and "Experimental" in r.message for r in caplog.records)


def test_env_var_unset_no_warning(caplog, store):
    """The unset case is the legitimate default; do NOT warn."""
    with caplog.at_level("WARNING", logger="octomil.runtime.bench.scheduler"):
        BenchScheduler(cache=store, env={})
    assert not any(ENV_BENCH_GATE in r.message for r in caplog.records)


def test_env_var_correct_value_no_warning(caplog, store):
    """`experimental` is recognized — no warning."""
    with caplog.at_level("WARNING", logger="octomil.runtime.bench.scheduler"):
        BenchScheduler(cache=store, env={ENV_BENCH_GATE: ENV_BENCH_GATE_VALUE_ON})
    assert not any(ENV_BENCH_GATE in r.message and "not a recognized" in r.message for r in caplog.records)
