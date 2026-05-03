"""Background-bench scheduler — v0.5 PR C.

Owns the worker thread that runs the bench harness off the dispatch
hot path. The dispatch path calls :meth:`BenchScheduler.lookup_or_schedule`
on every TTS request; if the cache holds a winner, it returns the
winner; if not, it returns ``None`` AND enqueues a background bench
that writes the empirical winner once it converges.

Architectural rules (per ``strategy/runtime-selection-bench.md`` §5
"PR C scope"):

  * **Default off.** The bench is gated behind
    ``OCTOMIL_RUNTIME_BENCH=experimental``. Any other value (including
    unset) disables scheduling — :meth:`lookup_or_schedule` returns
    ``None`` without touching the worker. v0.5 is opt-in experimental;
    v1 flips the default.
  * **First dispatch never blocks.** The worker runs on a daemon
    thread; ``lookup_or_schedule`` returns immediately. The very
    first dispatch on a cache miss falls through to declared defaults;
    dispatch ~N+ silently picks up the empirical winner once the
    background bench commits.
  * **Per-cache-key dedup.** Concurrent dispatches for the same
    ``CacheKey`` enqueue once; subsequent calls are no-ops while the
    bench is in flight.
  * **Calibration refusal.** Real-device runs MUST use calibrated
    thresholds. The scheduler refuses to enqueue with placeholder
    :class:`QualityGateThresholds` unless the
    ``OCTOMIL_RUNTIME_BENCH_ALLOW_PLACEHOLDER=1`` debug-bypass env
    var is set (used by CI / unit tests). This closes the v0.5 → v1
    graduation hole flagged by both reviewers in the consensus doc:
    a developer can't accidentally ship a `Result` with uncalibrated
    `dtw_correlation_min = 0.85` and have the dispatch path trust it.
  * **Hard cutover.** No back-compat. When v1 takes over, this whole
    module is removed in the same release. No deprecation aliases.

Hard-cutover note: PR D's CLI will call into the same
:class:`BenchScheduler` for foreground (`octomil bench run`) cycles.
Foreground cycles are NOT gated by the env var (the user is asking
for it explicitly via the CLI) but ARE gated by the calibration
refusal.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from octomil.runtime.bench.cache import (
    CacheKey,
    CacheStore,
    Result,
)
from octomil.runtime.bench.runner import (
    BenchHarness,
    BenchOutcome,
    CandidateConfig,
    QualityGateThresholds,
    ReferenceFixture,
    SynthesizeFactoryFn,
    _AsrFn,
    _SpeakerEmbeddingFn,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-var contract — single source of truth for what enables / bypasses.
# ---------------------------------------------------------------------------

#: When set to exactly ``"experimental"``, the scheduler enqueues
#: background benches on cache miss. Any other value (including
#: unset, ``"off"``, ``"on"``, or a typo) disables scheduling.
#: Strict equality is intentional — opt-in must be deliberate.
ENV_BENCH_GATE: str = "OCTOMIL_RUNTIME_BENCH"
ENV_BENCH_GATE_VALUE_ON: str = "experimental"

#: When set to ``"1"``, the scheduler accepts placeholder thresholds.
#: Used by unit tests + CI; production code never sets this. Closing
#: the graduation hole: a developer cannot accidentally ship a real
#: device with uncalibrated thresholds because the scheduler will
#: refuse with a clear error message.
ENV_ALLOW_PLACEHOLDER: str = "OCTOMIL_RUNTIME_BENCH_ALLOW_PLACEHOLDER"
ENV_ALLOW_PLACEHOLDER_VALUE_ON: str = "1"

#: Maximum queue depth. A backlog larger than this means the worker
#: can't keep up with cache misses; we drop newly enqueued cache_keys
#: rather than blocking the caller. v1 will surface this as a metric.
DEFAULT_QUEUE_MAX: int = 64


# ---------------------------------------------------------------------------
# Job records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BenchJob:
    """One enqueued bench cycle. Consumed by the worker thread."""

    cache_key: CacheKey
    candidates: tuple[CandidateConfig, ...]
    fixtures: tuple[ReferenceFixture, ...]
    synthesize_factory: SynthesizeFactoryFn
    cpu_baseline_factory: SynthesizeFactoryFn
    asr_fn: _AsrFn
    speaker_embedding_fn: _SpeakerEmbeddingFn
    thresholds: QualityGateThresholds
    runtime_build_tag: str


@dataclass
class _SchedulerState:
    """Mutable state guarded by ``_lock``.

    ``in_flight`` keys on the cache_key's stable leaf-filename digest
    rather than the dataclass instance — :class:`CacheKey` contains
    a :class:`DispatchShape` whose ``fields`` dict is unhashable, so
    we can't put :class:`CacheKey` in a Python ``set``. The leaf
    filename is the canonical-JSON SHA256 from
    ``octomil-contracts/schemas/core/runtime_bench_cache_key.json``."""

    queue: deque[_BenchJob] = field(default_factory=deque)
    in_flight: set[str] = field(default_factory=set)
    closed: bool = False


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CalibrationRefusedError(RuntimeError):
    """Raised when the scheduler refuses to enqueue a bench because
    the supplied :class:`QualityGateThresholds` are placeholders and
    the caller did not set the debug-bypass env var."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PLACEHOLDER_THRESHOLDS_FROZEN = QualityGateThresholds()


def _is_placeholder_thresholds(thresholds: QualityGateThresholds) -> bool:
    """Equality against the dataclass defaults. v1's calibration step
    constructs a :class:`QualityGateThresholds` with measured values;
    anything matching the literal class defaults is by definition
    uncalibrated."""
    return thresholds == _PLACEHOLDER_THRESHOLDS_FROZEN


def is_bench_enabled(env: Optional[dict[str, str]] = None) -> bool:
    """Public env-var check for the bench gate. Returns True iff
    ``OCTOMIL_RUNTIME_BENCH`` is exactly ``"experimental"``. The
    optional ``env`` arg is used by unit tests to avoid mutating
    process state."""
    source = env if env is not None else os.environ
    return source.get(ENV_BENCH_GATE) == ENV_BENCH_GATE_VALUE_ON


def is_placeholder_bypassed(env: Optional[dict[str, str]] = None) -> bool:
    """Returns True iff the debug-bypass env var
    (``OCTOMIL_RUNTIME_BENCH_ALLOW_PLACEHOLDER=1``) is set. Used by
    tests + CI."""
    source = env if env is not None else os.environ
    return source.get(ENV_ALLOW_PLACEHOLDER) == ENV_ALLOW_PLACEHOLDER_VALUE_ON


# ---------------------------------------------------------------------------
# BenchScheduler
# ---------------------------------------------------------------------------


class BenchScheduler:
    """Thread-safe background bench scheduler.

    Construct one per process (typically alongside the runtime kernel
    init). The scheduler owns a single daemon worker thread; the
    thread spins up lazily on first enqueue. Call :meth:`shutdown`
    to drain + stop the worker (idempotent).
    """

    def __init__(
        self,
        *,
        cache: CacheStore,
        env: Optional[dict[str, str]] = None,
        queue_max: int = DEFAULT_QUEUE_MAX,
    ) -> None:
        self._cache = cache
        self._env = env  # snapshot for tests; None → live os.environ
        self._queue_max = queue_max
        self._state = _SchedulerState()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._worker: Optional[threading.Thread] = None

    # --------- Public API -------------------------------------------------

    def lookup_or_schedule(
        self,
        cache_key: CacheKey,
        *,
        candidates: list[CandidateConfig],
        fixtures: list[ReferenceFixture],
        synthesize_factory: SynthesizeFactoryFn,
        cpu_baseline_factory: SynthesizeFactoryFn,
        asr_fn: _AsrFn,
        speaker_embedding_fn: _SpeakerEmbeddingFn,
        thresholds: Optional[QualityGateThresholds] = None,
        runtime_build_tag: str = "",
    ) -> Optional[Result]:
        """Look up the cache for ``cache_key``. If a committed winner
        exists, return it. Otherwise, if the bench is enabled, enqueue
        a background cycle and return ``None``. Either way, return
        immediately — never blocks on synthesis.

        Calibration refusal: if ``thresholds`` is a placeholder
        :class:`QualityGateThresholds` and the debug-bypass env var
        is unset, raises :class:`CalibrationRefusedError`. The dispatch
        path catches this and falls through to declared defaults
        (logging the refusal once per cache_key)."""
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if not is_bench_enabled(self._env):
            # Default-off path: no cache, no schedule — caller falls
            # through to declared defaults.
            return None

        thresholds = thresholds if thresholds is not None else QualityGateThresholds()
        if _is_placeholder_thresholds(thresholds) and not is_placeholder_bypassed(self._env):
            raise CalibrationRefusedError(
                f"BenchScheduler refuses to enqueue {cache_key.capability} bench for "
                f"model={cache_key.model_id!r} on real hardware: thresholds match the "
                f"v0.5 placeholders. Run the calibration step (v1) or set "
                f"{ENV_ALLOW_PLACEHOLDER}=1 to bypass for tests."
            )

        job = _BenchJob(
            cache_key=cache_key,
            candidates=tuple(candidates),
            fixtures=tuple(fixtures),
            synthesize_factory=synthesize_factory,
            cpu_baseline_factory=cpu_baseline_factory,
            asr_fn=asr_fn,
            speaker_embedding_fn=speaker_embedding_fn,
            thresholds=thresholds,
            runtime_build_tag=runtime_build_tag,
        )
        self._enqueue(job)
        return None

    def run_foreground(
        self,
        cache_key: CacheKey,
        *,
        candidates: list[CandidateConfig],
        fixtures: list[ReferenceFixture],
        synthesize_factory: SynthesizeFactoryFn,
        cpu_baseline_factory: SynthesizeFactoryFn,
        asr_fn: _AsrFn,
        speaker_embedding_fn: _SpeakerEmbeddingFn,
        thresholds: Optional[QualityGateThresholds] = None,
        runtime_build_tag: str = "",
        budget_s: Optional[float] = None,
    ) -> BenchOutcome:
        """Run a bench cycle synchronously in the calling thread.
        Used by PR D's ``octomil bench run`` CLI verb. NOT gated by
        the env var (explicit user request) but IS gated by the
        calibration refusal (same hardware-safety reason)."""
        thresholds = thresholds if thresholds is not None else QualityGateThresholds()
        if _is_placeholder_thresholds(thresholds) and not is_placeholder_bypassed(self._env):
            raise CalibrationRefusedError(
                f"BenchScheduler.run_foreground refuses placeholder thresholds for "
                f"{cache_key.capability} model={cache_key.model_id!r}. Calibrate or "
                f"set {ENV_ALLOW_PLACEHOLDER}=1 to bypass for tests."
            )

        kwargs: dict[str, object] = {
            "cache": self._cache,
            "cache_key": cache_key,
            "candidates": candidates,
            "fixtures": fixtures,
            "synthesize_factory": synthesize_factory,
            "cpu_baseline_factory": cpu_baseline_factory,
            "asr_fn": asr_fn,
            "speaker_embedding_fn": speaker_embedding_fn,
            "thresholds": thresholds,
            "runtime_build_tag": runtime_build_tag,
        }
        if budget_s is not None:
            kwargs["budget_s"] = budget_s
        harness = BenchHarness(**kwargs)  # type: ignore[arg-type]
        return harness.run()

    def shutdown(self, *, timeout_s: Optional[float] = 5.0) -> None:
        """Stop the worker. Idempotent. Drains the in-flight job
        (if any) up to ``timeout_s`` then returns."""
        with self._cv:
            if self._state.closed:
                return
            self._state.closed = True
            self._cv.notify_all()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout_s)
            if worker.is_alive():
                logger.warning("BenchScheduler worker did not exit within %.1fs", timeout_s or 0.0)

    # --------- Inspection (used by tests + CLI) ---------------------------

    @property
    def queue_depth(self) -> int:
        """Current queue depth. Useful for the CLI's `octomil bench
        status` view."""
        with self._lock:
            return len(self._state.queue)

    @property
    def in_flight(self) -> frozenset[str]:
        """Snapshot of cache-key leaf-filename digests currently being
        benched (worker is mid-cycle). Read-only; mutations on the
        underlying set are guarded by the lock. Returns digests, not
        :class:`CacheKey` instances, because :class:`CacheKey` isn't
        hashable (contains a dict-bearing dataclass)."""
        with self._lock:
            return frozenset(self._state.in_flight)

    # --------- Internal ---------------------------------------------------

    def _enqueue(self, job: _BenchJob) -> None:
        """Append ``job`` to the queue if not already in-flight or
        already queued. Starts the worker thread if it's not running."""
        leaf = job.cache_key.leaf_filename()
        with self._cv:
            if self._state.closed:
                logger.debug("scheduler closed; dropping job for %s", leaf)
                return
            if leaf in self._state.in_flight:
                logger.debug("dedup: bench already in-flight for %s", leaf)
                return
            if any(q.cache_key.leaf_filename() == leaf for q in self._state.queue):
                logger.debug("dedup: bench already queued for %s", leaf)
                return
            if len(self._state.queue) >= self._queue_max:
                logger.warning(
                    "scheduler queue at capacity (%d); dropping bench for %s",
                    self._queue_max,
                    leaf,
                )
                return
            self._state.queue.append(job)
            self._cv.notify()
            self._ensure_worker_locked()

    def _ensure_worker_locked(self) -> None:
        """Spawn the daemon worker if it hasn't started yet. Caller
        must hold ``_lock``."""
        if self._worker is not None and self._worker.is_alive():
            return
        worker = threading.Thread(
            target=self._worker_loop,
            name="octomil-bench-scheduler",
            daemon=True,
        )
        worker.start()
        self._worker = worker

    def _worker_loop(self) -> None:
        """Pump jobs off the queue. Exits when the scheduler is
        closed AND the queue is empty."""
        logger.debug("bench scheduler worker started")
        while True:
            with self._cv:
                while not self._state.queue and not self._state.closed:
                    self._cv.wait()
                if self._state.closed and not self._state.queue:
                    break
                job = self._state.queue.popleft()
                leaf = job.cache_key.leaf_filename()
                self._state.in_flight.add(leaf)
            try:
                self._run_job(job)
            except Exception:  # noqa: BLE001 — worker must never propagate
                logger.exception("bench job for %s raised", leaf)
            finally:
                with self._lock:
                    self._state.in_flight.discard(leaf)
        logger.debug("bench scheduler worker exiting")

    def _run_job(self, job: _BenchJob) -> BenchOutcome:
        """Construct + run a :class:`BenchHarness` for one job.
        Worker-thread-only — never call from the dispatch hot path."""
        harness = BenchHarness(
            cache=self._cache,
            cache_key=job.cache_key,
            candidates=list(job.candidates),
            fixtures=list(job.fixtures),
            synthesize_factory=job.synthesize_factory,
            cpu_baseline_factory=job.cpu_baseline_factory,
            asr_fn=job.asr_fn,
            speaker_embedding_fn=job.speaker_embedding_fn,
            thresholds=job.thresholds,
            runtime_build_tag=job.runtime_build_tag,
        )
        outcome = harness.run()
        logger.info(
            "bench cycle for %s: committed=%s confidence=%s elapsed_s=%.2f",
            job.cache_key,
            outcome.committed,
            outcome.confidence,
            outcome.elapsed_s,
        )
        return outcome


__all__ = [
    "BenchScheduler",
    "CalibrationRefusedError",
    "DEFAULT_QUEUE_MAX",
    "ENV_ALLOW_PLACEHOLDER",
    "ENV_ALLOW_PLACEHOLDER_VALUE_ON",
    "ENV_BENCH_GATE",
    "ENV_BENCH_GATE_VALUE_ON",
    "is_bench_enabled",
    "is_placeholder_bypassed",
]
