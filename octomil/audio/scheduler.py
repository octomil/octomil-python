"""TTS request scheduler.

Decides *which* synthesis call gets the engine *when*. Without this,
every ``client.audio.speech.create(...)`` / ``.stream(...)`` call grabs
the engine immediately and races, which means a low-priority
speculative prefetch already in flight can block a latency-critical
foreground call by hundreds of milliseconds — the engine is single-
threaded under the hood and a sentence-chunk callback does not
cooperatively yield to a higher-priority arrival.

Three priority tiers cover the game / kiosk use case:

  * :attr:`TtsRequestPriority.FOREGROUND` — the line the player is
    actively reading. Latency-critical. Preempts SPECULATIVE.
  * :attr:`TtsRequestPriority.PREFETCH` — the predicted-next line,
    started during foreground silence so it's warm when the player
    advances. Yields to FOREGROUND.
  * :attr:`TtsRequestPriority.SPECULATIVE` — branching choices the
    player *might* take. First to be cancelled when something
    higher-priority arrives.

Bounded concurrency per ``(model, backend)``: a single engine
instance can not usefully run two synthesis calls in parallel — they
trash each other's ONNX session state. The scheduler limits in-
flight count to 1 per backend (configurable) and queues the rest.

Honest metrics: queue-wait time is captured at ``acquire`` /
``release`` boundaries and added to a slot's ``queued_ms`` so the
kernel's ``setup_ms`` attribution can distinguish "queued behind a
foreground call" from "engine cold-start" from "voice validation".
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class TtsRequestPriority(str, Enum):
    """Priority tier for a TTS synthesis call.

    Subclass of ``str`` so callers can still pass the literal values
    (``"foreground"``, ``"prefetch"``, ``"speculative"``) and the
    SDK round-trips them through the enum without breaking older
    code paths.

    Comparison: higher tier means higher priority. The scheduler uses
    ``>`` so a FOREGROUND arrival with a SPECULATIVE in-flight will
    cancel the speculative.
    """

    SPECULATIVE = "speculative"
    PREFETCH = "prefetch"
    FOREGROUND = "foreground"

    @property
    def rank(self) -> int:
        """Higher rank = higher priority. Used for preemption checks."""
        return _PRIORITY_RANK[self]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TtsRequestPriority):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, TtsRequestPriority):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, TtsRequestPriority):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, TtsRequestPriority):
            return NotImplemented
        return self.rank >= other.rank


_PRIORITY_RANK: dict[TtsRequestPriority, int] = {
    TtsRequestPriority.SPECULATIVE: 0,
    TtsRequestPriority.PREFETCH: 1,
    TtsRequestPriority.FOREGROUND: 2,
}


def coerce_priority(value: Any) -> TtsRequestPriority:
    """Accept str | TtsRequestPriority | None and return the enum.

    Default is FOREGROUND so the existing API surface (no priority
    kwarg) keeps the latency-critical behaviour callers already
    expect. Unknown strings raise ``ValueError`` rather than
    silently downranking — a typo on ``"speculitave"`` should not
    silently demote a foreground call to background work.
    """
    if value is None:
        return TtsRequestPriority.FOREGROUND
    if isinstance(value, TtsRequestPriority):
        return value
    if isinstance(value, str):
        # Enum value lookup (case-insensitive convenience).
        try:
            return TtsRequestPriority(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"unknown TtsRequestPriority {value!r}; expected one of "
                f"{', '.join(p.value for p in TtsRequestPriority)}"
            ) from exc
    raise TypeError(f"priority must be str or TtsRequestPriority, got {type(value).__name__}")


@dataclass
class _Slot:
    """A single in-flight synthesis slot held by the scheduler.

    The slot is the unit the scheduler tracks: which priority is
    holding the engine, when it was acquired, and the cancel-handle
    a higher-priority arrival can pull to free the engine early.
    """

    priority: TtsRequestPriority
    acquired_at: float
    cancel: Callable[[], Awaitable[None]] | None = None
    cancelled: bool = False


@dataclass
class TtsSchedulerStats:
    """Snapshot of scheduler counters for telemetry / tests."""

    foreground_queued: int = 0
    prefetch_queued: int = 0
    speculative_queued: int = 0
    speculative_cancellations: int = 0
    prefetch_cancellations: int = 0
    in_flight: int = 0


class TtsScheduler:
    """Per-kernel scheduler. One instance covers all TTS backends.

    Concurrency is bounded *per key* (typically ``(model_id, backend
    identity)``) rather than globally — different models can run in
    parallel because they don't share an ONNX session. Within a key,
    only ``max_concurrency`` slots run simultaneously (default 1,
    matching the single-threaded sherpa-onnx engine).

    Usage from the kernel::

        async with kernel.tts_scheduler.acquire(
            key="kokoro-82m", priority=priority, on_cancel=cancel_fn
        ) as slot:
            ...synthesize...

    Holders MUST set ``slot.cancel`` to a coroutine that aborts
    in-flight work cooperatively when a higher-priority arrival
    preempts. For sherpa-onnx that's the ``aclose()`` on the inner
    stream; for the non-streaming path it's a flag the worker thread
    polls.
    """

    def __init__(self, *, max_concurrency: int = 1) -> None:
        self._max_concurrency = max_concurrency
        # Per-key state: list of in-flight Slot objects + an asyncio
        # condition that the queue waits on. Lazy-created in
        # ``_state_for`` so a busy SDK doesn't allocate state for
        # keys it never uses.
        self._states: dict[str, _KeyState] = {}
        self._stats = TtsSchedulerStats()
        # Outer lock: serialises the structural mutations
        # (preempt + slot insertion) so the preempt-then-acquire
        # sequence can't race a third arrival.
        self._mutex = asyncio.Lock()

    @property
    def stats(self) -> TtsSchedulerStats:
        """Defensive copy — callers can't mutate scheduler state."""
        return TtsSchedulerStats(
            foreground_queued=self._stats.foreground_queued,
            prefetch_queued=self._stats.prefetch_queued,
            speculative_queued=self._stats.speculative_queued,
            speculative_cancellations=self._stats.speculative_cancellations,
            prefetch_cancellations=self._stats.prefetch_cancellations,
            in_flight=sum(len(state.slots) for state in self._states.values()),
        )

    def _state_for(self, key: str) -> "_KeyState":
        state = self._states.get(key)
        if state is None:
            state = _KeyState()
            self._states[key] = state
        return state

    async def acquire(
        self,
        *,
        key: str,
        priority: TtsRequestPriority,
    ) -> "_AcquiredSlot":
        """Wait for an open slot under ``key`` at the given priority.

        Behaviour:

          * If a free slot is available immediately, return it
            without waiting (fast path — no scheduling overhead for
            the lone-foreground-call case).
          * If a lower-priority slot is in flight, request its
            cooperative cancellation and wait for a slot to free up.
            Preemption is *requested*; the holder still owns the
            cancellation timing (one chunk boundary for streaming).
          * Otherwise, queue at this priority. FOREGROUND arrivals
            jump the queue ahead of PREFETCH / SPECULATIVE waiters.

        Returns an :class:`_AcquiredSlot` context manager. Holders
        MUST set ``.cancel`` to a coroutine that aborts the
        in-flight synthesis cleanly; otherwise preemption has no
        teeth and a stuck speculative call will keep blocking
        FOREGROUND arrivals.
        """
        t0 = time.monotonic()
        # Bookkeeping for stats — incremented on entry, decremented on
        # exit so the queued counter reflects current waiters, not a
        # cumulative.
        self._inc_queued(priority)

        try:
            slot = await self._enter_slot(key, priority)
        finally:
            self._dec_queued(priority)

        queued_ms = (time.monotonic() - t0) * 1000.0
        return _AcquiredSlot(scheduler=self, key=key, slot=slot, queued_ms=queued_ms)

    async def _enter_slot(self, key: str, priority: TtsRequestPriority) -> _Slot:
        async with self._mutex:
            state = self._state_for(key)

            # Preempt lower-priority in-flight slots when a higher-
            # priority arrival shows up. Multiple lower-priority
            # in-flights all get a cancel signal — the cooperative
            # cancellation path runs concurrently.
            preempted: list[_Slot] = []
            for in_flight in state.slots:
                if in_flight.priority < priority and in_flight.cancel and not in_flight.cancelled:
                    in_flight.cancelled = True
                    preempted.append(in_flight)
                    if in_flight.priority == TtsRequestPriority.SPECULATIVE:
                        self._stats.speculative_cancellations += 1
                    elif in_flight.priority == TtsRequestPriority.PREFETCH:
                        self._stats.prefetch_cancellations += 1

            # Fire cancellation outside the mutex — the cancel
            # coroutine may do real I/O (sherpa-onnx aclose) and
            # holding the mutex around it would block fresh
            # arrivals.
            cancel_fns = [s.cancel for s in preempted if s.cancel is not None]

        for cancel_fn in cancel_fns:
            try:
                await cancel_fn()
            except Exception:
                logger.debug("scheduler: preemption cancel raised; continuing", exc_info=True)

        # Re-acquire the mutex and either grab a slot or wait.
        # When parking on the condition, we must hold the condition's
        # OWN lock (separate from ``self._mutex``), so the wait path
        # is wrapped in ``async with state.condition`` while the
        # immediate-acquire fast path stays under the mutex.
        while True:
            async with self._mutex:
                state = self._state_for(key)
                if len(state.slots) < self._max_concurrency and self._is_highest_priority_waiter(state, priority):
                    slot = _Slot(priority=priority, acquired_at=time.monotonic())
                    state.slots.append(slot)
                    return slot
                # No slot available — park on the condition. Track
                # this waiter's priority so other waiters can compute
                # ``_is_highest_priority_waiter`` correctly.
                if priority == TtsRequestPriority.FOREGROUND:
                    state.foreground_waiters += 1
                elif priority == TtsRequestPriority.PREFETCH:
                    state.prefetch_waiters += 1
                else:
                    state.speculative_waiters += 1
            try:
                async with state.condition:
                    await state.condition.wait()
            finally:
                async with self._mutex:
                    if priority == TtsRequestPriority.FOREGROUND:
                        state.foreground_waiters = max(0, state.foreground_waiters - 1)
                    elif priority == TtsRequestPriority.PREFETCH:
                        state.prefetch_waiters = max(0, state.prefetch_waiters - 1)
                    else:
                        state.speculative_waiters = max(0, state.speculative_waiters - 1)
            # Loop back and re-check eligibility.

    def _is_highest_priority_waiter(self, state: "_KeyState", priority: TtsRequestPriority) -> bool:
        """Return True iff no other waiter on ``state`` outranks
        ``priority``.

        The scheduler tracks per-priority waiter counts so the
        priority-ordered wakeup is O(1) — no need to maintain a
        sorted heap of waiters across multiple keys.
        """
        if priority == TtsRequestPriority.FOREGROUND:
            return True
        if priority == TtsRequestPriority.PREFETCH:
            return state.foreground_waiters == 0
        # SPECULATIVE
        return state.foreground_waiters == 0 and state.prefetch_waiters == 0

    async def _release(self, key: str, slot: _Slot) -> None:
        async with self._mutex:
            state = self._states.get(key)
            if state is None:
                return
            try:
                state.slots.remove(slot)
            except ValueError:
                pass
            # ``notify_all`` requires the condition's own lock to be
            # held — separate from ``self._mutex`` because
            # ``Condition.wait()`` releases the condition lock while
            # parked. Acquire briefly so waiters can be woken.
            async with state.condition:
                state.condition.notify_all()

    def _inc_queued(self, priority: TtsRequestPriority) -> None:
        if priority == TtsRequestPriority.FOREGROUND:
            self._stats.foreground_queued += 1
        elif priority == TtsRequestPriority.PREFETCH:
            self._stats.prefetch_queued += 1
        else:
            self._stats.speculative_queued += 1

    def _dec_queued(self, priority: TtsRequestPriority) -> None:
        if priority == TtsRequestPriority.FOREGROUND:
            self._stats.foreground_queued = max(0, self._stats.foreground_queued - 1)
        elif priority == TtsRequestPriority.PREFETCH:
            self._stats.prefetch_queued = max(0, self._stats.prefetch_queued - 1)
        else:
            self._stats.speculative_queued = max(0, self._stats.speculative_queued - 1)


@dataclass
class _KeyState:
    """Per-key bookkeeping inside the scheduler."""

    slots: list[_Slot] = field(default_factory=list)
    # Per-priority waiter counters — updated outside the public
    # acquire entry to keep the priority-aware wakeup O(1).
    foreground_waiters: int = 0
    prefetch_waiters: int = 0
    speculative_waiters: int = 0
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class _AcquiredSlot:
    """Context manager returned by :meth:`TtsScheduler.acquire`.

    Stores the queue-wait time so the kernel can attribute it to
    ``setup_ms`` honestly. ``slot.cancel`` is a writable hook the
    holder fills in once it has built the inner stream / worker
    thread; without it preemption has no effect on this in-flight
    request.
    """

    def __init__(
        self,
        *,
        scheduler: TtsScheduler,
        key: str,
        slot: _Slot,
        queued_ms: float,
    ) -> None:
        self._scheduler = scheduler
        self._key = key
        self._slot = slot
        self._queued_ms = queued_ms

    @property
    def queued_ms(self) -> float:
        """Wall-clock time the request spent waiting for a slot."""
        return self._queued_ms

    @property
    def priority(self) -> TtsRequestPriority:
        return self._slot.priority

    @property
    def cancelled(self) -> bool:
        """True iff a higher-priority arrival has requested
        preemption. The synthesis path should poll this between
        chunk boundaries and abort cleanly when set."""
        return self._slot.cancelled

    def set_cancel(self, cancel_fn: Callable[[], Awaitable[None]]) -> None:
        """Register the cancel coroutine the scheduler will call on
        preemption. Idempotent — overwrites any prior hook so the
        holder can swap the hook as the synthesis pipeline progresses
        (e.g. swap the validation-time no-op for a real ``aclose``
        once the inner stream is built)."""
        self._slot.cancel = cancel_fn

    async def __aenter__(self) -> "_AcquiredSlot":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._scheduler._release(self._key, self._slot)


__all__ = [
    "TtsRequestPriority",
    "TtsScheduler",
    "TtsSchedulerStats",
    "coerce_priority",
]
