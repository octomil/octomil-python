"""Unit tests for the TTS scheduler.

Pins the priority contract:
  * Default priority is FOREGROUND (back-compat — no kwarg = current behaviour).
  * Enum is comparable / orderable so preemption checks are typed.
  * coerce_priority accepts str / enum / None and rejects unknown strings.
  * Bounded concurrency = 1 per key by default.
  * FOREGROUND arrival cancels SPECULATIVE in flight via the
    cooperative cancel hook.
  * Higher-priority waiters jump the queue when a slot frees.
  * Stats counters reflect queued waiters and cancellations.
  * Idle-engine FOREGROUND call pays zero queue-wait time.
"""

from __future__ import annotations

import asyncio

import pytest

from octomil.audio.scheduler import (
    TtsRequestPriority,
    TtsScheduler,
    coerce_priority,
)

# ---------------------------------------------------------------------------
# Enum contract
# ---------------------------------------------------------------------------


def test_priority_is_str_enum_round_trips_through_strings():
    """Subclass of ``str`` so legacy callers passing the literal value
    keep working."""
    assert TtsRequestPriority.FOREGROUND == "foreground"
    assert TtsRequestPriority("foreground") is TtsRequestPriority.FOREGROUND
    assert TtsRequestPriority("prefetch") is TtsRequestPriority.PREFETCH
    assert TtsRequestPriority("speculative") is TtsRequestPriority.SPECULATIVE


def test_priority_ordering_supports_preemption_check():
    """``new.priority > running.priority`` is the preemption test the
    scheduler uses — typed comparison, not string magic."""
    assert TtsRequestPriority.FOREGROUND > TtsRequestPriority.PREFETCH
    assert TtsRequestPriority.PREFETCH > TtsRequestPriority.SPECULATIVE
    assert TtsRequestPriority.FOREGROUND > TtsRequestPriority.SPECULATIVE
    # And the reverse fails so the scheduler doesn't accidentally
    # demote a foreground call.
    assert not (TtsRequestPriority.SPECULATIVE > TtsRequestPriority.FOREGROUND)


def test_priority_rank_matches_documented_order():
    assert TtsRequestPriority.SPECULATIVE.rank == 0
    assert TtsRequestPriority.PREFETCH.rank == 1
    assert TtsRequestPriority.FOREGROUND.rank == 2


# ---------------------------------------------------------------------------
# coerce_priority
# ---------------------------------------------------------------------------


def test_coerce_priority_default_is_foreground():
    """The whole point of a back-compat default — older callers that
    pass no priority must get the latency-critical behaviour they
    already rely on."""
    assert coerce_priority(None) is TtsRequestPriority.FOREGROUND


def test_coerce_priority_accepts_string_and_enum():
    assert coerce_priority("speculative") is TtsRequestPriority.SPECULATIVE
    assert coerce_priority("PREFETCH") is TtsRequestPriority.PREFETCH  # case-insensitive
    assert coerce_priority(TtsRequestPriority.FOREGROUND) is TtsRequestPriority.FOREGROUND


def test_coerce_priority_rejects_typos_loudly():
    """A typo on ``"speculitave"`` should NOT silently demote the
    request — the resolver MUST raise so callers see their bug."""
    with pytest.raises(ValueError) as exc_info:
        coerce_priority("speculitave")
    assert "speculitave" in str(exc_info.value)


def test_coerce_priority_rejects_wrong_type():
    with pytest.raises(TypeError):
        coerce_priority(42)


# ---------------------------------------------------------------------------
# Fast path — single FOREGROUND call on idle scheduler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_foreground_acquire_pays_zero_queue_time():
    scheduler = TtsScheduler()
    async with await scheduler.acquire(key="kokoro", priority=TtsRequestPriority.FOREGROUND) as slot:
        assert slot.priority == TtsRequestPriority.FOREGROUND
        assert slot.queued_ms < 5.0  # microseconds in practice
        assert slot.cancelled is False


@pytest.mark.asyncio
async def test_release_makes_slot_available_for_next_caller():
    scheduler = TtsScheduler()
    async with await scheduler.acquire(key="k", priority=TtsRequestPriority.FOREGROUND):
        pass
    # Second acquire on the same key must also be immediate.
    async with await scheduler.acquire(key="k", priority=TtsRequestPriority.FOREGROUND) as slot:
        assert slot.queued_ms < 5.0


# ---------------------------------------------------------------------------
# Bounded concurrency — second caller waits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_concurrency_blocks_second_caller():
    scheduler = TtsScheduler(max_concurrency=1)
    first_acquired = asyncio.Event()
    second_acquired = asyncio.Event()
    release_first = asyncio.Event()

    async def first_holder():
        async with await scheduler.acquire(key="k", priority=TtsRequestPriority.FOREGROUND):
            first_acquired.set()
            await release_first.wait()

    async def second_caller():
        async with await scheduler.acquire(key="k", priority=TtsRequestPriority.FOREGROUND) as slot:
            second_acquired.set()
            assert slot.queued_ms >= 50.0  # at least the wait we forced

    holder_task = asyncio.create_task(first_holder())
    await first_acquired.wait()
    waiter_task = asyncio.create_task(second_caller())
    await asyncio.sleep(0.05)
    assert not second_acquired.is_set()  # blocked behind first holder
    release_first.set()
    await waiter_task
    await holder_task


# ---------------------------------------------------------------------------
# Different keys do not block each other
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_different_keys_allow_parallel_in_flight():
    scheduler = TtsScheduler(max_concurrency=1)
    async with await scheduler.acquire(key="kokoro", priority=TtsRequestPriority.FOREGROUND):
        # Second key must be free even though the first is held.
        async with await scheduler.acquire(key="pocket", priority=TtsRequestPriority.FOREGROUND) as slot:
            assert slot.queued_ms < 5.0


# ---------------------------------------------------------------------------
# Preemption — FOREGROUND arrival cancels SPECULATIVE in flight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_foreground_arrival_cancels_in_flight_speculative():
    scheduler = TtsScheduler(max_concurrency=1)
    speculative_done = asyncio.Event()
    cancel_called = asyncio.Event()

    async def speculative_holder():
        async with await scheduler.acquire(key="k", priority=TtsRequestPriority.SPECULATIVE) as slot:

            async def _cancel() -> None:
                cancel_called.set()

            slot.set_cancel(_cancel)
            # Hold until cancellation flag flips.
            for _ in range(100):
                if slot.cancelled:
                    break
                await asyncio.sleep(0.01)
            speculative_done.set()

    async def foreground_arrival():
        async with await scheduler.acquire(key="k", priority=TtsRequestPriority.FOREGROUND):
            pass

    spec_task = asyncio.create_task(speculative_holder())
    await asyncio.sleep(0.05)  # let speculative grab the slot
    fg_task = asyncio.create_task(foreground_arrival())

    await asyncio.wait_for(cancel_called.wait(), timeout=2.0)
    await asyncio.wait_for(speculative_done.wait(), timeout=2.0)
    await fg_task
    await spec_task

    assert scheduler.stats.speculative_cancellations == 1


@pytest.mark.asyncio
async def test_speculative_does_not_cancel_foreground():
    """The reverse direction must NOT happen — a SPECULATIVE arrival
    shouldn't dethrone a FOREGROUND in flight."""
    scheduler = TtsScheduler(max_concurrency=1)

    async def foreground_holder():
        async with await scheduler.acquire(key="k", priority=TtsRequestPriority.FOREGROUND) as slot:
            slot.set_cancel(lambda: asyncio.sleep(0))  # noop cancel
            # Hold long enough for the speculative arrival to register.
            await asyncio.sleep(0.05)
            assert slot.cancelled is False

    async def speculative_arrival():
        # Will simply queue; can't preempt foreground.
        async with await scheduler.acquire(key="k", priority=TtsRequestPriority.SPECULATIVE):
            pass

    holder_task = asyncio.create_task(foreground_holder())
    await asyncio.sleep(0.01)
    spec_task = asyncio.create_task(speculative_arrival())
    await holder_task
    await spec_task

    assert scheduler.stats.speculative_cancellations == 0


# ---------------------------------------------------------------------------
# Priority-ordered wakeup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_higher_priority_waiter_wakes_first_when_slot_frees():
    """Two waiters parked behind an in-flight call: when it releases,
    the higher-priority waiter must grab the slot first regardless
    of arrival order."""
    scheduler = TtsScheduler(max_concurrency=1)
    holder_release = asyncio.Event()
    log: list[str] = []

    async def holder():
        async with await scheduler.acquire(key="k", priority=TtsRequestPriority.FOREGROUND):
            await holder_release.wait()

    async def speculative_waiter():
        async with await scheduler.acquire(key="k", priority=TtsRequestPriority.SPECULATIVE):
            log.append("speculative")

    async def foreground_waiter():
        async with await scheduler.acquire(key="k", priority=TtsRequestPriority.FOREGROUND):
            log.append("foreground")

    holder_task = asyncio.create_task(holder())
    await asyncio.sleep(0.01)
    # Speculative arrives FIRST in wall-clock order...
    spec_task = asyncio.create_task(speculative_waiter())
    await asyncio.sleep(0.01)
    # ...but the foreground arriving second still wins the wakeup.
    fg_task = asyncio.create_task(foreground_waiter())
    await asyncio.sleep(0.01)

    holder_release.set()
    await asyncio.wait_for(asyncio.gather(spec_task, fg_task, holder_task), timeout=2.0)

    assert log == ["foreground", "speculative"]


# ---------------------------------------------------------------------------
# Stats snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_in_flight_count_reflects_active_holders():
    scheduler = TtsScheduler(max_concurrency=1)

    assert scheduler.stats.in_flight == 0
    async with await scheduler.acquire(key="k", priority=TtsRequestPriority.FOREGROUND):
        assert scheduler.stats.in_flight == 1
    assert scheduler.stats.in_flight == 0


@pytest.mark.asyncio
async def test_stats_queued_counter_increments_during_wait():
    scheduler = TtsScheduler(max_concurrency=1)
    holder_release = asyncio.Event()
    waiter_started = asyncio.Event()

    async def holder():
        async with await scheduler.acquire(key="k", priority=TtsRequestPriority.FOREGROUND):
            await holder_release.wait()

    async def waiter():
        waiter_started.set()
        async with await scheduler.acquire(key="k", priority=TtsRequestPriority.PREFETCH):
            pass

    holder_task = asyncio.create_task(holder())
    await asyncio.sleep(0.01)
    waiter_task = asyncio.create_task(waiter())
    await waiter_started.wait()
    await asyncio.sleep(0.05)
    # Waiter is queued at PREFETCH while holder owns the slot.
    assert scheduler.stats.prefetch_queued == 1

    holder_release.set()
    await asyncio.wait_for(asyncio.gather(holder_task, waiter_task), timeout=2.0)
    # Counter decrements after acquisition.
    assert scheduler.stats.prefetch_queued == 0
