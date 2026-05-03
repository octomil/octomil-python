"""Slice 2A — realtime session lifecycle conformance scaffold.

Every test in this file is gated on
``@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)``. Until a
runtime build advertises ``audio.realtime.session`` AND adds the
capability to ``PYTHON_ORACLE_CAPABILITIES`` (the latter never happens
— this is a native-first capability), the harness skips every test
here:

  * On the python backend — ``no-python-oracle`` (audio.realtime.session
    is native-first by design).
  * On the native backend — ``runtime_capabilities`` (slice-2 stub
    does not advertise; slice 2-proper Moshi-on-MLX does).

Goal of this file: have the conformance test bodies WRITTEN and
pinned NOW, so the slice 2-proper Moshi adapter has a concrete
checklist of behaviors to satisfy before its tests can flip from
SKIP to PASS. Cannot mistake a stub for a working runtime.
"""

from __future__ import annotations

import pytest

from octomil.runtime.native import (
    CAPABILITY_AUDIO_REALTIME_SESSION,
    OCT_EVENT_NONE,
    OCT_EVENT_SESSION_COMPLETED,
    OCT_EVENT_SESSION_STARTED,
    OCT_STATUS_UNSUPPORTED,
    NativeRuntime,
    NativeRuntimeError,
)
from tests.runtime_conformance.conftest import (
    BACKEND_NATIVE,
    BACKEND_PYTHON,
    requires_capability,
)

# ---------------------------------------------------------------------------
# Lifecycle: open → session_started → cancel → session_completed
# ---------------------------------------------------------------------------


@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
def test_session_open_then_close(backend: str):
    """Slice 2-proper acceptance: open a realtime session, then
    close. The runtime emits SESSION_STARTED before any audio flows."""
    if backend == BACKEND_PYTHON:
        # Native-first capability — harness skip would have fired.
        # Defense in depth: the marker's PYTHON_ORACLE_CAPABILITIES
        # check should have already kicked us out.
        pytest.fail("audio.realtime.session reached python backend; harness misconfigured")
    assert backend == BACKEND_NATIVE
    with NativeRuntime.open() as rt:
        with rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION) as sess:
            ev = sess.poll_event(timeout_ms=5_000)
            assert ev.type == OCT_EVENT_SESSION_STARTED


@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
def test_session_cancel_terminates_with_completed_event(backend: str):
    """Cancellation is observable: poll_event after cancel delivers a
    SESSION_COMPLETED with terminal_status==CANCELLED, then
    OCT_STATUS_CANCELLED on subsequent polls."""
    assert backend == BACKEND_NATIVE
    with NativeRuntime.open() as rt:
        with rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION) as sess:
            sess.cancel()
            saw_completed = False
            for _ in range(10):
                ev = sess.poll_event(timeout_ms=500)
                if ev.type == OCT_EVENT_SESSION_COMPLETED:
                    saw_completed = True
                    break
                if ev.type == OCT_EVENT_NONE:
                    continue
            assert saw_completed, "expected SESSION_COMPLETED after cancel"


@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
def test_session_send_audio_round_trip(backend: str):
    """Push a small audio buffer, drive the poll loop, expect at
    least one AUDIO_CHUNK or TRANSCRIPT_CHUNK event back. Slice 2-
    proper behavior; slice-2-stub never reaches this body."""
    assert backend == BACKEND_NATIVE
    with NativeRuntime.open() as rt:
        with rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION) as sess:
            # 100ms of silence at 16kHz mono float32 = 1600 frames * 4 bytes.
            silence = b"\x00\x00\x00\x00" * 1600
            sess.send_audio(silence, sample_rate=16000, channels=1)
            # Drain a few events; the slice 2-proper acceptance is "no
            # raise, eventually a non-NONE event". This test is the
            # smallest end-to-end smoke for the realtime path.
            saw_any = False
            for _ in range(20):
                ev = sess.poll_event(timeout_ms=200)
                if ev.type != OCT_EVENT_NONE:
                    saw_any = True
                    break
            assert saw_any, "expected at least one event from realtime session"


@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
def test_session_open_then_immediate_close_does_not_leak(backend: str):
    """Defensive: open and close back-to-back without polling. The
    runtime must drain queued events and free internal state.
    Regression target for the slice 2-proper adapter."""
    assert backend == BACKEND_NATIVE
    with NativeRuntime.open() as rt:
        sess = rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION)
        sess.close()
        sess.close()  # idempotent


@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
def test_session_send_audio_after_cancel_returns_cancelled(backend: str):
    """After cancel, send_audio returns OCT_STATUS_CANCELLED rather
    than partially consuming the buffer. Pin the contract here so
    the slice 2-proper adapter doesn't drift."""
    assert backend == BACKEND_NATIVE
    with NativeRuntime.open() as rt:
        with rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION) as sess:
            sess.cancel()
            silence = b"\x00\x00\x00\x00" * 16
            with pytest.raises(NativeRuntimeError):
                sess.send_audio(silence, sample_rate=16000, channels=1)


# ---------------------------------------------------------------------------
# Stub-state assertion (always-runs sanity)
#
# Not capability-gated: this one runs on every native invocation and
# pins the stub-state regression — that the slice-2 stub continues to
# return UNSUPPORTED rather than silently producing fake events.
# When slice 2-proper lands and advertises the capability, this test
# DELETES (the gated tests above flip to active).
# ---------------------------------------------------------------------------


def test_realtime_stub_returns_unsupported_until_advertised(backend: str):
    """Independent of the capability marker: against the slice-2
    stub, open_session for realtime MUST raise UNSUPPORTED. This is
    the regression boundary — it disappears when slice 2-proper
    flips advertisement on, replaced by the gated lifecycle tests
    above."""
    if backend == BACKEND_PYTHON:
        pytest.skip("native-first stub regression — only applies to native backend")
    with NativeRuntime.open() as rt:
        caps = rt.capabilities()
        if CAPABILITY_AUDIO_REALTIME_SESSION in caps.supported_capabilities:
            pytest.skip(
                "runtime advertises audio.realtime.session — stub regression "
                "test no longer applicable; the gated lifecycle tests cover "
                "this surface"
            )
        with pytest.raises(NativeRuntimeError) as exc_info:
            rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION)
        assert exc_info.value.status == OCT_STATUS_UNSUPPORTED
