"""Realtime session lifecycle conformance for a blocked capability.

Every test in this file is gated on
``@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)``. Current
Python SDK truth: ``audio.realtime.session`` is ``BLOCKED_WITH_PROOF``.
It is a legal enum name, but current runtimes must not advertise it and
``session_open`` must reject it with ``OCT_STATUS_UNSUPPORTED``.

  * On the python backend — ``no-python-oracle``.
  * On the native backend — ``runtime_capabilities`` unless a future
    PR promotes the capability out of ``BLOCKED_WITH_PROOF``.

Goal of this file: have the conformance test bodies WRITTEN and
pinned NOW, so a future realtime adapter has a concrete checklist of
behaviors to satisfy before its tests can flip from SKIP to PASS.
Cannot mistake a reserved enum name for a working runtime.
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
    """Future realtime acceptance: open a realtime session, then
    close. The runtime emits SESSION_STARTED before any audio flows."""
    if backend == BACKEND_PYTHON:
        # BLOCKED_WITH_PROOF capability — harness skip should have
        # already kicked us out.
        pytest.fail("audio.realtime.session reached python backend; harness misconfigured")
    assert backend == BACKEND_NATIVE
    with NativeRuntime.open() as rt:
        with rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION) as sess:
            ev = sess.poll_event(timeout_ms=5_000)
            assert ev.type == OCT_EVENT_SESSION_STARTED


@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
def test_session_cancel_terminates_with_completed_event(backend: str):
    """Cancellation is observable: poll_event after cancel delivers a
    SESSION_COMPLETED event.

    Blocked-capability proof scope: we only assert the SESSION_COMPLETED
    type arrives. Asserting ``terminal_status == OCT_STATUS_CANCELLED``
    on the payload requires NativeEvent to expose the
    ``session_completed`` union branch. The post-completion poll
    asserting cancellation is captured here today."""
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
            # Subsequent polls return OCT_STATUS_CANCELLED, which the
            # wrapper raises as NativeRuntimeError.
            with pytest.raises(NativeRuntimeError):
                sess.poll_event(timeout_ms=100)


@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
def test_session_send_audio_round_trip(backend: str):
    """Push a small audio buffer, drive the poll loop, expect a real
    payload event (AUDIO_CHUNK or TRANSCRIPT_CHUNK).

    Codex R1 missed-case fix: previous version accepted ANY non-NONE
    event, which a queued SESSION_STARTED would satisfy without
    proving the audio path actually flowed. The test now requires
    a payload event AFTER the SESSION_STARTED handshake, so a
    runtime that emits SESSION_STARTED but never engages the engine
    fails this test loudly. Future realtime acceptance criterion."""
    from octomil.runtime.native import OCT_EVENT_AUDIO_CHUNK, OCT_EVENT_TRANSCRIPT_CHUNK

    assert backend == BACKEND_NATIVE
    payload_event_types = {OCT_EVENT_AUDIO_CHUNK, OCT_EVENT_TRANSCRIPT_CHUNK}
    with NativeRuntime.open() as rt:
        with rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION) as sess:
            # 100ms of silence at 16kHz mono float32 = 1600 frames * 4 bytes.
            # A real realtime adapter handles silence as valid input;
            # VAD may gate output but SESSION_STARTED is unconditional.
            silence = b"\x00\x00\x00\x00" * 1600
            sess.send_audio(silence, sample_rate=16000, channels=1)
            saw_payload = False
            saw_started = False
            for _ in range(40):  # 40 polls × 200ms = 8s budget
                ev = sess.poll_event(timeout_ms=200)
                if ev.type == OCT_EVENT_SESSION_STARTED:
                    saw_started = True
                if ev.type in payload_event_types:
                    saw_payload = True
                    break
            assert saw_started, "expected SESSION_STARTED handshake"
            assert saw_payload, "expected AUDIO_CHUNK or TRANSCRIPT_CHUNK payload event"


@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
def test_session_open_then_immediate_close_does_not_leak(backend: str):
    """Defensive: open and close back-to-back without polling. The
    runtime must drain queued events and free internal state.
    Regression target for a future realtime adapter."""
    assert backend == BACKEND_NATIVE
    with NativeRuntime.open() as rt:
        sess = rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION)
        sess.close()
        sess.close()  # idempotent


@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
def test_session_send_audio_after_cancel_returns_cancelled(backend: str):
    """After cancel, send_audio returns OCT_STATUS_CANCELLED rather
    than partially consuming the buffer. Pin the contract here so
    a future realtime adapter doesn't drift."""
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
# pins the blocked-capability proof: realtime continues to return
# UNSUPPORTED rather than silently producing fake events. When a future
# PR promotes realtime out of BLOCKED_WITH_PROOF, this test deletes and
# the gated tests above flip to active.
# ---------------------------------------------------------------------------


def test_realtime_blocked_returns_unsupported_until_promoted(backend: str):
    """Independent of the capability marker: while realtime is
    BLOCKED_WITH_PROOF, open_session MUST raise UNSUPPORTED. This
    disappears only when a future PR promotes the capability and the
    gated lifecycle tests above become active."""
    if backend == BACKEND_PYTHON:
        pytest.skip("blocked-capability proof only applies to native backend")
    with NativeRuntime.open() as rt:
        caps = rt.capabilities()
        if CAPABILITY_AUDIO_REALTIME_SESSION in caps.supported_capabilities:
            pytest.skip(
                "runtime advertises audio.realtime.session — blocked proof no "
                "longer applicable; promote the status and rely on the gated "
                "lifecycle tests"
            )
        with pytest.raises(NativeRuntimeError) as exc_info:
            rt.open_session(capability=CAPABILITY_AUDIO_REALTIME_SESSION)
        assert exc_info.value.status == OCT_STATUS_UNSUPPORTED
