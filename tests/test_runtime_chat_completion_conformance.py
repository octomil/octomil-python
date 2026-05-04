"""Conformance test for `chat.completion` against an external
runtime (octomil-runtime v0.1.1+).

v0.1.1 capability honesty is **structural**: chat.completion is
advertised whenever the runtime build linked llama.cpp and the
platform is supported. The env var `OCTOMIL_LLAMA_CPP_GGUF` no
longer affects capability advertisement — it's a developer
convenience that points the test at a real artifact for the
end-to-end paths.

Layered gate:

  1. Skip cleanly when the cffi binding can't load a runtime
     (matches the existing `requires_runtime` pattern).
  2. With the v0.1.1 dylib loaded, chat.completion IS advertised
     even without a model warmed; this test asserts that.
  3. Open + warm a model and run the full event-sequence pin
     (SESSION_STARTED → TRANSCRIPT_CHUNK+ → SESSION_COMPLETED with
     `terminal_status = OCT_STATUS_OK`) when `OCTOMIL_LLAMA_CPP_GGUF`
     points at a valid GGUF.

Operators run the artifact-present paths by setting:

    OCTOMIL_LLAMA_CPP_GGUF=/path/to/small-model.gguf
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

cffi = pytest.importorskip("cffi", reason="cffi extra not installed")  # noqa: F841

if TYPE_CHECKING:
    pass


CHAT_COMPLETION = "chat.completion"


@pytest.mark.requires_runtime
def test_chat_completion_capability_advertised_structurally():
    """v0.1.1: chat.completion is advertised whenever the engine is
    built and the platform is supported. The env var has no effect on
    capability advertisement; artifact identity / integrity / warm-
    readiness move to oct_model_open / oct_model_warm.

    A binding that calls `oct_session_open(capability='chat.completion',
    model=None)` should observe `OCT_STATUS_INVALID_INPUT` — capability
    is honest, but session-open requires a model handle."""
    from octomil.runtime.native import (
        OCT_STATUS_INVALID_INPUT,
        NativeRuntime,
        NativeRuntimeError,
    )

    with NativeRuntime.open() as rt:
        caps = rt.capabilities()
        assert (
            CHAT_COMPLETION in caps.supported_capabilities
        ), "chat.completion should be advertised structurally in v0.1.1"
        assert "llama_cpp" in caps.supported_engines, "llama_cpp engine should be advertised in v0.1.1"

        # session_open without a model -> INVALID_INPUT (capability
        # advertised, but precondition unmet).
        with pytest.raises(NativeRuntimeError) as exc_info:
            rt.open_session(capability=CHAT_COMPLETION, locality="on_device")
        assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
        # last_error names the missing-model condition.
        msg = exc_info.value.last_error.lower()
        assert "model" in msg, f"last_error should mention model: {msg!r}"


@pytest.mark.requires_runtime
def test_chat_completion_open_model_rejects_non_gguf():
    """oct_model_open validates the GGUF magic before computing
    sha256. A file that exists but isn't a GGUF returns
    OCT_STATUS_INVALID_INPUT with a precise diagnostic."""
    import tempfile

    from octomil.runtime.native import (
        OCT_STATUS_INVALID_INPUT,
        NativeRuntime,
        NativeRuntimeError,
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(b"NOT_A_GGUF_FILE")
        bad_path = f.name
    try:
        with NativeRuntime.open() as rt:
            with pytest.raises(NativeRuntimeError) as exc_info:
                rt.open_model(model_uri=bad_path)
            assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
            assert "gguf" in exc_info.value.last_error.lower()
    finally:
        os.unlink(bad_path)


@pytest.mark.requires_runtime
def test_chat_completion_open_model_rejects_digest_mismatch():
    """oct_model_open with an explicit `artifact_digest` rejects
    when the computed sha256 doesn't match. Skipped without a
    staged GGUF (need a valid file to even reach digest verification)."""
    from octomil.runtime.native import (
        OCT_STATUS_INVALID_INPUT,
        NativeRuntime,
        NativeRuntimeError,
    )

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing")

    with NativeRuntime.open() as rt:
        # Bogus digest — runtime computes the real sha256, compares,
        # rejects.
        with pytest.raises(NativeRuntimeError) as exc_info:
            rt.open_model(
                model_uri=gguf,
                artifact_digest="sha256:" + ("0" * 64),
            )
        assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
        assert "digest" in exc_info.value.last_error.lower()


@pytest.mark.requires_runtime
@pytest.mark.timeout(90)
def test_chat_completion_event_sequence_against_real_gguf():
    """End-to-end conformance pin (v0.1.1):

      open_runtime → open_model → warm → open_session(model=mdl)
      → SESSION_STARTED → send_text → TRANSCRIPT_CHUNK+ →
      SESSION_COMPLETED with terminal_status=OK
      → close_session → close_model → close_runtime

    Skipped unless `OCTOMIL_LLAMA_CPP_GGUF` points at a real GGUF.

    Pins:
      - Event sequence (SESSION_STARTED first; ≥1 chunk; clean
        SESSION_COMPLETED).
      - Operational envelope echoed verbatim on every event
        (request_id / route_id / trace_id / engine_version).
      - Artifact digest set on the envelope (v0.1.1: ALWAYS populated
        for chat.completion sessions because the model handle has it).
    """
    from octomil.runtime.native import (
        OCT_EVENT_ERROR,
        OCT_EVENT_NONE,
        OCT_EVENT_SESSION_COMPLETED,
        OCT_EVENT_SESSION_STARTED,
        OCT_EVENT_TRANSCRIPT_CHUNK,
        NativeRuntime,
    )

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing")

    with NativeRuntime.open() as rt:
        # v0.1.1 lifecycle: open + warm before session_open.
        mdl = rt.open_model(model_uri=gguf)
        try:
            mdl.warm()
            sess = rt.open_session(
                capability=CHAT_COMPLETION,
                locality="on_device",
                policy_preset="private",
                request_id="conf-req-001",
                route_id="conf-route-001",
                trace_id="0123456789abcdef0123456789abcdef",
                model=mdl,
            )
            try:

                def assert_envelope(ev):
                    """Pin envelope echoed verbatim on every event."""
                    assert ev.request_id == "conf-req-001"
                    assert ev.route_id == "conf-route-001"
                    assert ev.trace_id == "0123456789abcdef0123456789abcdef"
                    assert ev.engine_version.startswith("llama_cpp@")
                    # v0.1.1: artifact_digest always populated as
                    # "sha256:<64-hex>" on chat.completion sessions.
                    assert ev.artifact_digest.startswith("sha256:"), (
                        f"artifact_digest must be 'sha256:<hex>', got " f"{ev.artifact_digest!r}"
                    )
                    assert len(ev.artifact_digest) == 7 + 64

                # Single overall deadline for the whole sequence —
                # SESSION_STARTED + send_text + chunks +
                # SESSION_COMPLETED. 60s easily covers a small-GGUF
                # cold-load + first-token latency on CPU; the
                # @pytest.mark.timeout(90) override gives a 30s
                # margin over this internal deadline so the test
                # reports its own AssertionError before pytest's
                # repo-wide 60s watchdog fires.
                import time

                overall_deadline = time.monotonic() + 60.0

                # SESSION_STARTED — first event after open.
                ev = _wait_event(sess, overall_deadline)
                assert ev.type == OCT_EVENT_SESSION_STARTED, f"first event must be SESSION_STARTED, got type={ev.type}"
                assert_envelope(ev)

                # send_text in the canonical chat-messages JSON shape.
                sess.send_text('[{"role":"user","content":"hi"}]')

                n_chunks = 0
                saw_completed = False
                while time.monotonic() < overall_deadline:
                    ev = sess.poll_event(timeout_ms=200)
                    if ev is None or ev.type == OCT_EVENT_NONE:
                        # Drained timeout on this poll cycle. Slow
                        # first-token / cold cache paths land here;
                        # the overall deadline is the real budget.
                        continue
                    if ev.type == OCT_EVENT_TRANSCRIPT_CHUNK:
                        n_chunks += 1
                        assert_envelope(ev)
                    elif ev.type == OCT_EVENT_SESSION_COMPLETED:
                        saw_completed = True
                        assert_envelope(ev)
                        break
                    elif ev.type == OCT_EVENT_ERROR:
                        pytest.fail(
                            f"unexpected ERROR event "
                            f"(monotonic_ns={ev.monotonic_ns}); "
                            f"see runtime last_error for details"
                        )
                    else:
                        pytest.fail(f"unexpected event type {ev.type} in " f"chat.completion sequence")
                assert n_chunks >= 1, "no transcript chunks produced before overall deadline"
                assert saw_completed, "no SESSION_COMPLETED before overall deadline"
            finally:
                sess.close()
        finally:
            # Close model AFTER the session — order matters in v0.1.1
            # (model_close returns BUSY if any session still borrows).
            close_status = mdl.close()
            from octomil.runtime.native import OCT_STATUS_OK

            assert close_status == OCT_STATUS_OK, f"model.close after session.close should be OK, got {close_status}"


@pytest.mark.requires_runtime
def test_chat_completion_model_close_returns_busy_when_session_live():
    """v0.1.1: oct_model_close returns OCT_STATUS_BUSY when any
    session still borrows the model. The handle remains valid;
    the binding closes its sessions and retries close. Skipped
    without a staged GGUF (need a real warmed model)."""
    from octomil.runtime.native import (
        OCT_STATUS_BUSY,
        OCT_STATUS_OK,
        NativeRuntime,
    )

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing")

    with NativeRuntime.open() as rt:
        mdl = rt.open_model(model_uri=gguf)
        mdl.warm()
        sess = rt.open_session(
            capability=CHAT_COMPLETION,
            locality="on_device",
            policy_preset="private",
            model=mdl,
        )
        try:
            # Close MUST return BUSY — session is still borrowing.
            assert mdl.close() == OCT_STATUS_BUSY, "model.close while session live should return BUSY"
            # evict has the same contract.
            assert mdl.evict() == OCT_STATUS_BUSY, "model.evict while session live should return BUSY"
        finally:
            sess.close()
        # After session close, model.close returns OK.
        assert mdl.close() == OCT_STATUS_OK, "model.close after session drain should return OK"


@pytest.mark.requires_runtime
def test_chat_completion_cross_runtime_model_rejected():
    """v0.1.1 R1 Codex: a NativeModel opened against runtime A
    cannot be passed to runtime B's open_session. Defense-in-depth
    at the binding layer surfaces a precise typed error before
    crossing the FFI boundary, regardless of whether the dylib
    rejects the same shape."""
    from octomil.runtime.native import (
        OCT_STATUS_INVALID_INPUT,
        NativeRuntime,
        NativeRuntimeError,
    )

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing")

    rt_a = NativeRuntime.open()
    rt_b = NativeRuntime.open()
    try:
        mdl_a = rt_a.open_model(model_uri=gguf)
        try:
            with pytest.raises(NativeRuntimeError) as exc_info:
                rt_b.open_session(
                    capability=CHAT_COMPLETION,
                    locality="on_device",
                    policy_preset="private",
                    model=mdl_a,
                )
            assert exc_info.value.status == OCT_STATUS_INVALID_INPUT
            assert "cross-runtime" in str(exc_info.value).lower() or "different" in str(exc_info.value).lower()
        finally:
            mdl_a.close()
    finally:
        rt_a.close()
        rt_b.close()


@pytest.mark.requires_runtime
def test_chat_completion_session_pins_borrowed_model_against_gc():
    """v0.1.1 R1 Codex P1 fix: a NativeSession holds a strong ref
    to the NativeModel it borrowed, so the user pattern
    `rt.open_session(model=rt.open_model(...))` doesn't leave the
    model wrapper unreferenced (NativeRuntime._models is a WeakSet;
    GC could otherwise drop the wrapper, leak the C-side handle,
    and cause runtime_close to refuse with last_error).
    """
    import gc

    from octomil.runtime.native import (
        NativeRuntime,
    )

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing")

    with NativeRuntime.open() as rt:
        # Inline temporary model — no local name. The session's
        # strong ref is the only thing keeping the wrapper alive.
        sess = rt.open_session(
            capability=CHAT_COMPLETION,
            locality="on_device",
            policy_preset="private",
            model=rt.open_model(model_uri=gguf),
        )
        try:
            # Force a GC sweep — the inline temporary would normally
            # be collected here. The session's _borrowed_model strong
            # ref must keep the wrapper alive.
            gc.collect()
            # The session is still operable; envelope.artifact_digest
            # should still be readable. We don't run a full inference
            # here (covered by event_sequence_against_real_gguf); just
            # confirm the session is open and the model wrapper is
            # reachable from the session.
            assert sess._borrowed_model is not None
            assert not sess._borrowed_model._closed
            assert not sess._borrowed_model._handle_invalid
        finally:
            # close() the session; the model close happens via the
            # WeakSet drain in NativeRuntime.close().
            sess.close()
        # After session close, the strong ref drops; the WeakSet
        # cleanup in runtime.close() reaps the model. We confirm by
        # closing the runtime and asserting last_error is empty
        # (no refusal because models were properly drained).
    # Re-open just to read last_error of the closed runtime is not
    # possible; the empty-last-error check is implicit — if the
    # cascade was broken, runtime_close would have left a refusal
    # diagnostic in the now-freed runtime, which is not observable
    # from here. The earlier `gc.collect()` survival is the
    # operational pin.


def _wait_event(sess, overall_deadline: float):
    """Poll `sess` until a non-NONE event arrives or the overall
    deadline elapses. Used for the first-event pin where a
    timeout is genuinely a failure."""
    import time

    from octomil.runtime.native import OCT_EVENT_NONE

    while time.monotonic() < overall_deadline:
        ev = sess.poll_event(timeout_ms=200)
        if ev is None or ev.type == OCT_EVENT_NONE:
            continue
        return ev
    raise AssertionError("no event before overall deadline")
