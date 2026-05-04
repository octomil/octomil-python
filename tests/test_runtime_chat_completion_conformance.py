"""Conformance test for `chat.completion` against an external
runtime (octomil-runtime v0.1.0+).

Layered gate:

  1. Skip cleanly when the cffi binding can't load a runtime
     (matches the existing `requires_runtime` pattern).
  2. Skip cleanly when `chat.completion` is NOT advertised — i.e.,
     when `OCTOMIL_LLAMA_CPP_GGUF` is unset OR points at a missing
     / non-GGUF file. The runtime's structural capability-honesty
     filter is the source of truth; this test reads
     `RuntimeCapabilities.supported` and skips when the capability
     is absent.
  3. When the capability IS advertised, run the full event-sequence
     pin: SESSION_STARTED → TRANSCRIPT_CHUNK+ → SESSION_COMPLETED
     with `terminal_status = OCT_STATUS_OK`.

Per the engineering-debate consensus on octomil-runtime#3, the
runtime ships v0.1.0 with `chat.completion` dev-gated. CI without
a staged GGUF hits the skip path at gate (2). Operators with a
small GGUF run the test by setting:

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
def test_chat_completion_capability_advertised_when_gguf_staged():
    """When `OCTOMIL_LLAMA_CPP_GGUF` points at a valid GGUF, the
    runtime MUST advertise `chat.completion` AND `llama_cpp` in
    its capability list. When unset / invalid, both must be
    absent.

    This test reflects the runtime's structural capability-honesty
    rule into the SDK side."""
    from octomil.runtime.native import NativeRuntime

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    with NativeRuntime.open() as rt:
        caps = rt.capabilities()
        if not gguf or not os.path.isfile(gguf):
            # Negative-case assertions. Both must hold per the
            # runtime's structural capability-honesty rule:
            # registry drops engines that contributed zero
            # loadable capabilities. Codex R1 missed-case fix:
            # also assert llama_cpp is absent (was implicit only).
            assert CHAT_COMPLETION not in caps.supported_capabilities, (
                "chat.completion advertised without a valid GGUF " "— capability honesty filter regression"
            )
            assert "llama_cpp" not in caps.supported_engines, (
                "llama_cpp engine advertised without a valid GGUF "
                "— registry should drop engines with zero loadable capabilities"
            )
            return
        # Otherwise the file exists; the runtime's own GGUF magic-
        # byte check is the authority on whether the capability
        # actually advertises. If the file isn't a real GGUF the
        # runtime drops it and we still skip.
        if CHAT_COMPLETION not in caps.supported_capabilities:
            pytest.skip(
                f"OCTOMIL_LLAMA_CPP_GGUF={gguf!r} does not pass the "
                f"runtime's GGUF magic-byte check; capability not "
                f"advertised"
            )
        assert (
            "llama_cpp" in caps.supported_engines
        ), "chat.completion advertised but llama_cpp not in supported_engines"


@pytest.mark.requires_runtime
def test_chat_completion_event_sequence_against_real_gguf():
    """End-to-end conformance pin: SESSION_STARTED →
    TRANSCRIPT_CHUNK+ → SESSION_COMPLETED, with the operational
    envelope echoed verbatim from the session config.

    Skipped unless `OCTOMIL_LLAMA_CPP_GGUF` points at a real GGUF
    AND the runtime is advertising `chat.completion`."""
    from octomil.runtime.native import NativeRuntime

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing")

    with NativeRuntime.open() as rt:
        caps = rt.capabilities()
        if CHAT_COMPLETION not in caps.supported_capabilities:
            pytest.skip("runtime does not advertise chat.completion (GGUF " "magic-byte check failed?)")

        sess = rt.open_session(
            capability=CHAT_COMPLETION,
            locality="on_device",
            policy_preset="private",
            request_id="conf-req-001",
            route_id="conf-route-001",
            trace_id="0123456789abcdef0123456789abcdef",
        )
        try:
            # SESSION_STARTED — first event after open.
            ev = _drain_one(sess, timeout_ms=500)
            assert ev.type_name == "SESSION_STARTED", ev.type_name
            assert ev.request_id == "conf-req-001"
            assert ev.route_id == "conf-route-001"
            assert ev.trace_id == "0123456789abcdef0123456789abcdef"
            assert ev.engine_version.startswith("llama_cpp@")

            # send_text in the canonical chat-messages JSON shape.
            sess.send_text('[{"role":"user","content":"hi"}]')

            n_chunks = 0
            saw_completed = False
            for _ in range(10_000):
                ev = _drain_one(sess, timeout_ms=200)
                if ev.type_name == "TRANSCRIPT_CHUNK":
                    n_chunks += 1
                    # Codex R1 missed-case fix: pin the FULL
                    # envelope on every chunk, not just request_id.
                    # Verifies the runtime echoes verbatim across
                    # every event type, not just SESSION_STARTED.
                    assert ev.request_id == "conf-req-001"
                    assert ev.route_id == "conf-route-001"
                    assert ev.trace_id == "0123456789abcdef0123456789abcdef"
                    assert ev.engine_version.startswith("llama_cpp@")
                elif ev.type_name == "SESSION_COMPLETED":
                    saw_completed = True
                    # Same envelope-echo assertions on the
                    # terminal event. A regression that drops the
                    # envelope on the terminal frame would lose
                    # observability for cancelled / errored
                    # sessions.
                    assert ev.request_id == "conf-req-001"
                    assert ev.route_id == "conf-route-001"
                    assert ev.trace_id == "0123456789abcdef0123456789abcdef"
                    # OCT_STATUS_OK = 0
                    assert ev.terminal_status == 0, f"expected terminal OK, got {ev.terminal_status}"
                    break
                elif ev.type_name == "ERROR":
                    pytest.fail(f"unexpected ERROR event: {ev.error_message}")
            assert n_chunks >= 1, "no transcript chunks produced"
            assert saw_completed, "no SESSION_COMPLETED before iter cap"
        finally:
            sess.close()


def _drain_one(sess, timeout_ms: int = 100):
    """Helper that polls `sess` until an event of type != NONE
    arrives, or timeout."""
    import time

    deadline = time.monotonic() + (timeout_ms / 1000.0)
    while time.monotonic() < deadline:
        ev = sess.poll_event(timeout_ms=10)
        if ev is None or ev.type_name == "NONE":
            continue
        return ev
    raise AssertionError(f"no event within {timeout_ms} ms")
