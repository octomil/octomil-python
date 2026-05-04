"""Verify the harness's per-backend skip semantics.

These are meta-tests of the harness itself, not behavioral
conformance tests. They exercise:

  * Capability owned by Python: runs on python, runs on native iff
    advertised.
  * Capability NOT owned by Python (native-first): skips on python
    with ``no-python-oracle``, skips on native iff not advertised.
  * Capability NOT in PYTHON_ORACLE_CAPABILITIES (e.g. ``embeddings.text``,
    which v0.4 admitted to the runtime enum but Python does NOT yet
    own as oracle): skips on python with ``no-python-oracle``, skips
    on native until the runtime advertises it.
"""

from __future__ import annotations

import pytest

from octomil.runtime.native.capabilities import (
    CAPABILITY_AUDIO_REALTIME_SESSION,
    CAPABILITY_AUDIO_TTS_STREAM,
    CAPABILITY_CHAT_COMPLETION,
)
from tests.runtime_conformance.conftest import (
    BACKEND_NATIVE,
    BACKEND_PYTHON,
    PYTHON_ORACLE_CAPABILITIES,
    requires_capability,
)

# ---------------------------------------------------------------------------
# Capability owned by Python (audio.tts.stream)
# ---------------------------------------------------------------------------


@requires_capability(CAPABILITY_AUDIO_TTS_STREAM)
def test_python_owned_capability(backend: str):
    """Python owns audio.tts.stream as oracle — runs on the python
    backend. On native, runs only if the runtime advertises it
    (slice-2 stub does not advertise → skipped via the marker)."""
    if backend == BACKEND_PYTHON:
        # Sanity: the marker's PYTHON_ORACLE_CAPABILITIES check
        # passed, so we got here.
        assert CAPABILITY_AUDIO_TTS_STREAM in PYTHON_ORACLE_CAPABILITIES
    elif backend == BACKEND_NATIVE:
        # Only reachable if the native runtime advertises this cap.
        # Slice-2-stub does not — this branch only runs post-slice-2-proper.
        pass


# ---------------------------------------------------------------------------
# Native-first capability (audio.realtime.session)
# ---------------------------------------------------------------------------


@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
def test_native_first_capability(backend: str):
    """audio.realtime.session is native-first — Python has no
    implementation. On python: skip with no-python-oracle (verified
    by the fact that this test body never runs on python). On native:
    skip with runtime_capabilities until slice 2-proper advertises
    it."""
    if backend == BACKEND_PYTHON:
        pytest.fail(
            "this branch should be unreachable — no-python-oracle skip should have fired BEFORE the test body ran"
        )
    elif backend == BACKEND_NATIVE:
        # Reachable iff the runtime advertises audio.realtime.session.
        # Slice-2-stub: skipped via the marker.
        pass


# ---------------------------------------------------------------------------
# Capability owned by Python (chat.completion)
# ---------------------------------------------------------------------------


@requires_capability(CAPABILITY_CHAT_COMPLETION)
def test_chat_completion_python_owned(backend: str):
    """chat.completion is in PYTHON_ORACLE_CAPABILITIES — runs on
    python. On native: skipped against the slice-2 stub."""
    if backend == BACKEND_PYTHON:
        assert CAPABILITY_CHAT_COMPLETION in PYTHON_ORACLE_CAPABILITIES


# ---------------------------------------------------------------------------
# Marker-less test (no capability gate) — runs on every backend
# ---------------------------------------------------------------------------


def test_no_marker_runs_on_every_backend(backend: str):
    """A test without a `requires_capability` marker has no skip
    rules — runs on whichever backend is selected."""
    assert backend in (BACKEND_PYTHON, BACKEND_NATIVE)


# ---------------------------------------------------------------------------
# PYTHON_ORACLE_CAPABILITIES integrity
# ---------------------------------------------------------------------------


def test_python_oracle_does_not_include_realtime():
    """Native-first realtime is intentionally absent from the
    Python oracle set."""
    assert CAPABILITY_AUDIO_REALTIME_SESSION not in PYTHON_ORACLE_CAPABILITIES


def test_python_oracle_includes_tts_chat_stt():
    """Sanity: capabilities Python actually owns are in the set."""
    assert CAPABILITY_AUDIO_TTS_STREAM in PYTHON_ORACLE_CAPABILITIES
    assert CAPABILITY_CHAT_COMPLETION in PYTHON_ORACLE_CAPABILITIES


def test_python_oracle_does_not_include_embeddings_text():
    """v0.4: embeddings.text IS in the runtime capability enum
    (admitted in octomil-contracts#99 + this PR's capabilities.py)
    BUT Python does NOT yet own it as oracle. The conformance
    harness skips embeddings.text tests on the python backend with
    `no-python-oracle` until/unless a Python embedding kernel
    becomes the comparison reference."""
    assert "embeddings.text" not in PYTHON_ORACLE_CAPABILITIES
