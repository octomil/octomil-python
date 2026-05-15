"""Verify per-backend skip semantics against the current status model.

These are meta-tests of the harness itself, not behavioral conformance
tests. They pin the Python SDK partition to exactly three capability
states: DONE_NATIVE_CUTOVER, LIVE_NATIVE_CONDITIONAL, and
BLOCKED_WITH_PROOF.
"""

from __future__ import annotations

import pytest

from octomil.runtime.native.capabilities import (
    BLOCKED_WITH_PROOF_CAPABILITIES,
    CAPABILITY_AUDIO_DIARIZATION,
    CAPABILITY_AUDIO_REALTIME_SESSION,
    CAPABILITY_AUDIO_SPEAKER_EMBEDDING,
    CAPABILITY_AUDIO_STT_BATCH,
    CAPABILITY_AUDIO_STT_STREAM,
    CAPABILITY_AUDIO_TRANSCRIPTION,
    CAPABILITY_AUDIO_TTS_BATCH,
    CAPABILITY_AUDIO_TTS_STREAM,
    CAPABILITY_AUDIO_VAD,
    CAPABILITY_CACHE_INTROSPECT,
    CAPABILITY_CHAT_COMPLETION,
    CAPABILITY_CHAT_STREAM,
    CAPABILITY_EMBEDDINGS_IMAGE,
    CAPABILITY_EMBEDDINGS_TEXT,
    CAPABILITY_INDEX_VECTOR_QUERY,
    CAPABILITY_STATUS_BLOCKED_WITH_PROOF,
    CAPABILITY_STATUS_DONE_NATIVE_CUTOVER,
    CAPABILITY_STATUS_LIVE_NATIVE_CONDITIONAL,
    DONE_NATIVE_CUTOVER_CAPABILITIES,
    LIVE_NATIVE_CONDITIONAL_CAPABILITIES,
    RUNTIME_CAPABILITIES,
    RUNTIME_CAPABILITY_STATUSES,
)
from tests.runtime_conformance.conftest import (
    BACKEND_NATIVE,
    BACKEND_PYTHON,
    PYTHON_ORACLE_CAPABILITIES,
    requires_capability,
)


@requires_capability(CAPABILITY_CHAT_COMPLETION)
def test_done_native_cutover_capability_runs_on_python_backend(backend: str):
    """DONE_NATIVE_CUTOVER capabilities are runnable on python and
    native when the runtime advertises them."""
    if backend == BACKEND_PYTHON:
        assert CAPABILITY_CHAT_COMPLETION in PYTHON_ORACLE_CAPABILITIES
        assert RUNTIME_CAPABILITY_STATUSES[CAPABILITY_CHAT_COMPLETION] == CAPABILITY_STATUS_DONE_NATIVE_CUTOVER
    elif backend == BACKEND_NATIVE:
        pass


@requires_capability(CAPABILITY_AUDIO_TTS_STREAM)
def test_live_native_conditional_capability_runs_on_python_backend(backend: str):
    """LIVE_NATIVE_CONDITIONAL means real native support exists, but
    runtime advertisement depends on build/artifact gates."""
    if backend == BACKEND_PYTHON:
        assert CAPABILITY_AUDIO_TTS_STREAM in PYTHON_ORACLE_CAPABILITIES
        assert RUNTIME_CAPABILITY_STATUSES[CAPABILITY_AUDIO_TTS_STREAM] == CAPABILITY_STATUS_LIVE_NATIVE_CONDITIONAL
    elif backend == BACKEND_NATIVE:
        pass


@requires_capability(CAPABILITY_AUDIO_REALTIME_SESSION)
def test_blocked_capability_never_reaches_python_backend(backend: str):
    """BLOCKED_WITH_PROOF capabilities are legal names, not live
    runtime surfaces. Python skips them; native must not advertise
    them while they remain blocked."""
    if backend == BACKEND_PYTHON:
        pytest.fail("blocked capability reached python backend; harness misconfigured")
    elif backend == BACKEND_NATIVE:
        pytest.fail("blocked capability was advertised by the native runtime")


def test_no_marker_runs_on_every_backend(backend: str):
    """A test without a marker has no capability skip rules."""
    assert backend in (BACKEND_PYTHON, BACKEND_NATIVE)


def test_capability_status_partition_is_complete_and_disjoint():
    """Every canonical capability belongs to exactly one live status."""
    allowed_statuses = {
        CAPABILITY_STATUS_DONE_NATIVE_CUTOVER,
        CAPABILITY_STATUS_LIVE_NATIVE_CONDITIONAL,
        CAPABILITY_STATUS_BLOCKED_WITH_PROOF,
    }

    assert set(RUNTIME_CAPABILITY_STATUSES) == set(RUNTIME_CAPABILITIES)
    assert set(RUNTIME_CAPABILITY_STATUSES.values()) == allowed_statuses
    assert DONE_NATIVE_CUTOVER_CAPABILITIES.isdisjoint(LIVE_NATIVE_CONDITIONAL_CAPABILITIES)
    assert DONE_NATIVE_CUTOVER_CAPABILITIES.isdisjoint(BLOCKED_WITH_PROOF_CAPABILITIES)
    assert LIVE_NATIVE_CONDITIONAL_CAPABILITIES.isdisjoint(BLOCKED_WITH_PROOF_CAPABILITIES)
    assert (
        DONE_NATIVE_CUTOVER_CAPABILITIES | LIVE_NATIVE_CONDITIONAL_CAPABILITIES | BLOCKED_WITH_PROOF_CAPABILITIES
    ) == RUNTIME_CAPABILITIES


def test_done_native_cutover_set_is_exact():
    assert DONE_NATIVE_CUTOVER_CAPABILITIES == {
        CAPABILITY_CHAT_COMPLETION,
        CAPABILITY_CHAT_STREAM,
        CAPABILITY_EMBEDDINGS_TEXT,
    }


def test_live_native_conditional_set_is_exact():
    assert LIVE_NATIVE_CONDITIONAL_CAPABILITIES == {
        CAPABILITY_AUDIO_DIARIZATION,
        CAPABILITY_AUDIO_SPEAKER_EMBEDDING,
        CAPABILITY_AUDIO_STT_BATCH,
        CAPABILITY_AUDIO_STT_STREAM,
        CAPABILITY_AUDIO_TRANSCRIPTION,
        CAPABILITY_AUDIO_TTS_BATCH,
        CAPABILITY_AUDIO_TTS_STREAM,
        CAPABILITY_AUDIO_VAD,
        CAPABILITY_CACHE_INTROSPECT,
        CAPABILITY_EMBEDDINGS_IMAGE,
    }


def test_blocked_with_proof_set_is_exact():
    assert BLOCKED_WITH_PROOF_CAPABILITIES == {
        CAPABILITY_AUDIO_REALTIME_SESSION,
        CAPABILITY_INDEX_VECTOR_QUERY,
    }


def test_python_oracle_is_derived_from_non_blocked_statuses():
    assert PYTHON_ORACLE_CAPABILITIES == DONE_NATIVE_CUTOVER_CAPABILITIES | LIVE_NATIVE_CONDITIONAL_CAPABILITIES
    assert PYTHON_ORACLE_CAPABILITIES.isdisjoint(BLOCKED_WITH_PROOF_CAPABILITIES)


def test_deprecated_status_names_are_absent():
    deprecated = {"".join(("SCAFFOLD", "_ONLY")), "".join(("LAYER", "_2B", "_ONLY"))}
    assert set(RUNTIME_CAPABILITY_STATUSES.values()).isdisjoint(deprecated)
