"""PR B: routing controls + private app refusal.

Three reviewer concerns get pinned here:

  1. ``policy=`` and ``app=`` flow through the public facade for TTS
     (and equivalently for transcription / chat / responses).
  2. ``@app/...`` refs do NOT silently cloud-route when the planner
     fails. With explicit ``policy='local_only'`` /
     ``policy='private'``, ``cloud_available`` is forced False
     downstream. Without an explicit policy, the SDK refuses with
     an actionable ``RUNTIME_UNAVAILABLE``.
  3. ``OCTOMIL_API_BASE`` is normalized so a stray ``/api/v1`` suffix
     doesn't double-prefix the planner URL.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.execution.kernel import (
    ExecutionKernel,
    _enforce_app_ref_routing_policy,
    _is_local_only_policy,
)
from octomil.runtime.core.policy import RoutingPolicy
from octomil.runtime.planner.planner import _normalize_api_base

# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("private", True),
        ("local_only", True),
        ("LOCAL_ONLY", True),  # case insensitive
        ("  private  ", True),  # whitespace stripped
        ("local_first", False),
        ("cloud_only", False),
        ("cloud_first", False),
        ("performance_first", False),
        (None, False),
        ("", False),
    ],
)
def test_is_local_only_policy(policy, expected):
    assert _is_local_only_policy(policy) is expected


def test_with_cloud_disabled_forces_local_only():
    p = RoutingPolicy.local_first(fallback="cloud")
    locked = p.with_cloud_disabled()
    from octomil._generated.routing_policy import RoutingPolicy as ContractRoutingPolicy

    assert locked.mode == ContractRoutingPolicy.LOCAL_ONLY
    assert locked.fallback == "none"


# ---------------------------------------------------------------------------
# @app ref refusal
# ---------------------------------------------------------------------------


def test_enforce_app_ref_routing_policy_raises_when_planner_failed_and_no_explicit_policy():
    """Reviewer P1: app ref + planner failure + no explicit policy →
    actionable RUNTIME_UNAVAILABLE so the SDK doesn't silently cloud."""
    with pytest.raises(OctomilError) as excinfo:
        _enforce_app_ref_routing_policy(
            requested_model="@app/eternum/tts",
            selection=None,
            explicit_policy=None,
        )
    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    msg = str(excinfo.value)
    assert "@app/eternum/tts" in msg
    assert "policy='local_only'" in msg or "policy=" in msg
    assert "planner" in msg.lower()


def test_enforce_app_ref_routing_policy_lets_explicit_local_only_through():
    """Caller has expressed intent — let the request proceed; the
    downstream local-only forcing handles cloud_available."""
    # No exception:
    _enforce_app_ref_routing_policy(
        requested_model="@app/eternum/tts",
        selection=None,
        explicit_policy="local_only",
    )
    _enforce_app_ref_routing_policy(
        requested_model="@app/eternum/tts",
        selection=None,
        explicit_policy="private",
    )


def test_enforce_app_ref_routing_policy_lets_explicit_cloud_through():
    """Cloud-allowing policies expressed explicitly are always honoured."""
    _enforce_app_ref_routing_policy(
        requested_model="@app/eternum/tts",
        selection=None,
        explicit_policy="cloud_first",
    )
    _enforce_app_ref_routing_policy(
        requested_model="@app/eternum/tts",
        selection=None,
        explicit_policy="performance_first",
    )


def test_enforce_app_ref_routing_policy_skips_non_app_refs():
    """Concrete model ids (no ``@app/`` prefix) skip the gate
    entirely — there's no app-side policy to consult."""
    _enforce_app_ref_routing_policy(
        requested_model="kokoro-en-v0_19",
        selection=None,
        explicit_policy=None,
    )


def test_enforce_app_ref_routing_policy_lets_planner_resolution_through():
    """When the planner returns a selection, even synthetic, the
    request has a planner-blessed path — no refusal needed."""

    class _Sel:
        candidates = []

    _enforce_app_ref_routing_policy(
        requested_model="@app/eternum/tts",
        selection=_Sel(),
        explicit_policy=None,
    )


# ---------------------------------------------------------------------------
# OCTOMIL_API_BASE normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "https://api.octomil.com"),
        ("", "https://api.octomil.com"),
        ("https://api.octomil.com", "https://api.octomil.com"),
        ("https://api.octomil.com/", "https://api.octomil.com"),
        ("https://api.octomil.com/api/v1", "https://api.octomil.com"),
        ("https://api.octomil.com/api/v2", "https://api.octomil.com"),
        ("https://api.octomil.com/api/v1/", "https://api.octomil.com"),
        ("https://api.octomil.com/v1", "https://api.octomil.com"),
        ("https://api.octomil.com/v2/", "https://api.octomil.com"),
        ("https://staging.octomil.com/api/v1", "https://staging.octomil.com"),
        # No version suffix → leave intact (caller may have set a
        # custom prefix path on purpose).
        ("https://gateway.example.com/octomil", "https://gateway.example.com/octomil"),
    ],
)
def test_normalize_api_base_strips_versioned_suffix(raw, expected):
    """Reviewer concern: setting ``OCTOMIL_API_BASE=…/api/v1`` makes
    the planner concatenate to ``…/api/v1/api/v2/runtime/plan``.
    Normalize so the planner client always appends its own versioned
    path cleanly."""
    assert _normalize_api_base(raw) == expected


# ---------------------------------------------------------------------------
# Facade kwargs flow through to the kernel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facade_speech_passes_policy_and_app_through():
    """Reviewer P1: the public ``client.audio.speech.create`` facade
    must accept ``policy=`` and ``app=`` and forward them to the
    kernel. Without this, embedded callers can't express their
    privacy requirement on the active TTS path."""
    from octomil.audio.speech import FacadeSpeech, SpeechResponse, SpeechRoute

    captured = {}

    class _StubKernel:
        async def synthesize_speech(self, **kwargs):
            captured.update(kwargs)
            return SpeechResponse(
                audio_bytes=b"",
                content_type="audio/wav",
                format="wav",
                model=kwargs.get("model", ""),
                provider=None,
                voice=None,
                sample_rate=None,
                duration_ms=None,
                latency_ms=0.0,
                route=SpeechRoute(locality="on_device", engine="sherpa-onnx", policy="local_only", fallback_used=False),
                billed_units=None,
                unit_kind=None,
            )

    speech = FacadeSpeech(_StubKernel())
    await speech.create(
        model="@app/eternum/tts",
        input="hello",
        policy="local_only",
        app="eternum",
    )
    assert captured["policy"] == "local_only"
    assert captured["app"] == "eternum"
    assert captured["model"] == "@app/eternum/tts"


# ---------------------------------------------------------------------------
# Kernel-level local-only forcing on chat dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_response_with_explicit_local_only_blocks_cloud(monkeypatch):
    """When the caller passes ``policy='local_only'``, even if cloud
    creds are present in the environment and the planner returns a
    cloud candidate, the runner must not see a cloud option."""
    kernel = ExecutionKernel()
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "kokoro-en-v0_19",
            "policy_preset": "local_only",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()

    captured = {"candidates": None}

    async def fake_build_router(model, capability, defaults, *, planner_selection=None, prepared_model_dir=None):
        # If a cloud candidate reaches here, we have a leak.
        loc = getattr(planner_selection, "locality", None)
        captured.setdefault("seen", []).append(loc)
        from octomil.runtime.core.types import RuntimeResponse, RuntimeUsage

        class _R:
            async def run(self, request, *, policy=None):
                return RuntimeResponse(
                    text="local",
                    usage=RuntimeUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                    finish_reason="stop",
                )

            async def stream(self, request, *, policy=None):
                yield None

        return _R()

    # Planner returns a cloud-only candidate; the dispatcher must
    # ignore it because policy='local_only' was explicit.
    from dataclasses import dataclass, field
    from typing import Any

    from octomil.runtime.planner.schemas import RuntimeCandidatePlan

    @dataclass
    class _Selection:
        candidates: list[RuntimeCandidatePlan] = field(default_factory=list)
        locality: str | None = None
        engine: str | None = None
        artifact: Any = None
        source: str | None = None
        fallback_allowed: bool = True
        reason: str = ""
        app_resolution: Any = None
        resolution: Any = None

    cloud_candidate = RuntimeCandidatePlan(locality="cloud", priority=0, confidence=0.9, reason="cloud-first")
    selection = _Selection(candidates=[cloud_candidate])

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection):
        with patch.object(kernel, "_build_router", side_effect=fake_build_router):
            with pytest.raises(OctomilError) as excinfo:
                await kernel.create_response("hi", model="kokoro-en-v0_19", policy="local_only")

    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    msg = str(excinfo.value)
    assert "local_runtime_unavailable" in msg
    assert "local_only" in msg
    # Cloud candidate must NOT have reached the runner.
    assert captured.get("seen", []) == []


# ---------------------------------------------------------------------------
# transcribe_audio refuses app-ref + planner outage without explicit policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcribe_audio_refuses_app_ref_when_planner_unavailable_and_no_policy():
    """Reviewer P1 applied to transcription: same gate as TTS / chat."""
    kernel = ExecutionKernel()
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "@app/notes/transcription",
            "policy_preset": "local_first",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()
    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=None):
        with pytest.raises(OctomilError) as excinfo:
            await kernel.transcribe_audio(
                audio_data=b"\x00",
                model="@app/notes/transcription",
                # NO explicit policy.
            )
    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    assert "@app/notes/transcription" in str(excinfo.value)
