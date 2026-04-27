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

from typing import Any
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


# ---------------------------------------------------------------------------
# Reviewer P1: ``policy='local_only'`` must not be rejected by preset
# normalization
# ---------------------------------------------------------------------------


def test_local_only_preset_is_accepted_by_resolve_capability_defaults():
    """The TTS facade documents ``policy='local_only'`` as accepted.
    ``resolve_capability_defaults`` calls ``_normalise_preset``,
    whose VALID_PRESETS set must include it — otherwise the public
    ``client.audio.speech.create(..., policy='local_only')`` call
    raises ``ValueError`` before any of the cloud-disabled routing
    even gets a chance to run."""
    from octomil.config.local import (
        VALID_PRESETS,
        RequestOverrides,
        load_standalone_config,
        resolve_capability_defaults,
    )

    assert "local_only" in VALID_PRESETS
    cfg = load_standalone_config()
    overrides = RequestOverrides(model="kokoro-en-v0_19", policy="local_only")
    defaults = resolve_capability_defaults("tts", overrides, cfg)
    assert defaults.policy_preset == "local_only"


@pytest.mark.asyncio
async def test_facade_speech_create_accepts_local_only_policy(monkeypatch, tmp_path):
    """End-to-end smoke (no stubbing of ``_resolve``): the public
    ``client.audio.speech.create(..., policy='local_only')`` path
    must not raise during preset normalization."""
    from octomil.audio.speech import FacadeSpeech
    from octomil.execution.kernel import ExecutionKernel

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    kernel = ExecutionKernel()
    speech = FacadeSpeech(kernel)

    try:
        await speech.create(
            model="kokoro-en-v0_19",
            input="hello",
            policy="local_only",
        )
    except ValueError as exc:  # pragma: no cover - regression marker
        pytest.fail(f"local_only must not be rejected by preset normalization: {exc}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Reviewer P1: ``app=`` must reach planner routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_kwarg_reaches_planner_via_synthesized_app_ref():
    """Reviewer P1: the public facade exposes ``app=``, but the
    kernel previously only stored it on ``ResolvedExecutionDefaults``;
    planner resolution still got the bare model id and the app's
    routing policy never applied. The fix synthesizes
    ``@app/<app>/<capability>`` as the planner-facing model so the
    existing app-ref machinery does the rest."""
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel()
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "kokoro-en-v0_19",
            "policy_preset": "local_first",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()

    captured: dict[str, Any] = {}

    def fake_resolve_planner_selection(model, capability, policy_preset):
        captured["planner_model"] = model
        captured["capability"] = capability
        return None

    with patch(
        "octomil.execution.kernel._resolve_planner_selection",
        side_effect=fake_resolve_planner_selection,
    ):
        with pytest.raises(OctomilError) as excinfo:
            await kernel.synthesize_speech(
                model="kokoro-en-v0_19",
                input="hi",
                app="eternum",
            )

    assert captured["planner_model"] == "@app/eternum/tts", captured
    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    assert "eternum" in str(excinfo.value)


def test_planner_model_helper_synthesizes_app_ref():
    from octomil.execution.kernel import _planner_model_for_request

    assert _planner_model_for_request(effective_model="kokoro-82m", app=None, capability="tts") == "kokoro-82m"
    assert (
        _planner_model_for_request(effective_model="@app/notes/tts", app="other", capability="tts") == "@app/notes/tts"
    )
    assert (
        _planner_model_for_request(effective_model="kokoro-82m", app="eternum", capability="tts") == "@app/eternum/tts"
    )
    assert _planner_model_for_request(effective_model="kokoro-82m", app="", capability="tts") == "kokoro-82m"


# ---------------------------------------------------------------------------
# Reviewer P1: embeddings must enforce the same gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embeddings_refuses_app_scoped_request_when_planner_unavailable_and_no_policy():
    """Reviewer P1: ``create_embeddings`` was the lone surface where
    ``model='nomic-embed', app='private-app', policy=None`` plus a
    planner outage could still silently route to cloud, recreating
    the private-app DX/security bug fixed elsewhere. Same refusal
    gate must apply."""
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel()
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "nomic-embed-text-v1.5",
            "policy_preset": "local_first",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=None):
        with pytest.raises(OctomilError) as excinfo:
            await kernel.create_embeddings(
                ["hello"],
                model="nomic-embed-text-v1.5",
                app="private-app",
            )

    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    msg = str(excinfo.value)
    assert "private-app" in msg
    assert "nomic-embed-text-v1.5" in msg


@pytest.mark.asyncio
async def test_embeddings_explicit_local_only_disables_cloud():
    """When the caller passes ``policy='local_only'``, embeddings
    must not promote to cloud even if cloud creds are present.
    This mirrors the TTS / chat / transcription gating."""
    from octomil.execution.kernel import ExecutionKernel

    class _StubCloudProfile:
        api_key_env = "STUB_API_KEY"
        api_base = "https://stub.example.com"

    kernel = ExecutionKernel()
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "nomic-embed-text-v1.5",
            "policy_preset": "local_only",
            "inline_policy": None,
            "cloud_profile": _StubCloudProfile(),
        },
    )()
    # Pretend the local backend is unavailable so the dispatcher
    # would normally fall back to cloud — but local_only must block
    # that and surface the local-unavailable error.
    kernel._can_local = lambda model, capability: False  # type: ignore[method-assign]

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=None):
        with pytest.raises(Exception) as excinfo:
            await kernel.create_embeddings(
                ["hello"],
                model="nomic-embed-text-v1.5",
                policy="local_only",
            )

    # Must not be a cloud request — the error indicates local
    # unavailability, NOT a successful cloud dispatch.
    msg = str(excinfo.value).lower()
    assert "local" in msg or "unavailable" in msg


# ---------------------------------------------------------------------------
# Reviewer P1: cloud dispatch must run under the app identity
# ---------------------------------------------------------------------------


def test_execution_model_for_cloud_dispatch_prefers_app_ref():
    from octomil.execution.kernel import _execution_model_for_cloud_dispatch

    # No app, concrete model → resolved model.
    assert (
        _execution_model_for_cloud_dispatch(
            requested_model="kokoro-82m",
            effective_model="kokoro-en-v0_19",
            planner_model="kokoro-82m",
            app=None,
        )
        == "kokoro-en-v0_19"
    )
    # Explicit @app/ in the request → preserve as-is.
    assert (
        _execution_model_for_cloud_dispatch(
            requested_model="@app/eternum/tts",
            effective_model="kokoro-en-v0_19",
            planner_model="@app/eternum/tts",
            app=None,
        )
        == "@app/eternum/tts"
    )
    # Concrete model + ``app=`` → synthesized app ref.
    assert (
        _execution_model_for_cloud_dispatch(
            requested_model="kokoro-82m",
            effective_model="kokoro-en-v0_19",
            planner_model="@app/eternum/tts",
            app="eternum",
        )
        == "@app/eternum/tts"
    )


@pytest.mark.asyncio
async def test_chat_cloud_dispatch_uses_app_ref_not_resolved_model():
    """Reviewer P1: when chat routes to cloud and ``app=`` was
    explicit, the hosted call must go out under the synthesized
    ``@app/<app>/responses`` identity, not the resolved underlying
    model id. Otherwise server-side quota / billing / policy can't
    be applied."""
    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel()
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "gpt-4o-mini",
            "policy_preset": "cloud_first",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()

    captured_kwargs: dict[str, Any] = {}

    async def fake_build_router(
        model,
        capability,
        defaults,
        *,
        planner_selection=None,
        prepared_model_dir=None,
        cloud_execution_model=None,
    ):
        captured_kwargs["cloud_execution_model"] = cloud_execution_model
        captured_kwargs["model"] = model

        from octomil.runtime.core.types import RuntimeResponse, RuntimeUsage

        class _R:
            async def run(self, request, *, policy=None):
                return RuntimeResponse(
                    text="ok",
                    usage=RuntimeUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                    finish_reason="stop",
                )

            async def stream(self, request, *, policy=None):
                yield None

        return _R()

    # Cloud-only candidate so the runner picks cloud immediately.
    from dataclasses import dataclass, field

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

    cloud_only = RuntimeCandidatePlan(locality="cloud", priority=0, confidence=0.9, reason="cloud")
    selection = _Selection(candidates=[cloud_only])

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection):
        with patch.object(kernel, "_build_router", side_effect=fake_build_router):
            await kernel.create_response("hi", model="kokoro-82m", app="eternum")

    # Local resolution still uses the concrete model id (for engine
    # registry lookup if local were chosen); cloud dispatch sees the
    # app ref.
    assert captured_kwargs["cloud_execution_model"] == "@app/eternum/chat", captured_kwargs


def test_app_kwarg_triggers_refusal_gate_even_when_model_is_concrete():
    """Reviewer P1: when the caller passes ``app=`` (without an
    ``@app/`` prefix in ``model=``) and the planner returns no
    selection, the refusal gate must still fire — the request IS
    app-scoped."""
    from octomil.execution.kernel import _enforce_app_ref_routing_policy

    with pytest.raises(OctomilError) as excinfo:
        _enforce_app_ref_routing_policy(
            requested_model="kokoro-en-v0_19",
            selection=None,
            explicit_policy=None,
            explicit_app="eternum",
        )
    assert excinfo.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE
    msg = str(excinfo.value)
    assert "kokoro-en-v0_19" in msg
    assert "eternum" in msg


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
