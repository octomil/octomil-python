"""End-to-end tests for PR 10c: chat dispatch threading.

PR 10c wires ``model_dir`` from PrepareManager through the chat
dispatch path into ``MLXBackend`` and ``LlamaCppBackend``. The
*capability* (``chat`` / ``responses``) remains intentionally gated in
``_PREPAREABLE_CAPABILITIES`` because:

  - the public ``client.responses.create`` facade goes through
    ``OctomilResponses``, which does NOT thread ``model_dir``;
  - PrepareManager has no snapshot/manifest support, so MLX (which
    needs a directory with ``config.json`` + tokenizer + weights)
    cannot consume a prepared dir today even though
    ``mlx_lm.load(<dir>)`` is wired.

These tests therefore assert the *threading* contract for the kernel
path (the part this PR ships), the prepare-ordering invariants the
reviewer asked for, and that the public ``prepare(capability='chat')`` /
``prepare(capability='responses')`` surface still rejects with an
actionable error so we don't false-success the unfinished pipeline.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.execution.kernel import ExecutionKernel
from octomil.runtime.lifecycle.prepare_manager import PrepareOutcome
from octomil.runtime.planner.schemas import (
    ArtifactDownloadEndpoint,
    RuntimeArtifactPlan,
    RuntimeCandidatePlan,
)
from octomil.serve.types import InferenceMetrics


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


def _local_chat_candidate(engine: str = "mlx-lm") -> RuntimeCandidatePlan:
    return RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="local-first",
        engine=engine,
        artifact=RuntimeArtifactPlan(
            model_id="gemma3-1b",
            artifact_id="gemma3-1b",
            digest="sha256:" + "0" * 64,
            download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
        ),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )


def _cloud_candidate() -> RuntimeCandidatePlan:
    return RuntimeCandidatePlan(
        locality="cloud",
        priority=0,
        confidence=0.9,
        reason="cloud-first",
    )


class _RecordingPM:
    """Records every ``prepare()`` call so tests can prove cloud-first
    plans don't trigger local prepare."""

    def __init__(self, artifact_dir: Path):
        self._dir = artifact_dir
        self.prepare_calls: list[str] = []

    def can_prepare(self, candidate) -> bool:
        return True

    def prepare(self, candidate, *, mode=None):
        self.prepare_calls.append(candidate.artifact.artifact_id)
        return PrepareOutcome(
            artifact_id=candidate.artifact.artifact_id,
            artifact_dir=self._dir,
            files={"": self._dir / "artifact"},
            engine=candidate.engine,
            delivery_mode=candidate.delivery_mode or "sdk_runtime",
            prepare_policy=candidate.prepare_policy,
            cached=False,
        )


class _FakeBackend:
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def generate(self, request: Any) -> tuple[str, InferenceMetrics]:
        return (
            "from prepared dir",
            InferenceMetrics(
                prompt_tokens=2,
                total_tokens=10,
                tokens_per_second=42.0,
                total_duration_ms=1.0,
            ),
        )


class _FakeEngine:
    """Stand-in for MLXEngine / LlamaCppEngine. Records ``model_dir``."""

    name = "mlx-lm"
    last_kwargs: dict[str, Any] | None = None
    last_model_dir: str | None = None
    last_model: str | None = None
    create_calls: int = 0

    def detect(self) -> bool:
        return True

    def create_backend(self, model: str, **kwargs: Any) -> _FakeBackend:
        _FakeEngine.last_kwargs = kwargs
        _FakeEngine.last_model_dir = kwargs.get("model_dir")
        _FakeEngine.last_model = model
        _FakeEngine.create_calls += 1
        return _FakeBackend(model, **kwargs)


class _FakeEngineRegistry:
    def get_engine(self, name: str) -> Any:
        if name == "mlx-lm":
            return _FakeEngine()
        return None


@pytest.fixture(autouse=True)
def _reset_fakes():
    _FakeEngine.last_kwargs = None
    _FakeEngine.last_model_dir = None
    _FakeEngine.last_model = None
    _FakeEngine.create_calls = 0
    yield


def _make_defaults():
    return type(
        "_D",
        (),
        {
            "model": "gemma3-1b",
            "policy_preset": "local_first",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()


# ---------------------------------------------------------------------------
# Threading: when a local sdk_runtime candidate runs, it gets the prepared dir
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_chat_attempt_threads_artifact_dir_into_backend(tmp_path):
    """Direction-of-travel test: when the kernel selects a local
    sdk_runtime candidate, ``engine.create_backend`` is called with
    ``model_dir=<prepared_dir>``. The capability remains gated in
    ``_PREPAREABLE_CAPABILITIES`` until the public responses facade and
    MLX snapshot materialization are in, but the threading itself is
    pinned by this test so it doesn't regress before that flip lands."""
    candidate = _local_chat_candidate(engine="mlx-lm")
    selection = _Selection(candidates=[candidate])
    artifact_dir = tmp_path / "gemma3-1b"
    artifact_dir.mkdir()
    (artifact_dir / "artifact").write_bytes(b"fake mlx weights")
    pm = _RecordingPM(artifact_dir)
    kernel = ExecutionKernel(prepare_manager=pm)
    kernel._resolve = lambda capability, **kw: _make_defaults()

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch("octomil.runtime.engines.get_registry", return_value=_FakeEngineRegistry()))
        result = await kernel.create_response("Hello", model="gemma3-1b")

    assert _FakeEngine.last_model_dir == str(artifact_dir)
    assert _FakeEngine.last_model == "gemma3-1b"
    assert result.output_text == "from prepared dir"
    assert result.locality == "on_device"
    # Lazy prepare: invoked exactly once for the single local attempt.
    assert pm.prepare_calls == ["gemma3-1b"]


# ---------------------------------------------------------------------------
# Reviewer P1 #1: cloud-first plans must NOT call prepare
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cloud_first_plan_does_not_prepare_local_fallback(tmp_path):
    """Reviewer P1: with a cloud primary plus a preparable local
    fallback, the kernel must dispatch the cloud attempt without
    calling ``PrepareManager.prepare`` at all. The prior implementation
    materialized the local artifact before the cloud branch ran;
    prepare now lives inside the local candidate attempt."""
    cloud = _cloud_candidate()
    local_fallback = _local_chat_candidate(engine="mlx-lm")
    selection = _Selection(candidates=[cloud, local_fallback])
    artifact_dir = tmp_path / "gemma3-1b"
    artifact_dir.mkdir()
    pm = _RecordingPM(artifact_dir)
    kernel = ExecutionKernel(prepare_manager=pm)
    kernel._resolve = lambda capability, **kw: _make_defaults()

    async def fake_build_router(model, capability, defaults, *, planner_selection=None, prepared_model_dir=None):
        return _StubRouter(prepared_model_dir=prepared_model_dir)

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch.object(kernel, "_build_router", side_effect=fake_build_router))
        result = await kernel.create_response("Hello", model="gemma3-1b")

    assert pm.prepare_calls == [], (
        f"Cloud-first plan triggered local prepare: {pm.prepare_calls!r}. "
        "Prepare must run inside the local candidate branch only."
    )
    assert result.output_text == "from cloud"


# ---------------------------------------------------------------------------
# Reviewer P1 #2: synthetic local candidates must be filtered (hard veto)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthetic_local_candidate_is_filtered_before_runner(tmp_path):
    """Reviewer P1: a synthetic prepare_required local candidate
    (missing digest/url) must NOT reach the runner. Previously the
    helper returned ``None`` and the runner still called
    ``_build_router`` with no model_dir, letting the engine fall back
    to its own HF download path — synthetic plans could win local
    routing. The kernel now filters the unpreparable local candidate
    so cloud-fallback is the actual selected branch."""
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    synthetic = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="synthetic",
        engine="mlx-lm",
        artifact=RuntimeArtifactPlan(model_id="gemma3-1b"),  # no digest/url
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )
    cloud = _cloud_candidate()
    selection = _Selection(candidates=[synthetic, cloud])
    real_pm = PrepareManager(cache_dir=tmp_path)
    kernel = ExecutionKernel(prepare_manager=real_pm)
    kernel._resolve = lambda capability, **kw: _make_defaults()

    captured: list[dict[str, Any]] = []

    async def capturing_build_router(*args, **kwargs):
        captured.append({"args": args, "kwargs": kwargs})
        return _StubRouter()

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch.object(kernel, "_build_router", side_effect=capturing_build_router))
        await kernel.create_response("Hello", model="gemma3-1b")

    # Unpreparable local candidate is filtered; only cloud reaches the runner.
    assert len(captured) == 1
    only_call = captured[0]
    selected_candidate = only_call["kwargs"].get("planner_selection")
    assert selected_candidate is not None
    assert getattr(selected_candidate, "locality", None) == "cloud"
    assert only_call["kwargs"].get("prepared_model_dir") is None


# ---------------------------------------------------------------------------
# Capability gate: prepare(capability='chat'/'responses') stays rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capability", ["chat", "responses"])
def test_prepare_rejects_chat_and_responses_until_facade_and_mlx_ready(capability, tmp_path):
    """Until the public ``client.responses.create`` facade routes
    through the kernel (or threads ``model_dir`` through OctomilResponses)
    AND PrepareManager grows snapshot/manifest materialization for
    multi-file MLX artifacts, ``client.prepare(capability='chat')`` /
    ``prepare(capability='responses')`` must reject with INVALID_INPUT.
    This regression-pins the gate so the next attempt to widen
    ``_PREPAREABLE_CAPABILITIES`` cannot ship without removing this
    test (and adding the e2e proofs for both blockers)."""
    kernel = ExecutionKernel()
    with pytest.raises(OctomilError) as excinfo:
        kernel.prepare(model="m", capability=capability)
    assert excinfo.value.code == OctomilErrorCode.INVALID_INPUT
    msg = str(excinfo.value)
    assert capability in msg


def test_preparable_capabilities_excludes_chat_and_responses():
    """Direct invariant on the kernel's allowlist."""
    from octomil.execution.kernel import _PREPAREABLE_CAPABILITIES

    assert "chat" not in _PREPAREABLE_CAPABILITIES
    assert "responses" not in _PREPAREABLE_CAPABILITIES
    assert "tts" in _PREPAREABLE_CAPABILITIES
    assert "transcription" in _PREPAREABLE_CAPABILITIES


# ---------------------------------------------------------------------------
# Lazy-prepare cache: repeated local attempts don't re-prepare
# ---------------------------------------------------------------------------


def test_local_prepare_cache_skips_redundant_calls_within_one_request(tmp_path):
    """If two local attempts point at the same artifact_id, the cached
    helper must call ``PrepareManager.prepare`` only once per
    ``create_response`` call. Future fallback retries (or planner
    selections that emit the same local candidate twice) shouldn't
    re-download."""
    candidate = _local_chat_candidate(engine="mlx-lm")
    artifact_dir = tmp_path / "gemma3-1b"
    artifact_dir.mkdir()
    pm = _RecordingPM(artifact_dir)
    kernel = ExecutionKernel(prepare_manager=pm)

    cache: dict[str, Any] = {}
    candidate_selection = _Selection(candidates=[candidate], locality="local", engine="mlx-lm")
    first = kernel._prepare_local_chat_artifact_cached(candidate_selection, cache)
    second = kernel._prepare_local_chat_artifact_cached(candidate_selection, cache)
    assert first == second == str(artifact_dir)
    assert pm.prepare_calls == ["gemma3-1b"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubRouter:
    """Minimal RouterModelRuntime stub for kernel-path tests."""

    def __init__(self, prepared_model_dir: str | None = None) -> None:
        self.prepared_model_dir = prepared_model_dir

    async def run(self, request: Any, *, policy: Any = None) -> Any:
        from octomil.runtime.core.types import RuntimeResponse, RuntimeUsage

        text = "stub"
        if policy is not None:
            mode = getattr(policy, "mode", None)
            if mode is not None and "cloud" in str(mode).lower():
                text = "from cloud"
        return RuntimeResponse(
            text=text,
            usage=RuntimeUsage(prompt_tokens=2, completion_tokens=8, total_tokens=10),
            finish_reason="stop",
        )

    async def stream(self, request: Any, *, policy: Any = None):
        from octomil.runtime.core.types import RuntimeChunk

        yield RuntimeChunk(text="stub", finish_reason=None)
        yield RuntimeChunk(text="", finish_reason="stop")
