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
from typing import Any, Optional
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
    """Repeated lookups for the same candidate must hit the cache and
    call ``PrepareManager.prepare`` only once per ``create_response``."""
    candidate = _local_chat_candidate(engine="mlx-lm")
    artifact_dir = tmp_path / "gemma3-1b"
    artifact_dir.mkdir()
    pm = _RecordingPM(artifact_dir)
    kernel = ExecutionKernel(prepare_manager=pm)

    cache: dict[str, Any] = {}
    first = kernel._prepare_local_chat_artifact_cached(candidate, cache)
    second = kernel._prepare_local_chat_artifact_cached(candidate, cache)
    assert first == second == str(artifact_dir)
    assert pm.prepare_calls == ["gemma3-1b"]


# ---------------------------------------------------------------------------
# Reviewer P1 (multi-candidate): per-candidate filtering and per-candidate
# prepare. Plans like [bad local mlx, good local llama, cloud] must keep
# the good local candidate; ``fallback_allowed=False`` plans must not be
# silently promoted to cloud.
# ---------------------------------------------------------------------------


def _local_chat_candidate_with(
    engine: str,
    artifact_id: str,
    *,
    synthetic: bool = False,
) -> RuntimeCandidatePlan:
    """Local sdk_runtime candidate. ``synthetic=True`` strips digest/url
    so PrepareManager.can_prepare rejects it."""
    artifact_kwargs: dict[str, Any] = {"model_id": artifact_id, "artifact_id": artifact_id}
    if not synthetic:
        artifact_kwargs["digest"] = "sha256:" + "0" * 64
        artifact_kwargs["download_urls"] = [ArtifactDownloadEndpoint(url="https://cdn.example.com/")]
    return RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="planner",
        engine=engine,
        artifact=RuntimeArtifactPlan(**artifact_kwargs),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )


@pytest.mark.asyncio
async def test_filter_only_drops_individually_unpreparable_local_candidate(tmp_path):
    """Reviewer P1: the prior selection-wide filter dropped *all* local
    candidates whenever the first one was unpreparable. Plans like
    ``[synthetic local mlx, real local llama, cloud]`` lost the good
    llama fallback. The per-candidate filter must keep llama."""

    class _SelectivePM:
        """``can_prepare`` rejects synthetic candidates (no digest/url),
        accepts real ones; ``prepare`` succeeds for real ones."""

        def __init__(self, artifact_dir: Path) -> None:
            self._dir = artifact_dir
            self.can_prepare_calls: list[str] = []
            self.prepare_calls: list[str] = []

        def can_prepare(self, candidate) -> bool:
            artifact = candidate.artifact
            self.can_prepare_calls.append(artifact.artifact_id)
            return bool(getattr(artifact, "digest", None) and getattr(artifact, "download_urls", None))

        def prepare(self, candidate, *, mode=None):
            self.prepare_calls.append(candidate.artifact.artifact_id)
            return PrepareOutcome(
                artifact_id=candidate.artifact.artifact_id,
                artifact_dir=self._dir,
                files={"": self._dir / "artifact"},
                engine=candidate.engine,
                delivery_mode="sdk_runtime",
                prepare_policy="lazy",
                cached=False,
            )

    artifact_dir = tmp_path / "qwen-2-1b"
    artifact_dir.mkdir()
    pm = _SelectivePM(artifact_dir)

    bad_mlx = _local_chat_candidate_with("mlx-lm", "gemma3-1b", synthetic=True)
    good_llama = _local_chat_candidate_with("llama.cpp", "qwen-2-1b")
    cloud = _cloud_candidate()
    selection = _Selection(candidates=[bad_mlx, good_llama, cloud], fallback_allowed=True)
    kernel = ExecutionKernel(prepare_manager=pm)
    kernel._resolve = lambda capability, **kw: _make_defaults()

    captured: list[dict[str, Any]] = []

    async def capturing_build_router(*args, **kwargs):
        captured.append(kwargs)
        # llama attempt fails at routing so we can observe that cloud
        # is a true fallback and not the silent winner.
        raise RuntimeError("llama unavailable in test")

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch.object(kernel, "_build_router", side_effect=capturing_build_router))
        with pytest.raises(Exception):
            await kernel.create_response("Hello", model="gemma3-1b")

    localities = [getattr(call.get("planner_selection"), "locality", None) for call in captured]
    # First attempt: the surviving local llama (bad mlx filtered out).
    # Second: cloud as fallback.
    assert localities[:2] == ["local", "cloud"], f"expected llama then cloud, got {localities!r} (full: {captured!r})"
    assert captured[0].get("prepared_model_dir") == str(artifact_dir)
    assert pm.prepare_calls == ["qwen-2-1b"], f"expected only llama prepared, got {pm.prepare_calls!r}"


@pytest.mark.asyncio
async def test_filter_returns_empty_when_primary_unpreparable_and_fallback_disallowed(tmp_path):
    """Reviewer P1: when ``fallback_allowed=False`` and the primary
    candidate is unpreparable, the helper must not silently promote
    cloud (or any other remaining candidate). Returns an empty list so
    the runner raises its actionable "no runnable candidate" error."""
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    bad_local = _local_chat_candidate_with("mlx-lm", "gemma3-1b", synthetic=True)
    cloud = _cloud_candidate()
    selection = _Selection(candidates=[bad_local, cloud], fallback_allowed=False)
    real_pm = PrepareManager(cache_dir=tmp_path)
    kernel = ExecutionKernel(prepare_manager=real_pm)
    kernel._resolve = lambda capability, **kw: _make_defaults()

    captured: list[dict[str, Any]] = []

    async def capturing_build_router(*args, **kwargs):
        captured.append(kwargs)
        return _StubRouter()

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch.object(kernel, "_build_router", side_effect=capturing_build_router))
        with pytest.raises(Exception):
            await kernel.create_response("Hello", model="gemma3-1b")

    assert captured == [], (
        "fallback_allowed=False with unpreparable primary must surface the "
        f"runner's no-candidate error, not promote cloud. Got attempts: {captured!r}"
    )


@pytest.mark.asyncio
async def test_each_local_attempt_prepares_its_own_candidate(tmp_path):
    """Reviewer P1: the lazy helper used to prepare the first local
    candidate of the whole selection regardless of which attempt was
    running, so a [local mlx, local llama] plan would prepare the mlx
    artifact for both attempts. The candidate-specific helper must
    prepare each candidate's own artifact."""
    artifact_a = tmp_path / "gemma3-1b"
    artifact_a.mkdir()
    artifact_b = tmp_path / "qwen-2-1b"
    artifact_b.mkdir()

    class _MultiArtifactPM:
        def __init__(self) -> None:
            self.prepare_calls: list[str] = []

        def can_prepare(self, candidate) -> bool:
            return True

        def prepare(self, candidate, *, mode=None):
            artifact_id = candidate.artifact.artifact_id
            self.prepare_calls.append(artifact_id)
            target = artifact_a if artifact_id == "gemma3-1b" else artifact_b
            return PrepareOutcome(
                artifact_id=artifact_id,
                artifact_dir=target,
                files={"": target / "artifact"},
                engine=candidate.engine,
                delivery_mode="sdk_runtime",
                prepare_policy="lazy",
                cached=False,
            )

    pm = _MultiArtifactPM()
    candidate_a = _local_chat_candidate_with("mlx-lm", "gemma3-1b")
    candidate_b = _local_chat_candidate_with("llama.cpp", "qwen-2-1b")
    selection = _Selection(candidates=[candidate_a, candidate_b], fallback_allowed=True)
    kernel = ExecutionKernel(prepare_manager=pm)
    kernel._resolve = lambda capability, **kw: _make_defaults()

    captured_dirs: list[Optional[str]] = []

    async def capturing_build_router(*args, **kwargs):
        captured_dirs.append(kwargs.get("prepared_model_dir"))
        # First attempt fails so the runner moves to the second local.
        if len(captured_dirs) == 1:
            raise RuntimeError("first local attempt failed")
        return _StubRouter()

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch.object(kernel, "_build_router", side_effect=capturing_build_router))
        await kernel.create_response("Hello", model="gemma3-1b")

    assert captured_dirs == [
        str(artifact_a),
        str(artifact_b),
    ], f"each local attempt must prepare its own artifact; got {captured_dirs!r}"
    assert pm.prepare_calls == ["gemma3-1b", "qwen-2-1b"]


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
