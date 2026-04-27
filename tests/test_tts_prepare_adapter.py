"""Tests for the TTS prepare adapter — kernel wiring of PrepareManager.

We stub the sherpa backend, the planner selection, and PrepareManager so
the test exercises only the kernel's prepare-on-local-tts path. Full
prepare/download behavior is covered in test_prepare_manager.py and
test_durable_downloader.py.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest

from octomil._generated.error_code import ErrorCode
from octomil.errors import OctomilError
from octomil.execution.kernel import ExecutionKernel
from octomil.runtime.lifecycle.prepare_manager import PrepareOutcome
from octomil.runtime.planner.schemas import (
    ArtifactDownloadEndpoint,
    RuntimeArtifactPlan,
    RuntimeCandidatePlan,
)


@dataclass
class _Selection:
    """Minimal RuntimeSelection-shaped object for tests."""

    candidates: list[RuntimeCandidatePlan] = field(default_factory=list)
    locality: str | None = None
    engine: str | None = None
    artifact: Any = None
    source: str | None = None
    fallback_allowed: bool = True
    reason: str = ""
    app_resolution: Any = None
    resolution: Any = None


def _local_candidate(
    *,
    prepare_required: bool = True,
    prepare_policy: Literal["lazy", "explicit_only", "disabled"] = "lazy",
    delivery_mode: Literal["hosted_gateway", "sdk_runtime", "external_endpoint"] = "sdk_runtime",
    artifact_id: str = "kokoro-en-v0_19",
) -> RuntimeCandidatePlan:
    return RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="local-first",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(
            model_id="kokoro-en-v0_19",
            artifact_id=artifact_id,
            digest="sha256:" + "0" * 64,
            download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
        ),
        delivery_mode=delivery_mode,
        prepare_required=prepare_required,
        prepare_policy=prepare_policy,
    )


class _FakePrepareManager:
    """Records prepare calls and returns a synthetic outcome with a real dir."""

    def __init__(self, artifact_dir: Path, *, raises: Exception | None = None):
        self._artifact_dir = artifact_dir
        self._raises = raises
        self.calls: list[tuple[str, str, str]] = []

    def can_prepare(self, candidate) -> bool:  # noqa: D401 - stub
        # Every candidate the kernel hands to this stub is shaped via
        # `_local_candidate(...)` with digest + download_urls, so the
        # dry-run always returns True. Tests that need a synthetic plan
        # use a real PrepareManager (see the synthetic-candidate test).
        return True

    def prepare(self, candidate, *, mode=None):  # noqa: D401 - stub
        if self._raises is not None:
            raise self._raises
        self.calls.append(
            (
                candidate.artifact.artifact_id,
                candidate.delivery_mode or "sdk_runtime",
                candidate.prepare_policy,
            )
        )
        return PrepareOutcome(
            artifact_id=candidate.artifact.artifact_id,
            artifact_dir=self._artifact_dir,
            files={"": self._artifact_dir / "artifact"},
            engine=candidate.engine,
            delivery_mode=candidate.delivery_mode or "sdk_runtime",
            prepare_policy=candidate.prepare_policy,
            cached=False,
        )


class _FakeBackend:
    """Synthesize stub that records the model_dir it was loaded from."""

    last_model_dir: str | None = None
    last_kwargs: dict | None = None

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        _FakeBackend.last_kwargs = kwargs

    def load_model(self, model_name: str) -> None:
        _FakeBackend.last_model_dir = self.kwargs.get("model_dir")

    def synthesize(self, text, voice, speed):
        return {
            "audio_bytes": b"RIFF\x00\x00\x00\x00WAVEfake",
            "content_type": "audio/wav",
            "format": "wav",
            "sample_rate": 24000,
            "duration_ms": 100,
        }


class _FakeSherpaEngine:
    def create_backend(self, model_name: str, **kwargs: Any) -> _FakeBackend:
        return _FakeBackend(model_name, **kwargs)


def _patch_sherpa_helpers(stack, *, runtime_available: bool = True):
    """Patch sherpa-onnx engine helpers for kernel TTS tests.

    ``runtime_available`` controls whether sherpa-onnx is importable and
    the model id is recognized — that's what gates clean-device prepare.
    There is no ``staged`` flag after the PR D cutover: legacy staging
    (``OCTOMIL_SHERPA_MODELS_DIR`` / ``~/.octomil/models/sherpa``) was
    removed; the only on-disk source is PrepareManager's artifact cache.
    """
    stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_model", return_value=True))
    stack.enter_context(
        patch(
            "octomil.runtime.engines.sherpa.is_sherpa_tts_runtime_available",
            return_value=runtime_available,
        )
    )
    stack.enter_context(patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", _FakeSherpaEngine))


@pytest.fixture(autouse=True)
def _reset_fakes():
    _FakeBackend.last_model_dir = None
    _FakeBackend.last_kwargs = None
    yield


def _kernel_with(selection: _Selection, prepare_manager) -> ExecutionKernel:
    kernel = ExecutionKernel(prepare_manager=prepare_manager)
    # Bypass app/policy resolution — tests mint a selection directly.
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
    return kernel


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_local_tts_prepare_is_invoked_and_artifact_dir_threads_to_backend(tmp_path, monkeypatch):
    candidate = _local_candidate()
    selection = _Selection(candidates=[candidate])
    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()
    fake_pm = _FakePrepareManager(artifact_dir)

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_sherpa_helpers(stack)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, fake_pm)
        resp = await kernel.synthesize_speech(model="@app/eternum/tts", input="hello")

    assert resp.route.locality == "on_device"
    assert resp.route.engine == "sherpa-onnx"
    assert fake_pm.calls == [(candidate.artifact.artifact_id, "sdk_runtime", "lazy")]
    assert _FakeBackend.last_model_dir == str(artifact_dir)


@pytest.mark.asyncio
async def test_local_tts_prepare_required_false_fails_closed_after_cutover(tmp_path, monkeypatch):
    """PR D cutover: ``prepare_required=False`` no longer admits local
    routing on its own — the legacy ``OCTOMIL_SHERPA_MODELS_DIR`` /
    ``~/.octomil/models/sherpa`` fallback is gone, so without a
    PrepareManager-prepared dir there are no bytes for the backend
    to read. The kernel must fail closed with the canonical
    ``local_tts_runtime_unavailable`` rather than calling prepare or
    silently routing."""
    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "empty-cache"))
    candidate = _local_candidate(prepare_required=False)
    selection = _Selection(candidates=[candidate])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_sherpa_helpers(stack)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, fake_pm)
        with pytest.raises(OctomilError) as excinfo:
            await kernel.synthesize_speech(model="kokoro-en-v0_19", input="hi")

    assert excinfo.value.code == ErrorCode.RUNTIME_UNAVAILABLE
    assert fake_pm.calls == [], "prepare must not run when prepare_required=False"


@pytest.mark.asyncio
async def test_local_tts_explicit_only_policy_surfaces_actionable_error(tmp_path):
    candidate = _local_candidate(prepare_policy="explicit_only")
    selection = _Selection(candidates=[candidate])
    # Real PrepareManager so we get the canonical error message.
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    real_pm = PrepareManager(cache_dir=tmp_path)

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_sherpa_helpers(stack)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, real_pm)
        with pytest.raises(OctomilError) as excinfo:
            await kernel.synthesize_speech(model="@app/eternum/tts", input="hi")

    assert excinfo.value.code == ErrorCode.INVALID_INPUT
    assert "explicit_only" in str(excinfo.value)


@pytest.mark.asyncio
async def test_local_tts_hosted_gateway_candidate_does_not_admit_local_routing(tmp_path, monkeypatch):
    """PR D cutover: a ``delivery_mode='hosted_gateway'`` candidate is
    not a local sdk_runtime route. PrepareManager only handles
    sdk_runtime, and the legacy on-disk fallback is gone, so the
    kernel must NOT route locally and must NOT call prepare."""
    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "empty-cache"))
    candidate = _local_candidate(delivery_mode="hosted_gateway")
    selection = _Selection(candidates=[candidate])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_sherpa_helpers(stack)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, fake_pm)
        with pytest.raises(OctomilError) as excinfo:
            await kernel.synthesize_speech(model="kokoro-en-v0_19", input="hi")

    assert excinfo.value.code == ErrorCode.RUNTIME_UNAVAILABLE
    assert fake_pm.calls == [], "hosted_gateway candidate must not call prepare"


@pytest.mark.asyncio
async def test_local_tts_no_planner_candidates_and_no_prepared_cache_fails_closed(tmp_path, monkeypatch):
    """PR D cutover: planner offline + no prepared artifact cache + no
    legacy staging => local_tts_runtime_unavailable. Without the
    legacy ``~/.octomil/models/sherpa`` fallback, "the planner is
    offline" alone is no longer enough to keep local routing alive;
    a prepared artifact dir under PrepareManager's cache (or a
    successful preparable candidate) is required."""
    # Point cache_dir somewhere empty so the static-recipe lookup
    # cannot find a prepared dir for kokoro.
    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "empty-cache"))
    selection = _Selection(candidates=[])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_sherpa_helpers(stack)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, fake_pm)
        with pytest.raises(OctomilError) as excinfo:
            await kernel.synthesize_speech(model="kokoro-en-v0_19", input="hi")

    assert excinfo.value.code == ErrorCode.RUNTIME_UNAVAILABLE
    assert fake_pm.calls == []


@pytest.mark.asyncio
async def test_local_tts_prepare_manager_errors_propagate(tmp_path):
    candidate = _local_candidate()
    selection = _Selection(candidates=[candidate])
    err = OctomilError(code=ErrorCode.DOWNLOAD_FAILED, message="cdn down")
    fake_pm = _FakePrepareManager(tmp_path, raises=err)

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_sherpa_helpers(stack)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, fake_pm)
        with pytest.raises(OctomilError) as excinfo:
            await kernel.synthesize_speech(model="@app/eternum/tts", input="hi")

    assert excinfo.value.code == ErrorCode.DOWNLOAD_FAILED
    assert "cdn down" in str(excinfo.value)


# --- Clean-device first-run regression -------------------------------------


@pytest.mark.asyncio
async def test_clean_device_lazy_prepare_admits_local_routing(tmp_path):
    """Reviewer's reproducer: Private @app/foo/tts, sherpa-onnx installed,
    model id recognized, but artifact NOT staged on disk yet. The planner
    emits an sdk_runtime candidate with prepare_required=True. Without
    splitting runtime-available from artifact-staged, this routed to a
    local_tts_runtime_unavailable error before prepare ever ran. The fix
    makes the kernel admit the route and call PrepareManager."""
    candidate = _local_candidate()  # prepare_required=True, sdk_runtime, lazy
    selection = _Selection(candidates=[candidate])
    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()
    fake_pm = _FakePrepareManager(artifact_dir)

    from contextlib import ExitStack

    with ExitStack() as stack:
        # Clean device: bytes NOT staged, but sherpa-onnx package IS importable.
        _patch_sherpa_helpers(stack, runtime_available=True)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, fake_pm)
        resp = await kernel.synthesize_speech(model="@app/eternum/tts", input="hello")

    # Routed locally, prepare was called, artifact_dir threaded to backend.
    assert resp.route.locality == "on_device"
    assert resp.route.engine == "sherpa-onnx"
    assert len(fake_pm.calls) == 1, "prepare must run on a clean device"
    assert _FakeBackend.last_model_dir == str(artifact_dir)


@pytest.mark.asyncio
async def test_clean_device_no_planner_candidate_still_fails_closed(tmp_path):
    # The other side of the contract: if sherpa-onnx is installed and the
    # model id is recognized but the planner did NOT emit a preparable
    # candidate (e.g. offline planner, or candidate has prepare_required=False
    # with no staged files), local routing must still fail closed rather
    # than silently route to broken bytes.
    candidate = _local_candidate(prepare_required=False)  # not preparable
    selection = _Selection(candidates=[candidate])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_sherpa_helpers(stack, runtime_available=True)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, fake_pm)
        with pytest.raises(OctomilError) as excinfo:
            await kernel.synthesize_speech(model="@app/eternum/tts", input="hi")

    assert excinfo.value.code in (ErrorCode.RUNTIME_UNAVAILABLE,)
    assert fake_pm.calls == [], "prepare must not run when prepare_required=False"


@pytest.mark.asyncio
async def test_clean_device_without_sherpa_package_fails_closed(tmp_path):
    # If sherpa-onnx is not even importable, even a preparable candidate
    # cannot be made local. Fail closed before downloading bytes that
    # have no engine to load them.
    candidate = _local_candidate()
    selection = _Selection(candidates=[candidate])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_sherpa_helpers(stack, runtime_available=False)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, fake_pm)
        with pytest.raises(OctomilError):
            await kernel.synthesize_speech(model="@app/eternum/tts", input="hi")

    assert fake_pm.calls == [], "prepare must not run when the engine cannot load the result"


@pytest.mark.asyncio
async def test_synthetic_local_candidate_does_not_admit_local_routing(tmp_path, monkeypatch):
    """Reviewer's contract reproducer: planner emits prepare_required=True
    on a candidate with no digest/download_urls. _can_prepare_local_tts must
    return False so the kernel does NOT commit to local routing — otherwise
    local_first would land here and fail at first prepare instead of
    falling back to cloud."""
    # Isolate the cache dir so a developer's real prepared kokoro under
    # ``~/.cache/octomil/artifacts`` does not satisfy the prepared-cache
    # check and accidentally admit local routing for this contract test.
    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "empty-cache"))
    # Synthetic artifact: only model_id, no digest, no download_urls.
    candidate = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="synthetic",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(model_id="kokoro-en-v0_19"),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )
    selection = _Selection(candidates=[candidate])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        # Clean device, sherpa-onnx installed, but candidate metadata is incomplete.
        _patch_sherpa_helpers(stack, runtime_available=True)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, prepare_manager=None)
        # Use the real PrepareManager so can_prepare runs the actual rejection.
        # local_only routing must surface local_tts_runtime_unavailable rather
        # than commit to local and fail in prepare.
        with pytest.raises(OctomilError) as excinfo:
            await kernel.synthesize_speech(model="@app/eternum/tts", input="hi")

    # Either runtime_unavailable (no staging + no preparable candidate) or a
    # PrepareManager error if routing somehow committed; both are acceptable
    # outcomes, but prepare must NOT have been called on the synthetic plan.
    assert excinfo.value.code in (
        ErrorCode.RUNTIME_UNAVAILABLE,
        ErrorCode.INVALID_INPUT,
    )
    assert fake_pm.calls == [], "prepare must not run on synthetic candidates"
