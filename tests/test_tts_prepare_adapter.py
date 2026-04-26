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


def _patch_sherpa_helpers(stack):
    # Pretend the local TTS runtime is fully staged so kernel takes the local branch.
    stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_model_staged", return_value=True))
    stack.enter_context(patch("octomil.runtime.engines.sherpa.is_sherpa_tts_model", return_value=True))
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
async def test_local_tts_skips_prepare_when_prepare_required_false(tmp_path):
    candidate = _local_candidate(prepare_required=False)
    selection = _Selection(candidates=[candidate])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_sherpa_helpers(stack)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, fake_pm)
        await kernel.synthesize_speech(model="kokoro-en-v0_19", input="hi")

    assert fake_pm.calls == []
    # No prepared dir means the backend uses its env/home fallback (no model_dir kwarg).
    assert _FakeBackend.last_kwargs == {}


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
async def test_local_tts_skips_prepare_for_hosted_gateway_candidate(tmp_path):
    # Local candidate, but delivery_mode='hosted_gateway' (e.g. some niche
    # local-bridge case). Manager's job is sdk_runtime only, so prepare must
    # be skipped entirely; the kernel proceeds with whatever the backend
    # finds via its existing resolution path.
    candidate = _local_candidate(delivery_mode="hosted_gateway")
    selection = _Selection(candidates=[candidate])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_sherpa_helpers(stack)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, fake_pm)
        await kernel.synthesize_speech(model="kokoro-en-v0_19", input="hi")

    assert fake_pm.calls == [], "hosted_gateway candidate must not call prepare"
    assert _FakeBackend.last_kwargs == {}


@pytest.mark.asyncio
async def test_local_tts_no_planner_candidates_skips_prepare(tmp_path):
    # Planner returned no candidates (e.g. offline). Prepare adapter must
    # tolerate this and let the existing local resolution path run.
    selection = _Selection(candidates=[])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_sherpa_helpers(stack)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(selection, fake_pm)
        await kernel.synthesize_speech(model="kokoro-en-v0_19", input="hi")

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
