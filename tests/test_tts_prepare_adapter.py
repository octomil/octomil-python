"""Tests for native TTS cutover in the kernel route.

The legacy Python Sherpa path prepared an artifact directory and
threaded it into ``SherpaTtsEngine``. Native cutover makes
``audio.tts.batch`` advertisement the local-routing truth instead:
the runtime either advertises the capability and the SDK opens a
native backend, or the route fails closed without running
PrepareManager.
"""

from __future__ import annotations

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
    artifact_id: str = "piper-en-amy",
) -> RuntimeCandidatePlan:
    return RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="local-first",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(
            model_id="piper-en-amy",
            artifact_id=artifact_id,
            digest="sha256:" + "0" * 64,
            download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
        ),
        delivery_mode=delivery_mode,
        prepare_required=prepare_required,
        prepare_policy=prepare_policy,
    )


class _FakePrepareManager:
    """Records prepare calls; native TTS should not invoke it."""

    def __init__(self, artifact_dir: Path, *, raises: Exception | None = None):
        self._artifact_dir = artifact_dir
        self._raises = raises
        self.calls: list[tuple[str, str, str]] = []

    def can_prepare(self, candidate) -> bool:  # noqa: D401 - stub
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


class _FakeNativeTtsBatchBackend:
    supported_model_names = frozenset({"piper-en-amy", "piper-amy"})
    last_loaded_model: str | None = None
    synthesize_calls: list[tuple[str, str | None, float]] = []

    def load_model(self, model_name: str) -> None:
        self.__class__.last_loaded_model = model_name

    def synthesize(self, text: str, voice: str | None = None, speed: float = 1.0, **kwargs: Any):
        self.__class__.synthesize_calls.append((text, voice, speed))
        return {
            "audio_bytes": b"RIFF\x00\x00\x00\x00WAVEfake",
            "content_type": "audio/wav",
            "format": "wav",
            "model": self.last_loaded_model or "piper-en-amy",
            "voice": voice or "0",
            "sample_rate": 22050,
            "duration_ms": 100,
        }


class _FailingNativeTtsBatchBackend(_FakeNativeTtsBatchBackend):
    def load_model(self, model_name: str) -> None:
        raise OctomilError(code=ErrorCode.RUNTIME_UNAVAILABLE, message="native load failed")


def _patch_native_tts(stack, *, runtime_available: bool = True, backend_cls=_FakeNativeTtsBatchBackend):
    stack.enter_context(patch.object(ExecutionKernel, "_sherpa_tts_runtime_loadable", return_value=runtime_available))
    stack.enter_context(patch("octomil.runtime.native.tts_batch_backend.NativeTtsBatchBackend", backend_cls))


@pytest.fixture(autouse=True)
def _reset_fakes():
    _FakeNativeTtsBatchBackend.last_loaded_model = None
    _FakeNativeTtsBatchBackend.synthesize_calls = []
    yield


def _kernel_with(prepare_manager) -> ExecutionKernel:
    kernel = ExecutionKernel(prepare_manager=prepare_manager)
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": "piper-en-amy",
            "policy_preset": "local_first",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()
    return kernel


@pytest.mark.asyncio
async def test_native_tts_probe_admits_local_route_without_prepare(tmp_path):
    candidate = _local_candidate()
    selection = _Selection(candidates=[candidate])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_native_tts(stack, runtime_available=True)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(fake_pm)
        resp = await kernel.synthesize_speech(model="@app/eternum/tts", input="hello")

    assert resp.route.locality == "on_device"
    assert resp.route.engine == "sherpa-onnx"
    assert fake_pm.calls == []
    assert _FakeNativeTtsBatchBackend.last_loaded_model == "piper-en-amy"
    assert _FakeNativeTtsBatchBackend.synthesize_calls == [("hello", None, 1.0)]


@pytest.mark.asyncio
async def test_native_tts_unavailable_fails_closed_without_prepare(tmp_path):
    selection = _Selection(candidates=[_local_candidate()])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_native_tts(stack, runtime_available=False)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(fake_pm)
        with pytest.raises(OctomilError) as excinfo:
            await kernel.synthesize_speech(model="@app/eternum/tts", input="hi")

    assert excinfo.value.code == ErrorCode.RUNTIME_UNAVAILABLE
    assert fake_pm.calls == []


@pytest.mark.asyncio
async def test_prepare_required_false_does_not_block_advertised_native_runtime(tmp_path):
    selection = _Selection(candidates=[_local_candidate(prepare_required=False)])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_native_tts(stack, runtime_available=True)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(fake_pm)
        resp = await kernel.synthesize_speech(model="piper-en-amy", input="hi")

    assert resp.route.locality == "on_device"
    assert fake_pm.calls == []


@pytest.mark.asyncio
async def test_explicit_only_policy_no_longer_calls_prepare_on_native_path(tmp_path):
    selection = _Selection(candidates=[_local_candidate(prepare_policy="explicit_only")])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_native_tts(stack, runtime_available=True)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(fake_pm)
        resp = await kernel.synthesize_speech(model="@app/eternum/tts", input="hi")

    assert resp.route.locality == "on_device"
    assert fake_pm.calls == []


@pytest.mark.asyncio
async def test_prepare_manager_errors_do_not_surface_when_native_is_advertised(tmp_path):
    selection = _Selection(candidates=[_local_candidate()])
    err = OctomilError(code=ErrorCode.DOWNLOAD_FAILED, message="cdn down")
    fake_pm = _FakePrepareManager(tmp_path / "never-used", raises=err)

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_native_tts(stack, runtime_available=True)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(fake_pm)
        resp = await kernel.synthesize_speech(model="@app/eternum/tts", input="hi")

    assert resp.route.locality == "on_device"
    assert fake_pm.calls == []


@pytest.mark.asyncio
async def test_native_backend_load_error_surfaces_runtime_unavailable(tmp_path):
    selection = _Selection(candidates=[_local_candidate()])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_native_tts(stack, runtime_available=True, backend_cls=_FailingNativeTtsBatchBackend)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(fake_pm)
        with pytest.raises(OctomilError) as excinfo:
            await kernel.synthesize_speech(model="@app/eternum/tts", input="hi")

    assert excinfo.value.code == ErrorCode.RUNTIME_UNAVAILABLE
    assert "native audio.tts.batch" in str(excinfo.value)
    assert fake_pm.calls == []


@pytest.mark.asyncio
async def test_synthetic_candidate_with_native_unavailable_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("OCTOMIL_CACHE_DIR", str(tmp_path / "empty-cache"))
    candidate = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="synthetic",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(model_id="piper-en-amy"),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )
    selection = _Selection(candidates=[candidate])
    fake_pm = _FakePrepareManager(tmp_path / "never-used")

    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_native_tts(stack, runtime_available=False)
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        kernel = _kernel_with(fake_pm)
        with pytest.raises(OctomilError) as excinfo:
            await kernel.synthesize_speech(model="@app/eternum/tts", input="hi")

    assert excinfo.value.code == ErrorCode.RUNTIME_UNAVAILABLE
    assert fake_pm.calls == []
