"""Public-facade evidence for transcription/prepare and transcription/warmup.

The kernel-level prepare + warmup paths for transcription are pinned by
``test_facade_prepare_ux.py::test_kernel_prepare_accepts_transcription_now``,
``test_transcription_prepare_adapter.py::test_transcription_prepare_threads_artifact_dir_into_whisper_backend``,
and ``test_warmup.py::test_transcription_warmup_calls_load_model_before_caching``.

These tests close the public-facade loop: the same async
``await client.prepare(capability='transcription')`` and
``await client.warmup(capability='transcription')`` paths users actually
call must reach the kernel and return the kernel's PrepareOutcome /
WarmupOutcome unchanged.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from octomil.execution.kernel import ExecutionKernel
from octomil.facade import Octomil, OctomilNotInitializedError
from octomil.runtime.lifecycle.prepare_manager import PrepareOutcome
from octomil.runtime.planner.schemas import (
    ArtifactDownloadEndpoint,
    RuntimeArtifactPlan,
    RuntimeCandidatePlan,
)


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


def _local_whisper_candidate() -> RuntimeCandidatePlan:
    return RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="local-first",
        engine="whisper.cpp",
        artifact=RuntimeArtifactPlan(
            model_id="whisper-tiny",
            artifact_id="whisper-tiny",
            digest="sha256:" + "0" * 64,
            download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
        ),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )


class _RecordingPM:
    def __init__(self, artifact_dir: Path):
        self._dir = artifact_dir
        self.calls: list[tuple[str, Any]] = []

    def can_prepare(self, candidate) -> bool:
        return True

    def prepare(self, candidate, *, mode=None):
        self.calls.append((candidate.artifact.artifact_id, mode))
        return PrepareOutcome(
            artifact_id=candidate.artifact.artifact_id,
            artifact_dir=self._dir,
            files={"": self._dir / "ggml-tiny.bin"},
            engine=candidate.engine,
            delivery_mode=candidate.delivery_mode or "sdk_runtime",
            prepare_policy=candidate.prepare_policy,
            cached=False,
        )


def _stub_resolve(kernel: ExecutionKernel, *, model: str = "whisper-tiny") -> None:
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {"model": model, "policy_preset": "local_first", "inline_policy": None, "cloud_profile": None},
    )()


# ---------------------------------------------------------------------------
# Octomil.prepare(capability='transcription')
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facade_prepare_transcription_delegates_to_kernel(tmp_path):
    candidate = _local_whisper_candidate()
    selection = _Selection(candidates=[candidate])
    pm = _RecordingPM(tmp_path)
    kernel = ExecutionKernel(prepare_manager=pm)
    _stub_resolve(kernel)

    client = Octomil.__new__(Octomil)
    client._initialized = True
    client._kernel = kernel

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection):
        outcome = await client.prepare(model="@app/notes/transcription", capability="transcription")

    from octomil.runtime.lifecycle.prepare_manager import PrepareMode

    assert outcome.artifact_id == "whisper-tiny"
    assert outcome.artifact_dir == tmp_path
    assert pm.calls == [("whisper-tiny", PrepareMode.EXPLICIT)]


@pytest.mark.asyncio
async def test_facade_prepare_transcription_raises_before_initialize():
    client = Octomil.__new__(Octomil)
    client._initialized = False
    with pytest.raises(OctomilNotInitializedError):
        await client.prepare(model="@app/notes/transcription", capability="transcription")


# ---------------------------------------------------------------------------
# Octomil.warmup(capability='transcription')
# ---------------------------------------------------------------------------


class _LazyWhisperBackend:
    load_calls = 0

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs

    def load_model(self, model_name: str) -> None:
        type(self).load_calls += 1

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        return {"text": "ok"}


@pytest.fixture(autouse=True)
def _reset_lazy_load_counter():
    _LazyWhisperBackend.load_calls = 0
    yield


@pytest.mark.asyncio
async def test_facade_warmup_transcription_loads_backend_and_caches(tmp_path):
    """End-to-end: ``await client.warmup(capability='transcription')``
    must (a) reach the kernel, (b) prepare the artifact, (c) call
    ``backend.load_model`` before caching, (d) populate the warmed
    backend cache so the next ``transcribe_audio`` reuses it."""
    artifact_dir = tmp_path / "whisper"
    artifact_dir.mkdir()
    pm = _RecordingPM(artifact_dir)
    kernel = ExecutionKernel(prepare_manager=pm)
    _stub_resolve(kernel)

    client = Octomil.__new__(Octomil)
    client._initialized = True
    client._kernel = kernel

    selection = _Selection(candidates=[_local_whisper_candidate()])

    @dataclass
    class _Detection:
        engine: Any
        available: bool

    class _FakeEngine:
        def create_backend(self, model: str, **kwargs: Any) -> _LazyWhisperBackend:
            return _LazyWhisperBackend(model, **kwargs)

    class _FakeRegistry:
        def detect_all(self, model: str):
            return [_Detection(engine=_FakeEngine(), available=True)]

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch("octomil.runtime.engines.get_registry", return_value=_FakeRegistry()))
        outcome = await client.warmup(model="whisper-tiny", capability="transcription")

    assert outcome.capability == "transcription"
    assert outcome.model == "whisper-tiny"
    assert outcome.backend_loaded is True
    assert _LazyWhisperBackend.load_calls == 1
    cached_keys = list(kernel._warmed_backends.keys())
    assert any(k[0] == "transcription" and k[1] == "whisper-tiny" for k in cached_keys), cached_keys


@pytest.mark.asyncio
async def test_facade_warmup_transcription_raises_before_initialize():
    client = Octomil.__new__(Octomil)
    client._initialized = False
    with pytest.raises(OctomilNotInitializedError):
        await client.warmup(model="whisper-tiny", capability="transcription")
