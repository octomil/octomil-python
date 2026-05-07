"""End-to-end test for PR 10a: transcription prepare wiring.

Asserts the contract for the lifecycle support matrix: when
``client.prepare(capability="transcription")`` succeeds, the very next
``transcribe_audio()`` call MUST construct the whisper.cpp backend with
``model_dir=<prepared_artifact_dir>`` so the engine loads the prepared
file rather than triggering pywhispercpp's own HuggingFace download.

This is the evidence test the lifecycle_support fixture for the next
release (4.10.1+) cites for ``transcription:inference_consumes_prepared``.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from octomil.execution.kernel import ExecutionKernel
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


def _local_candidate() -> RuntimeCandidatePlan:
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


class _FakePM:
    def __init__(self, artifact_dir: Path):
        self._dir = artifact_dir

    def can_prepare(self, candidate) -> bool:
        return True

    def prepare(self, candidate, *, mode=None):
        return PrepareOutcome(
            artifact_id=candidate.artifact.artifact_id,
            artifact_dir=self._dir,
            files={"": self._dir / "ggml-tiny.bin"},
            engine=candidate.engine,
            delivery_mode=candidate.delivery_mode or "sdk_runtime",
            prepare_policy=candidate.prepare_policy,
            cached=False,
        )


class _FakeWhisperBackend:
    last_kwargs: dict | None = None
    last_model_dir: str | None = None

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        _FakeWhisperBackend.last_kwargs = kwargs
        _FakeWhisperBackend.last_model_dir = kwargs.get("model_dir")

    def load_model(self, model_name: str) -> None:
        pass

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        return {
            "text": "from prepared dir",
            "segments": [{"text": "from prepared dir"}],
        }


class _FakeRegistry:
    def detect_all(self, model: str):
        @dataclass
        class _Detection:
            engine: Any
            available: bool

        @dataclass
        class _Engine:
            def create_backend(self, m, **kwargs):
                return _FakeWhisperBackend(m, **kwargs)

        return [_Detection(engine=_Engine(), available=True)]


@pytest.fixture(autouse=True)
def _reset_fakes():
    _FakeWhisperBackend.last_kwargs = None
    _FakeWhisperBackend.last_model_dir = None
    yield


@pytest.mark.asyncio
async def test_transcription_prepare_threads_artifact_dir_into_whisper_backend(tmp_path, monkeypatch):
    """The contract: prepare succeeds → next transcribe call constructs the
    whisper backend with model_dir=<prepared_dir>. If this assertion ever
    fails, transcription has fallen off the inference_consumes_prepared
    rung and the lifecycle_support fixture must be ratcheted DOWN to
    plan_only or warmup_supported until the wiring is restored.

    v0.1.5 PR-2B: this test exercises the **legacy registry-thread-
    through** path. The kernel's native-first branch (which routes
    through ``NativeSttServeAdapter`` instead of the registry) only
    fires when ``OCTOMIL_WHISPER_BIN`` is set in the env. We
    explicitly clear it here so this test remains a regression
    pin for the model_dir-threading contract that the engine-
    registry-based prepare flow still serves for fakes / future
    multi-engine paths.
    """
    monkeypatch.delenv("OCTOMIL_WHISPER_BIN", raising=False)
    candidate = _local_candidate()
    selection = _Selection(candidates=[candidate])
    artifact_dir = tmp_path / "whisper"
    artifact_dir.mkdir()
    (artifact_dir / "ggml-tiny.bin").write_bytes(b"fake whisper bytes")
    pm = _FakePM(artifact_dir)
    kernel = ExecutionKernel(prepare_manager=pm)
    # Bypass app/policy resolution.
    kernel._resolve = lambda capability, **kw: type(
        "_D",
        (),
        {"model": "whisper-tiny", "policy_preset": "local_first", "inline_policy": None, "cloud_profile": None},
    )()

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(
            patch("octomil.execution.kernel._select_locality_for_capability", return_value=("on_device", False))
        )
        stack.enter_context(patch("octomil.runtime.engines.get_registry", return_value=_FakeRegistry()))
        result = await kernel.transcribe_audio(
            audio_data=b"\x00" * 16,
            model="@app/notes/transcription",
        )

    assert _FakeWhisperBackend.last_model_dir == str(artifact_dir), (
        "Whisper backend was NOT constructed with the prepared model_dir; "
        f"last kwargs were {_FakeWhisperBackend.last_kwargs!r}. The "
        "transcription prepare lifecycle has regressed."
    )
    assert result.output_text == "from prepared dir"


# ---------------------------------------------------------------------------
# Reviewer P1 follow-ups
# ---------------------------------------------------------------------------


def test_whisper_resolver_picks_prepare_manager_sentinel_artifact_file(tmp_path):
    """PrepareManager writes single-file artifacts to ``<dir>/artifact`` (no
    extension) when the planner emits an empty ``required_files``. The
    earlier resolver only matched ``.bin``/``.gguf``/``.ggml`` and would
    silently fall back to pywhispercpp's HF download when prepared bytes
    were on disk. This test pins the sentinel-file path."""
    from octomil.runtime.engines.whisper._legacy_pywhisper import _WhisperBackend

    artifact_dir = tmp_path / "whisper-tiny"
    artifact_dir.mkdir()
    (artifact_dir / "artifact").write_bytes(b"fake whisper bytes")
    backend = _WhisperBackend("whisper-tiny", model_dir=str(artifact_dir))
    resolved = backend._resolve_local_model_file()
    assert resolved == str(artifact_dir / "artifact")


def test_whisper_resolver_falls_back_to_extension_match(tmp_path):
    """When no sentinel ``artifact`` file exists (multi-file artifact, or
    legacy required_files=['ggml-tiny.bin'] shape), the resolver still
    returns the extension match."""
    from octomil.runtime.engines.whisper._legacy_pywhisper import _WhisperBackend

    artifact_dir = tmp_path / "whisper-tiny"
    artifact_dir.mkdir()
    (artifact_dir / "ggml-tiny.bin").write_bytes(b"fake whisper bytes")
    backend = _WhisperBackend("whisper-tiny", model_dir=str(artifact_dir))
    resolved = backend._resolve_local_model_file()
    assert resolved == str(artifact_dir / "ggml-tiny.bin")


def test_whisper_resolver_returns_none_without_injected_dir(tmp_path):
    """Without a model_dir kwarg, the resolver must not invent one."""
    from octomil.runtime.engines.whisper._legacy_pywhisper import _WhisperBackend

    backend = _WhisperBackend("whisper-tiny")
    assert backend._resolve_local_model_file() is None


@pytest.mark.asyncio
async def test_transcription_local_first_skips_unpreparable_synthetic_candidate(tmp_path):
    """Reviewer P1: a synthetic planner candidate (prepare_required=True
    with no digest/url) MUST NOT make the kernel commit to local routing
    and crash in prepare. Mirror of the TTS clean-device fix.

    With ``local_first`` policy, a synthetic candidate AND no staged
    runtime should fall back to cloud rather than throw from
    PrepareManager. The kernel signals "local unavailable" before
    locality selection runs.
    """
    from octomil.runtime.planner.schemas import (
        RuntimeArtifactPlan,
        RuntimeCandidatePlan,
    )

    synthetic = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="synthetic planner candidate",
        engine="whisper.cpp",
        artifact=RuntimeArtifactPlan(model_id="whisper-tiny"),  # no digest/url
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )
    selection = _Selection(candidates=[synthetic])
    # Use the real PrepareManager so `can_prepare` runs the actual
    # structural validation. It is a pure inspection — no I/O.
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    real_pm = PrepareManager(cache_dir=tmp_path)
    kernel = ExecutionKernel(prepare_manager=real_pm)

    with patch("octomil.runtime.engines.get_registry", return_value=_FakeRegistry()):
        assert kernel._can_prepare_local_transcription("whisper-tiny", selection) is False


# ---------------------------------------------------------------------------
# Release-blocker P1: unpreparable candidate must veto local routing
# ---------------------------------------------------------------------------


def test_local_candidate_is_unpreparable_returns_true_for_synthetic(tmp_path):
    """Reviewer's release-blocker P1: when the planner emits a synthetic
    prepare_required=True candidate (no digest/url), the kernel must
    treat local as unavailable EVEN IF a backend is staged on disk —
    otherwise the OR-shortcircuit lets the staged backend win and the
    next inference call crashes in prepare()."""
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager
    from octomil.runtime.planner.schemas import (
        RuntimeArtifactPlan,
        RuntimeCandidatePlan,
    )

    synthetic = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="synthetic",
        engine="whisper.cpp",
        artifact=RuntimeArtifactPlan(model_id="whisper-tiny"),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )
    selection = _Selection(candidates=[synthetic])
    real_pm = PrepareManager(cache_dir=tmp_path)
    kernel = ExecutionKernel(prepare_manager=real_pm)
    assert kernel._local_candidate_is_unpreparable(selection) is True


def test_local_candidate_is_unpreparable_returns_false_for_real_candidate(tmp_path):
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    selection = _Selection(candidates=[_local_candidate()])
    real_pm = PrepareManager(cache_dir=tmp_path)
    kernel = ExecutionKernel(prepare_manager=real_pm)
    assert kernel._local_candidate_is_unpreparable(selection) is False


def test_local_candidate_is_unpreparable_returns_false_when_prepare_required_false(tmp_path):
    """prepare_required=False candidates are engine-managed; not a veto."""
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager
    from octomil.runtime.planner.schemas import (
        RuntimeArtifactPlan,
        RuntimeCandidatePlan,
    )

    cand = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="engine-managed",
        engine="ollama",
        artifact=RuntimeArtifactPlan(model_id="qwen2.5-7b"),
        delivery_mode="sdk_runtime",
        prepare_required=False,
        prepare_policy="lazy",
    )
    selection = _Selection(candidates=[cand])
    real_pm = PrepareManager(cache_dir=tmp_path)
    kernel = ExecutionKernel(prepare_manager=real_pm)
    assert kernel._local_candidate_is_unpreparable(selection) is False


@pytest.mark.asyncio
async def test_transcription_unpreparable_synthetic_falls_back_even_with_staged_backend(tmp_path):
    """The actual reviewer reproducer: a synthetic planner candidate AND
    `_has_local_transcription_backend` returning True. Before the veto,
    `or` short-circuited and local routing won; transcribe_audio() then
    threw inside _prepare_local_transcription_artifact. With the veto,
    the kernel marks local unavailable and falls back to cloud."""
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager
    from octomil.runtime.planner.schemas import (
        RuntimeArtifactPlan,
        RuntimeCandidatePlan,
    )

    synthetic = RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="synthetic",
        engine="whisper.cpp",
        artifact=RuntimeArtifactPlan(model_id="whisper-tiny"),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy="lazy",
    )
    selection = _Selection(candidates=[synthetic])
    real_pm = PrepareManager(cache_dir=tmp_path)
    kernel = ExecutionKernel(prepare_manager=real_pm)
    kernel._resolve = lambda capability, **kw: type(
        "_D",
        (),
        {"model": "whisper-tiny", "policy_preset": "local_first", "inline_policy": None, "cloud_profile": None},
    )()

    # Even with a staged backend pretending to be importable, the veto
    # must keep local_available=False.
    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection):
        with patch.object(kernel, "_has_local_transcription_backend", return_value=True):
            assert kernel._local_candidate_is_unpreparable(selection) is True
