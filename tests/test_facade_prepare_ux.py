"""Tests for the prepare-lifecycle facade UX:

- ExecutionKernel.prepare() — public, mode=EXPLICIT
- Octomil.prepare() — async wrapper on the facade
- ``octomil prepare`` CLI command
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from octomil._generated.error_code import ErrorCode
from octomil.commands.prepare import prepare_cmd
from octomil.errors import OctomilError
from octomil.execution.kernel import ExecutionKernel
from octomil.runtime.lifecycle.prepare_manager import PrepareOutcome
from octomil.runtime.planner.schemas import (
    ArtifactDownloadEndpoint,
    RuntimeArtifactPlan,
    RuntimeCandidatePlan,
)

# ---------------------------------------------------------------------------
# Fakes shared with the TTS adapter tests
# ---------------------------------------------------------------------------


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


def _local_candidate(*, prepare_policy: str = "lazy") -> RuntimeCandidatePlan:
    return RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="r",
        engine="sherpa-onnx",
        artifact=RuntimeArtifactPlan(
            model_id="kokoro-en-v0_19",
            artifact_id="kokoro-en-v0_19",
            digest="sha256:" + "0" * 64,
            download_urls=[ArtifactDownloadEndpoint(url="https://cdn.example.com/")],
        ),
        delivery_mode="sdk_runtime",
        prepare_required=True,
        prepare_policy=prepare_policy,  # type: ignore[arg-type]
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
            files={"": self._dir / "artifact"},
            engine=candidate.engine,
            delivery_mode=candidate.delivery_mode or "sdk_runtime",
            prepare_policy=candidate.prepare_policy,
            cached=False,
        )


def _stub_resolve(kernel: ExecutionKernel, *, model: str = "kokoro-en-v0_19") -> None:
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {"model": model, "policy_preset": "local_first", "inline_policy": None, "cloud_profile": None},
    )()


# ---------------------------------------------------------------------------
# Kernel.prepare()
# ---------------------------------------------------------------------------


def test_kernel_prepare_runs_with_explicit_mode_and_returns_outcome(tmp_path):
    candidate = _local_candidate()
    selection = _Selection(candidates=[candidate])
    pm = _RecordingPM(tmp_path)
    kernel = ExecutionKernel(prepare_manager=pm)
    _stub_resolve(kernel)

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection):
        outcome = kernel.prepare(model="@app/eternum/tts")

    from octomil.runtime.lifecycle.prepare_manager import PrepareMode

    assert outcome.artifact_dir == tmp_path
    assert pm.calls == [("kokoro-en-v0_19", PrepareMode.EXPLICIT)]


def test_kernel_prepare_succeeds_for_explicit_only_policy(tmp_path):
    """The whole point of mode=EXPLICIT: explicit_only candidates must work."""
    candidate = _local_candidate(prepare_policy="explicit_only")
    selection = _Selection(candidates=[candidate])
    pm = _RecordingPM(tmp_path)
    kernel = ExecutionKernel(prepare_manager=pm)
    _stub_resolve(kernel)

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection):
        outcome = kernel.prepare(model="@app/eternum/tts")

    assert outcome.artifact_id == "kokoro-en-v0_19"


def test_kernel_prepare_rejects_unknown_capability(tmp_path):
    kernel = ExecutionKernel()
    with pytest.raises(OctomilError) as excinfo:
        kernel.prepare(model="m", capability="vision")
    assert excinfo.value.code == ErrorCode.INVALID_INPUT
    assert "vision" in str(excinfo.value)


@pytest.mark.parametrize("capability", ["embedding", "chat"])
def test_kernel_prepare_rejects_unwired_capabilities(tmp_path, capability):
    """The remaining unwired capabilities (embedding, chat). Their inference
    adapters do NOT yet thread the prepared model_dir, so prepare must
    reject them with an actionable INVALID_INPUT message. Transcription
    was added to the supported set in PR 10a."""
    kernel = ExecutionKernel()
    with pytest.raises(OctomilError) as excinfo:
        kernel.prepare(model="m", capability=capability)
    assert excinfo.value.code == ErrorCode.INVALID_INPUT
    msg = str(excinfo.value)
    assert capability in msg
    assert "tts" in msg.lower()


def test_kernel_prepare_accepts_transcription_now(tmp_path):
    """PR 10a: transcription joined the supported set. The kernel must
    resolve the planner selection and call PrepareManager just like it
    does for tts."""
    candidate = _local_candidate()
    selection = _Selection(candidates=[candidate])
    pm = _RecordingPM(tmp_path)
    kernel = ExecutionKernel(prepare_manager=pm)
    _stub_resolve(kernel)

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection):
        outcome = kernel.prepare(model="@app/notes/transcription", capability="transcription")

    from octomil.runtime.lifecycle.prepare_manager import PrepareMode

    assert outcome.artifact_id == "kokoro-en-v0_19"  # whatever the stub returns
    assert pm.calls == [("kokoro-en-v0_19", PrepareMode.EXPLICIT)]


def test_kernel_prepare_raises_when_no_local_candidate(tmp_path):
    selection = _Selection(candidates=[])  # planner returned nothing local
    kernel = ExecutionKernel()
    _stub_resolve(kernel)
    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection):
        with pytest.raises(OctomilError) as excinfo:
            kernel.prepare(model="@app/eternum/tts")
    assert excinfo.value.code == ErrorCode.RUNTIME_UNAVAILABLE


# ---------------------------------------------------------------------------
# Octomil.prepare() (facade)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facade_prepare_delegates_to_kernel_prepare(tmp_path, monkeypatch):
    from octomil.facade import Octomil

    # Construct without going through initialize() — stub state directly.
    client = Octomil.__new__(Octomil)
    client._initialized = True

    candidate = _local_candidate(prepare_policy="explicit_only")
    selection = _Selection(candidates=[candidate])
    pm = _RecordingPM(tmp_path)
    kernel = ExecutionKernel(prepare_manager=pm)
    _stub_resolve(kernel)
    client._kernel = kernel

    with patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection):
        outcome = await client.prepare(model="@app/eternum/tts")

    from octomil.runtime.lifecycle.prepare_manager import PrepareMode

    assert outcome.artifact_id == "kokoro-en-v0_19"
    assert pm.calls == [("kokoro-en-v0_19", PrepareMode.EXPLICIT)]


@pytest.mark.asyncio
async def test_facade_prepare_raises_before_initialize():
    from octomil.facade import Octomil, OctomilNotInitializedError

    client = Octomil.__new__(Octomil)
    client._initialized = False
    with pytest.raises(OctomilNotInitializedError):
        await client.prepare(model="@app/eternum/tts")


# ---------------------------------------------------------------------------
# CLI: `octomil prepare`
# ---------------------------------------------------------------------------


def test_cli_prepare_prints_downloaded_artifact_dir(tmp_path):
    candidate = _local_candidate()
    selection = _Selection(candidates=[candidate])
    pm = _RecordingPM(tmp_path / "artifacts")
    runner = CliRunner()

    fake_kernel = _kernel_with_pm(pm, selection)

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel.ExecutionKernel", lambda **kw: fake_kernel))
        result = runner.invoke(prepare_cmd, ["@app/eternum/tts"])

    assert result.exit_code == 0, result.output
    assert "downloaded:" in result.output
    assert "kokoro-en-v0_19" in result.output


def test_cli_prepare_prints_actionable_error_on_failure(tmp_path):
    err = OctomilError(code=ErrorCode.RUNTIME_UNAVAILABLE, message="planner offline")

    class _BadKernel:
        def prepare(self, **kw):
            raise err

    runner = CliRunner()
    with patch("octomil.execution.kernel.ExecutionKernel", lambda **kw: _BadKernel()):
        result = runner.invoke(prepare_cmd, ["@app/eternum/tts"])

    assert result.exit_code == 1
    assert "planner offline" in result.output


def test_cli_prepare_rejects_unsupported_capability():
    runner = CliRunner()
    # 'vision' is not in the planner enum.
    result = runner.invoke(prepare_cmd, ["@app/eternum/tts", "--capability", "vision"])
    assert result.exit_code != 0
    assert "vision" in result.output


def test_cli_prepare_accepts_tts_capability(tmp_path):
    candidate = _local_candidate()
    selection = _Selection(candidates=[candidate])
    pm = _RecordingPM(tmp_path / "artifacts")
    fake_kernel = _kernel_with_pm(pm, selection)
    runner = CliRunner()

    with patch("octomil.execution.kernel.ExecutionKernel", lambda **kw: fake_kernel):
        result = runner.invoke(prepare_cmd, ["m", "--capability", "tts"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("cap", ["embedding", "chat"])
def test_cli_prepare_rejects_unwired_capabilities_at_choice_constraint(cap):
    """CLI surface must match the kernel: tts and transcription are wired,
    so embedding/chat are rejected at the click choice constraint until
    their backends consume the prepared dir."""
    runner = CliRunner()
    result = runner.invoke(prepare_cmd, ["m", "--capability", cap])
    assert result.exit_code != 0
    assert cap in result.output


def _kernel_with_pm(pm, selection):
    kernel = ExecutionKernel(prepare_manager=pm)
    _stub_resolve(kernel)
    # Make planner resolution return our scripted selection.
    import octomil.execution.kernel as _kmod

    _kmod._resolve_planner_selection_default = _kmod._resolve_planner_selection  # save
    _kmod._resolve_planner_selection = lambda *a, **kw: selection
    return kernel
