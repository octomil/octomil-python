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
async def test_transcription_prepare_threads_artifact_dir_into_whisper_backend(tmp_path):
    """The contract: prepare succeeds → next transcribe call constructs the
    whisper backend with model_dir=<prepared_dir>. If this assertion ever
    fails, transcription has fallen off the inference_consumes_prepared
    rung and the lifecycle_support fixture must be ratcheted DOWN to
    plan_only or warmup_supported until the wiring is restored."""
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
