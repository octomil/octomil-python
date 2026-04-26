"""End-to-end tests for PR 11: unified ``client.warmup()``.

``warmup`` is a strict superset of ``prepare``: bytes on disk *plus*
the local backend constructed and ``load_model``'d, with the loaded
instance cached on the kernel so the next inference dispatch in the
same process reuses it instead of paying ``engine.create_backend`` +
``backend.load_model`` again.

Tests pin:
  - the cache hit (post-warmup, the resolver returns the cached
    backend without touching the engine registry);
  - the supported-capabilities gate (chat/responses/embedding still
    rejected with INVALID_INPUT);
  - the partial-success shape (prepare succeeds, backend constructor
    fails → ``backend_loaded=False`` and bytes still on disk);
  - the CLI surface.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from octomil.commands.warmup import warmup_cmd
from octomil.errors import OctomilError, OctomilErrorCode
from octomil.execution.kernel import (
    _WARMUPABLE_CAPABILITIES,
    ExecutionKernel,
    WarmupOutcome,
)
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


def _local_candidate(engine: str = "sherpa-onnx", artifact_id: str = "kokoro-en-v0_19") -> RuntimeCandidatePlan:
    return RuntimeCandidatePlan(
        locality="local",
        priority=0,
        confidence=0.9,
        reason="local-first",
        engine=engine,
        artifact=RuntimeArtifactPlan(
            model_id=artifact_id,
            artifact_id=artifact_id,
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


def _stub_kernel_resolve(kernel: ExecutionKernel, model: str) -> None:
    kernel._resolve = lambda capability, **kw: type(  # type: ignore[method-assign]
        "_D",
        (),
        {
            "model": model,
            "policy_preset": "local_first",
            "inline_policy": None,
            "cloud_profile": None,
        },
    )()


# ---------------------------------------------------------------------------
# Capability gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capability", ["chat", "responses", "embedding", "vision"])
def test_warmup_rejects_unwired_capabilities(capability, tmp_path):
    kernel = ExecutionKernel()
    with pytest.raises(OctomilError) as excinfo:
        kernel.warmup(model="m", capability=capability)
    assert excinfo.value.code == OctomilErrorCode.INVALID_INPUT
    msg = str(excinfo.value)
    assert capability in msg
    assert "tts" in msg.lower()


def test_warmupable_capabilities_invariant():
    assert "tts" in _WARMUPABLE_CAPABILITIES
    assert "transcription" in _WARMUPABLE_CAPABILITIES
    assert "chat" not in _WARMUPABLE_CAPABILITIES
    assert "responses" not in _WARMUPABLE_CAPABILITIES
    assert "embedding" not in _WARMUPABLE_CAPABILITIES


# ---------------------------------------------------------------------------
# Cache hit: post-warmup TTS dispatch reuses the loaded backend
# ---------------------------------------------------------------------------


class _FakeSherpaBackend:
    """Stand-in sherpa-onnx backend; counts how many times it's loaded."""

    load_calls = 0

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def load_model(self, model_name: str) -> None:
        _FakeSherpaBackend.load_calls += 1


@pytest.fixture(autouse=True)
def _reset_load_counter():
    _FakeSherpaBackend.load_calls = 0
    yield


def test_warmup_loads_backend_and_caches_it(tmp_path):
    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()
    pm = _FakePM(artifact_dir)
    kernel = ExecutionKernel(prepare_manager=pm)
    _stub_kernel_resolve(kernel, "kokoro-en-v0_19")

    selection = _Selection(candidates=[_local_candidate(engine="sherpa-onnx")])

    class _FakeEngine:
        def create_backend(self, model: str, **kwargs: Any) -> _FakeSherpaBackend:
            return _FakeSherpaBackend(model, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", _FakeEngine))
        outcome = kernel.warmup(model="kokoro-en-v0_19", capability="tts")

    assert isinstance(outcome, WarmupOutcome)
    assert outcome.capability == "tts"
    assert outcome.model == "kokoro-en-v0_19"
    assert outcome.backend_loaded is True
    assert outcome.prepare_outcome.artifact_id == "kokoro-en-v0_19"
    assert pm.prepare_calls == ["kokoro-en-v0_19"]
    assert _FakeSherpaBackend.load_calls == 1
    # Cache populated.
    assert ("tts", "kokoro-en-v0_19") in kernel._warmed_backends


def test_post_warmup_dispatch_skips_backend_construction(tmp_path):
    """The contract: after warmup, the next ``_resolve_local_tts_backend``
    call returns the cached instance without re-running
    ``engine.create_backend`` or ``backend.load_model``."""
    kernel = ExecutionKernel()

    sentinel_backend = object()
    kernel._warmed_backends[("tts", "kokoro-en-v0_19")] = sentinel_backend

    # The resolver short-circuits on the cache hit; the engine import
    # is never reached. (If it were, the patch below would fail the
    # test by raising.)
    with patch("octomil.runtime.engines.sherpa.SherpaTtsEngine") as fake_engine:
        fake_engine.side_effect = AssertionError("engine constructor must not be called on cache hit")
        result = kernel._resolve_local_tts_backend("kokoro-en-v0_19")

    assert result is sentinel_backend


def test_post_warmup_transcription_dispatch_skips_registry_walk(tmp_path):
    kernel = ExecutionKernel()

    sentinel_backend = object()
    kernel._warmed_backends[("transcription", "whisper-tiny")] = sentinel_backend

    with patch("octomil.runtime.engines.get_registry") as fake_registry:
        fake_registry.side_effect = AssertionError("registry walk must not be called on cache hit")
        result = kernel._resolve_local_transcription_backend("whisper-tiny")

    assert result is sentinel_backend


# ---------------------------------------------------------------------------
# Partial success: prepare ok, backend load fails → backend_loaded=False
# ---------------------------------------------------------------------------


def test_warmup_returns_partial_success_when_backend_construction_fails(tmp_path):
    """Prepare succeeded, backend constructor refused. The outcome
    must report ``backend_loaded=False`` so the caller knows the cache
    didn't get populated and inference will fall through the cold
    path. Critically, ``prepare`` is not retried — bytes are on disk."""
    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()
    pm = _FakePM(artifact_dir)
    kernel = ExecutionKernel(prepare_manager=pm)
    _stub_kernel_resolve(kernel, "kokoro-en-v0_19")

    selection = _Selection(candidates=[_local_candidate(engine="sherpa-onnx")])

    class _FailingEngine:
        def create_backend(self, model: str, **kwargs: Any):
            raise RuntimeError("sherpa native lib missing on this host")

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch("octomil.runtime.engines.sherpa.SherpaTtsEngine", _FailingEngine))
        outcome = kernel.warmup(model="kokoro-en-v0_19", capability="tts")

    assert outcome.backend_loaded is False
    assert outcome.prepare_outcome.artifact_id == "kokoro-en-v0_19"
    assert pm.prepare_calls == ["kokoro-en-v0_19"]
    assert ("tts", "kokoro-en-v0_19") not in kernel._warmed_backends


# ---------------------------------------------------------------------------
# release_warmed_backends drops the cache
# ---------------------------------------------------------------------------


def test_release_warmed_backends_clears_cache():
    kernel = ExecutionKernel()
    kernel._warmed_backends[("tts", "model-a")] = object()
    kernel._warmed_backends[("transcription", "model-b")] = object()
    kernel.release_warmed_backends()
    assert kernel._warmed_backends == {}
    # Idempotent.
    kernel.release_warmed_backends()
    assert kernel._warmed_backends == {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_warmup_reports_loaded_state_and_latency(tmp_path):
    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()

    fake_outcome = WarmupOutcome(
        capability="tts",
        model="kokoro-en-v0_19",
        prepare_outcome=PrepareOutcome(
            artifact_id="kokoro-en-v0_19",
            artifact_dir=artifact_dir,
            files={"": artifact_dir / "artifact"},
            engine="sherpa-onnx",
            delivery_mode="sdk_runtime",
            prepare_policy="lazy",
            cached=False,
        ),
        backend_loaded=True,
        latency_ms=42.0,
    )

    class _StubKernel:
        def warmup(self, **kw):
            return fake_outcome

    runner = CliRunner()
    with patch("octomil.execution.kernel.ExecutionKernel", lambda **kw: _StubKernel()):
        result = runner.invoke(warmup_cmd, ["kokoro-en-v0_19"])
    assert result.exit_code == 0, result.output
    assert "downloaded+loaded" in result.output
    assert "kokoro-en-v0_19" in result.output
    assert "42 ms" in result.output


def test_cli_warmup_warns_when_backend_not_loaded(tmp_path):
    artifact_dir = tmp_path / "kokoro"
    artifact_dir.mkdir()

    fake_outcome = WarmupOutcome(
        capability="tts",
        model="kokoro-en-v0_19",
        prepare_outcome=PrepareOutcome(
            artifact_id="kokoro-en-v0_19",
            artifact_dir=artifact_dir,
            files={},
            engine="sherpa-onnx",
            delivery_mode="sdk_runtime",
            prepare_policy="lazy",
            cached=True,
        ),
        backend_loaded=False,
        latency_ms=12.0,
    )

    class _StubKernel:
        def warmup(self, **kw):
            return fake_outcome

    runner = CliRunner()
    with patch("octomil.execution.kernel.ExecutionKernel", lambda **kw: _StubKernel()):
        result = runner.invoke(warmup_cmd, ["kokoro-en-v0_19"])
    assert result.exit_code == 0, result.output
    assert "cached+load_skipped" in result.output
    # The note is written to stderr but click's CliRunner captures it
    # in the same .output buffer by default.
    assert "backend was not loaded" in result.output


def test_cli_warmup_rejects_unwired_capability(tmp_path):
    runner = CliRunner()
    result = runner.invoke(warmup_cmd, ["m", "--capability", "chat"])
    assert result.exit_code != 0
    assert "chat" in result.output
