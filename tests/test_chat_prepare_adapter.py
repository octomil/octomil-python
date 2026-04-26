"""End-to-end test for PR 10c: chat/responses prepare wiring.

Asserts the contract for the lifecycle support matrix: when
``client.prepare(capability="chat")`` succeeds, the very next
``create_response()`` / ``stream_response()`` call MUST construct the
local chat backend (mlx-lm or llama.cpp) with ``model_dir=<prepared_dir>``
so the engine loads the prepared bytes rather than triggering its own
HuggingFace ``snapshot_download`` / ``from_pretrained`` path.

This is the evidence test the lifecycle_support fixture for the next
release cites for ``chat:inference_consumes_prepared`` and
``responses:inference_consumes_prepared``.
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


class _FakePM:
    def __init__(self, artifact_dir: Path):
        self._dir = artifact_dir

    def can_prepare(self, candidate) -> bool:
        return True

    def prepare(self, candidate, *, mode=None):
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
    """Stand-in InferenceBackend; records the ``model_dir`` it was built with."""

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
    """Stand-in for MLXEngine / LlamaCppEngine.

    ``_build_router``'s planner-engine branch calls
    ``engine.create_backend(model, **backend_kwargs)``; we record the
    ``model_dir`` passed in so the test can assert PR 10c's contract.
    """

    name = "mlx-lm"
    last_kwargs: dict[str, Any] | None = None
    last_model_dir: str | None = None
    last_model: str | None = None

    def detect(self) -> bool:
        return True

    def create_backend(self, model: str, **kwargs: Any) -> _FakeBackend:
        _FakeEngine.last_kwargs = kwargs
        _FakeEngine.last_model_dir = kwargs.get("model_dir")
        _FakeEngine.last_model = model
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


@pytest.mark.asyncio
async def test_chat_prepare_threads_artifact_dir_into_local_chat_backend(tmp_path):
    """The contract: prepare succeeds → next create_response constructs
    the local chat engine's backend with ``model_dir=<prepared_dir>``.

    If this assertion ever fails, chat/responses has fallen off the
    ``inference_consumes_prepared`` rung and the lifecycle_support
    fixture must be ratcheted DOWN to ``plan_only`` /
    ``warmup_supported`` until the wiring is restored.
    """
    candidate = _local_chat_candidate(engine="mlx-lm")
    selection = _Selection(candidates=[candidate])
    artifact_dir = tmp_path / "gemma3-1b"
    artifact_dir.mkdir()
    (artifact_dir / "artifact").write_bytes(b"fake mlx weights")
    pm = _FakePM(artifact_dir)
    kernel = ExecutionKernel(prepare_manager=pm)
    kernel._resolve = lambda capability, **kw: _make_defaults()

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch("octomil.runtime.engines.get_registry", return_value=_FakeEngineRegistry()))
        result = await kernel.create_response(
            "Hello",
            model="gemma3-1b",
        )

    assert _FakeEngine.last_model_dir == str(artifact_dir), (
        "Local chat backend was NOT constructed with the prepared model_dir; "
        f"last kwargs were {_FakeEngine.last_kwargs!r}. The chat prepare "
        "lifecycle has regressed."
    )
    assert _FakeEngine.last_model == "gemma3-1b"
    assert result.output_text == "from prepared dir"
    assert result.locality == "on_device"


@pytest.mark.asyncio
async def test_stream_response_threads_artifact_dir_into_local_chat_backend(tmp_path):
    """``stream_response`` must apply the same prepared-dir threading
    as ``create_response``. Without this, switching from non-streaming
    to streaming would silently drop back to the engine's own download
    path on the same model that was just prepared."""
    candidate = _local_chat_candidate(engine="mlx-lm")
    selection = _Selection(candidates=[candidate])
    artifact_dir = tmp_path / "gemma3-1b"
    artifact_dir.mkdir()
    (artifact_dir / "artifact").write_bytes(b"fake mlx weights")
    pm = _FakePM(artifact_dir)
    kernel = ExecutionKernel(prepare_manager=pm)
    kernel._resolve = lambda capability, **kw: _make_defaults()

    captured: dict[str, Any] = {}
    original_build_router = kernel._build_router

    async def capturing_build_router(*args, **kwargs):
        captured.setdefault("calls", []).append(kwargs)
        # Return a stub router whose stream() yields one chunk.
        return _StubRouter()

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch.object(kernel, "_build_router", side_effect=capturing_build_router))
        chunks = []
        async for chunk in kernel.stream_response("Hello", model="gemma3-1b"):
            chunks.append(chunk)

    assert any(
        call.get("prepared_model_dir") == str(artifact_dir) for call in captured["calls"]
    ), f"stream_response did not pass prepared_model_dir into _build_router; calls were {captured['calls']!r}"
    # Sanity: at least one delta and a terminal done chunk
    assert chunks
    assert chunks[-1].done
    # The non-streaming path used the original method; restore reference.
    assert original_build_router is not None


@pytest.mark.asyncio
async def test_stream_chat_messages_threads_artifact_dir_into_local_chat_backend(tmp_path):
    """``stream_chat_messages`` is the multi-turn streaming entrypoint;
    it must thread the prepared dir for the same reason."""
    candidate = _local_chat_candidate(engine="mlx-lm")
    selection = _Selection(candidates=[candidate])
    artifact_dir = tmp_path / "gemma3-1b"
    artifact_dir.mkdir()
    (artifact_dir / "artifact").write_bytes(b"fake mlx weights")
    pm = _FakePM(artifact_dir)
    kernel = ExecutionKernel(prepare_manager=pm)
    kernel._resolve = lambda capability, **kw: _make_defaults()

    captured: dict[str, Any] = {}

    async def capturing_build_router(*args, **kwargs):
        captured.setdefault("calls", []).append(kwargs)
        return _StubRouter()

    with ExitStack() as stack:
        stack.enter_context(patch("octomil.execution.kernel._resolve_planner_selection", return_value=selection))
        stack.enter_context(patch.object(kernel, "_build_router", side_effect=capturing_build_router))
        chunks = []
        async for chunk in kernel.stream_chat_messages(
            [{"role": "user", "content": "Hello"}],
            model="gemma3-1b",
        ):
            chunks.append(chunk)

    assert any(call.get("prepared_model_dir") == str(artifact_dir) for call in captured["calls"])


def test_prepare_accepts_chat_capability(tmp_path):
    """``client.prepare(capability="chat")`` must succeed (or at least
    not be rejected on the capability gate). PR 10c widens the
    ``_PREPAREABLE_CAPABILITIES`` set."""
    from octomil.execution.kernel import _PREPAREABLE_CAPABILITIES

    assert "chat" in _PREPAREABLE_CAPABILITIES
    assert "responses" in _PREPAREABLE_CAPABILITIES
    assert "tts" in _PREPAREABLE_CAPABILITIES
    assert "transcription" in _PREPAREABLE_CAPABILITIES


def test_prepare_local_chat_artifact_skips_unpreparable_synthetic(tmp_path):
    """A synthetic local candidate (prepare_required=True with no
    digest/url) MUST NOT crash ``create_response`` in prepare. The
    helper returns ``None`` so the runner's existing fallback handles
    promotion to cloud after the local attempt fails."""
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
    selection = _Selection(candidates=[synthetic])
    real_pm = PrepareManager(cache_dir=tmp_path)
    kernel = ExecutionKernel(prepare_manager=real_pm)

    assert kernel._prepare_local_chat_artifact(selection) is None


def test_prepare_local_chat_artifact_returns_none_for_no_local_candidate(tmp_path):
    """Cloud-only or no-candidate selections must return ``None``."""
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    cloud_only = RuntimeCandidatePlan(
        locality="cloud",
        priority=0,
        confidence=0.9,
        reason="cloud-only",
    )
    selection = _Selection(candidates=[cloud_only])
    real_pm = PrepareManager(cache_dir=tmp_path)
    kernel = ExecutionKernel(prepare_manager=real_pm)
    assert kernel._prepare_local_chat_artifact(selection) is None
    assert kernel._prepare_local_chat_artifact(None) is None


def test_prepare_local_chat_artifact_returns_none_for_engine_managed(tmp_path):
    """``prepare_required=False`` candidates are engine-managed (e.g.
    ollama) — the helper must skip them."""
    from octomil.runtime.lifecycle.prepare_manager import PrepareManager

    engine_managed = RuntimeCandidatePlan(
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
    selection = _Selection(candidates=[engine_managed])
    real_pm = PrepareManager(cache_dir=tmp_path)
    kernel = ExecutionKernel(prepare_manager=real_pm)
    assert kernel._prepare_local_chat_artifact(selection) is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubRouter:
    """Minimal RouterModelRuntime stub for streaming tests."""

    async def run(self, request: Any, *, policy: Any = None) -> Any:
        from octomil.runtime.core.types import RuntimeResponse, RuntimeUsage

        return RuntimeResponse(
            text="from prepared dir",
            usage=RuntimeUsage(prompt_tokens=2, completion_tokens=8, total_tokens=10),
            finish_reason="stop",
        )

    async def stream(self, request: Any, *, policy: Any = None):
        from octomil.runtime.core.types import RuntimeChunk

        yield RuntimeChunk(text="from prepared dir", finish_reason=None)
        yield RuntimeChunk(text="", finish_reason="stop")
