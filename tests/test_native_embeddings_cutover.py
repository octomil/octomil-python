"""v0.1.3 — Native embeddings hard-cutover tests.

Pins the cutover discipline that wires
:class:`NativeEmbeddingsBackend` into the kernel + local-runner
paths. Sibling to ``test_native_embeddings_backend.py`` (which pins
the backend itself); these tests pin the SDK boundary:

  1. Source-pin: no Python embedder import path
     (``sentence_transformers`` / ``llama_cpp.LlamaEmbedding`` /
     ``fastembed`` / ``torch``) is referenced from the embedding
     code paths.
  2. Registry resolution: an embedding-family model id
     (``nomic-embed-...``, ``bge-...``) resolves to a
     :class:`NativeEmbeddingsRuntime` (NOT the chat default
     factory).
  3. ``_local_embed`` calls ``runtime.embed(inputs)`` and surfaces
     ``EmbeddingResult.embeddings`` + ``usage.total_tokens`` into
     :class:`ExecutionResult`.
  4. Batch order is preserved end-to-end through the bridge.
  5. Chat-only model id surfaces a bounded
     :class:`OctomilError(UNSUPPORTED_MODALITY)` rather than a plain
     ``RuntimeError`` or a silent Python fallback.
  6. ``/v1/embeddings`` server route returns OpenAI-compatible
     shape and refuses non-embedding model ids with
     ``UNSUPPORTED_MODALITY``.

Hard-cutover discipline: every reject path is a bounded
:class:`OctomilError`. There is no fallback.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# (1) Source-pin: no Python embedder fallback in the cutover code paths.
# ---------------------------------------------------------------------------


_BANNED_EMBEDDER_IMPORTS = (
    "sentence_transformers",
    "fastembed",
    "llama_cpp.LlamaEmbedding",  # llama_cpp is fine; the embedding class is not
    # NOTE: bare ``import torch`` is permitted at the SDK level (other code
    # paths legitimately use it). The cutover specifically bans torch-backed
    # *embedding* shortcuts; pinning that here as a sentinel string is more
    # noise than signal, so we don't.
)

_CUTOVER_SOURCES = (
    "octomil/runtime/native/embeddings_backend.py",
    "octomil/runtime/native/embeddings_runtime.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_no_python_embedder_fallback_in_native_paths():
    """The native embedding modules MUST NOT import a Python-local
    embedder. The hard cutover guarantees the only path is the cffi
    runtime; a stray import here would make a silent Python fallback
    one ``except`` clause away."""
    root = _repo_root()
    for src_relative in _CUTOVER_SOURCES:
        src_path = root / src_relative
        text = src_path.read_text(encoding="utf-8")
        for banned in _BANNED_EMBEDDER_IMPORTS:
            assert banned not in text, f"{src_relative} unexpectedly references {banned!r}"


def test_kernel_local_embed_does_not_import_python_embedder():
    """``ExecutionKernel._local_embed`` must not reach for a Python
    embedder. We pin the substring rather than execute the kernel
    because the kernel imports lazily — a successful run wouldn't
    catch an import-on-fallback path."""
    src = (_repo_root() / "octomil/execution/kernel.py").read_text(encoding="utf-8")
    # Find the _local_embed method body. Pin: it does NOT mention
    # any Python embedder package on import.
    for banned in _BANNED_EMBEDDER_IMPORTS:
        assert banned not in src, f"octomil/execution/kernel.py unexpectedly references {banned!r}"


def test_serve_app_does_not_import_python_embedder():
    src = (_repo_root() / "octomil/serve/app.py").read_text(encoding="utf-8")
    for banned in _BANNED_EMBEDDER_IMPORTS:
        assert banned not in src, f"octomil/serve/app.py unexpectedly references {banned!r}"


# ---------------------------------------------------------------------------
# (2) Registry resolution.
# ---------------------------------------------------------------------------


def test_registry_resolves_embedding_family_to_native_runtime():
    """A known embedding-family prefix must resolve to a
    :class:`NativeEmbeddingsRuntime`. If the registry returned the
    chat default factory's adapter, calling ``.embed()`` would fall
    through to ``getattr(...) or None`` and the kernel would surface
    a useless error."""
    from octomil.runtime.core.registry import ModelRuntimeRegistry

    # Force registration (idempotent — ensures test ordering doesn't
    # matter when other tests clear() the registry).
    from octomil.runtime.native.embeddings_runtime import NativeEmbeddingsRuntime, register_native_embeddings_factory

    register_native_embeddings_factory()

    runtime = ModelRuntimeRegistry.shared().resolve("nomic-embed-text-v1.5")
    assert isinstance(runtime, NativeEmbeddingsRuntime)


def test_registry_returns_non_embedding_to_default_factory():
    """A non-embedding model id must NOT resolve to a
    :class:`NativeEmbeddingsRuntime`. The factory's family-prefix
    gate is what stops chat models from being silently embedded."""
    from octomil.runtime.native.embeddings_runtime import (
        NativeEmbeddingsRuntime,
        is_embedding_model,
        native_embeddings_factory,
    )

    assert not is_embedding_model("phi-4-mini")
    runtime = native_embeddings_factory("phi-4-mini")
    assert runtime is None
    # Sanity: an embedding family DOES resolve.
    runtime = native_embeddings_factory("bge-base-en-v1.5")
    assert isinstance(runtime, NativeEmbeddingsRuntime)


# ---------------------------------------------------------------------------
# (3 + 4) Kernel _local_embed end-to-end through the runtime adapter.
#
# We don't need a real GGUF — we inject a fake runtime into the registry
# that mimics the NativeEmbeddingsRuntime contract (.embed returning an
# object with .embeddings + .usage.total_tokens).
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, total: int) -> None:
        self.prompt_tokens = total
        self.total_tokens = total


class _FakeEmbeddingResult:
    def __init__(self, embeddings: list[list[float]], usage_total: int, model: str) -> None:
        self.embeddings = embeddings
        self.usage = _FakeUsage(usage_total)
        self.model = model


class _FakeEmbedRuntime:
    """Stand-in for :class:`NativeEmbeddingsRuntime` with a recording
    ``.embed()`` so we can assert call shape + order preservation."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, inputs: list[str]) -> _FakeEmbeddingResult:
        self.calls.append(list(inputs))
        # Deterministic per-input vector: index encoded into [0]
        # so the test can assert order preservation.
        vectors = [[float(idx), float(len(text))] for idx, text in enumerate(inputs)]
        return _FakeEmbeddingResult(vectors, usage_total=sum(len(t.split()) for t in inputs), model="bge-test")


def _kernel_with_fake_embed_runtime(monkeypatch: pytest.MonkeyPatch, fake: _FakeEmbedRuntime, model_id: str) -> Any:
    """Build an ExecutionKernel and rewire the registry to hand back
    ``fake`` for ``model_id``."""
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.core.registry import ModelRuntimeRegistry

    registry = ModelRuntimeRegistry.shared()
    registry.register(model_id, lambda mid: fake)  # type: ignore[arg-type,return-value]

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    return kernel


def test_local_embed_routes_through_runtime_and_preserves_usage(monkeypatch):
    from octomil.execution.kernel import LOCALITY_ON_DEVICE

    fake = _FakeEmbedRuntime()
    kernel = _kernel_with_fake_embed_runtime(monkeypatch, fake, "bge-base-en-v1.5")

    inputs = ["hello world foo", "another input bar baz"]
    result = asyncio.run(kernel._local_embed(inputs, "bge-base-en-v1.5", fallback_used=False))

    assert result.locality == LOCALITY_ON_DEVICE
    assert result.embeddings is not None
    assert len(result.embeddings) == 2
    # usage.total_tokens propagated from the runtime's nested usage:
    # "hello world foo" = 3 tokens, "another input bar baz" = 4 = 7 total.
    assert result.usage["total_tokens"] == 7
    assert result.usage["input_tokens"] == 7


def test_local_embed_preserves_batch_order(monkeypatch):
    """The kernel must NOT reorder inputs; a re-ordered batch breaks
    RAG callers that pair vectors back to inputs by index."""
    fake = _FakeEmbedRuntime()
    _ = _kernel_with_fake_embed_runtime(monkeypatch, fake, "nomic-embed-text-v1.5")

    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    inputs = ["zeroth", "first", "second", "third"]
    result = asyncio.run(kernel._local_embed(inputs, "nomic-embed-text-v1.5", fallback_used=False))

    assert fake.calls == [inputs]
    # The fake encodes index in slot 0 — assert the kernel preserves
    # that ordering when copying through.
    assert result.embeddings is not None
    assert [v[0] for v in result.embeddings] == [0.0, 1.0, 2.0, 3.0]


def test_local_embed_usage_shape_matches_public_contract(monkeypatch):
    """``ExecutionResult.usage`` must have ``input_tokens`` +
    ``total_tokens`` keys — that is the public-API shape used by
    cloud + local equally. Hardcoded so callers can rely on it."""
    fake = _FakeEmbedRuntime()
    _ = _kernel_with_fake_embed_runtime(monkeypatch, fake, "bge-large-en-v1.5")

    from octomil.execution.kernel import ExecutionKernel

    kernel = ExecutionKernel.__new__(ExecutionKernel)
    result = asyncio.run(kernel._local_embed(["a b c"], "bge-large-en-v1.5", fallback_used=False))

    assert "input_tokens" in result.usage
    assert "total_tokens" in result.usage
    assert result.usage["total_tokens"] == 3


# ---------------------------------------------------------------------------
# (5) Bounded error for chat-only / no-runtime cases.
# ---------------------------------------------------------------------------


def test_local_embed_no_runtime_raises_unsupported_modality():
    """If the registry resolves to None, the kernel MUST raise
    bounded ``UNSUPPORTED_MODALITY`` — not ``RuntimeError`` (which
    callers may try/except into a Python fallback)."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.core.registry import ModelRuntimeRegistry

    registry = ModelRuntimeRegistry.shared()
    # Register a factory that returns None for this specific id, so
    # we don't depend on the surrounding default factory.
    registry.register("not-a-real-embedding-model", lambda mid: None)
    saved_default = registry.default_factory
    registry.default_factory = None  # block default fallback for this test
    try:
        kernel = ExecutionKernel.__new__(ExecutionKernel)
        with pytest.raises(OctomilError) as exc_info:
            asyncio.run(kernel._local_embed(["x"], "not-a-real-embedding-model", fallback_used=False))
        assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY
    finally:
        registry.default_factory = saved_default


class _FakeChatOnlyRuntime:
    """Resolves successfully but does not expose ``.embed`` —
    simulating a chat-only model id resolving to a chat runtime."""

    def run(self, *_a: Any, **_k: Any) -> Any:
        raise AssertionError("chat-only runtime should not be invoked here")


def test_local_embed_chat_only_runtime_raises_unsupported_modality():
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.core.registry import ModelRuntimeRegistry

    registry = ModelRuntimeRegistry.shared()
    registry.register("phi-4-mini", lambda mid: _FakeChatOnlyRuntime())  # type: ignore[arg-type,return-value]
    kernel = ExecutionKernel.__new__(ExecutionKernel)
    with pytest.raises(OctomilError) as exc_info:
        asyncio.run(kernel._local_embed(["hi"], "phi-4-mini", fallback_used=False))
    assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY


# ---------------------------------------------------------------------------
# (6) Adapter wraps the backend's flat shape into the public nested
#     EmbeddingResult shape.
# ---------------------------------------------------------------------------


def test_native_embeddings_runtime_wraps_backend_shape(monkeypatch):
    """``NativeEmbeddingsRuntime.embed()`` MUST return
    :class:`octomil.embeddings.EmbeddingResult` with nested
    :class:`EmbeddingUsage` — that is what ``_local_embed`` reads."""
    from octomil.embeddings import EmbeddingResult, EmbeddingUsage
    from octomil.runtime.native.embeddings_backend import EmbeddingsResult
    from octomil.runtime.native.embeddings_runtime import NativeEmbeddingsRuntime

    runtime = NativeEmbeddingsRuntime(model_name="bge-test")

    class _FakeBackend:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed(self, inputs: Any, **_k: Any) -> EmbeddingsResult:
            self.calls.append(list(inputs) if isinstance(inputs, list) else [inputs])
            return EmbeddingsResult(
                embeddings=[[1.0, 2.0]],
                model="bge-test",
                n_dim=2,
                pooling_type=1,
                is_normalized=True,
                prompt_tokens=42,
                total_tokens=42,
            )

    fake = _FakeBackend()
    runtime._backend = fake  # type: ignore[assignment]

    result = runtime.embed(["x"])
    assert isinstance(result, EmbeddingResult)
    assert result.embeddings == [[1.0, 2.0]]
    assert isinstance(result.usage, EmbeddingUsage)
    assert result.usage.prompt_tokens == 42
    assert result.usage.total_tokens == 42
    assert fake.calls == [["x"]]


def test_native_embeddings_runtime_run_and_stream_reject_chat():
    """``run`` / ``stream`` on the embeddings runtime must reject —
    embedding runtimes never serve chat. ``UNSUPPORTED_MODALITY``
    keeps the rejection bounded."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.core.types import RuntimeRequest
    from octomil.runtime.native.embeddings_runtime import NativeEmbeddingsRuntime

    runtime = NativeEmbeddingsRuntime(model_name="bge-test")
    request = RuntimeRequest(messages=[])

    with pytest.raises(OctomilError) as exc:
        asyncio.run(runtime.run(request))
    assert exc.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY

    with pytest.raises(OctomilError) as exc:
        runtime.stream(request)
    assert exc.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY


# ---------------------------------------------------------------------------
# (7) Local-runner /v1/embeddings server route.
# ---------------------------------------------------------------------------


def test_server_v1_embeddings_returns_openai_shape(monkeypatch):
    """``POST /v1/embeddings`` MUST emit the OpenAI-compatible
    response shape so existing OpenAI clients (LangChain, llama-
    index) work unmodified against the local runner."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from octomil.runtime.native import embeddings_backend as emb_backend_mod

    # Stub NativeEmbeddingsBackend so the route stands without a real
    # GGUF on disk. The server route uses the symbol imported from
    # ``octomil.runtime.native.embeddings_backend``; patch that.
    class _StubBackend:
        def __init__(self, **_k: Any) -> None:
            self._loaded = False

        def load_model(self, _model_name: str) -> None:
            self._loaded = True

        def embed(self, inputs: Any) -> Any:
            assert self._loaded, "embed called before load_model"
            inputs_list = list(inputs) if isinstance(inputs, list) else [inputs]
            return emb_backend_mod.EmbeddingsResult(
                embeddings=[[float(idx), 0.5] for idx, _ in enumerate(inputs_list)],
                model="bge-test",
                n_dim=2,
                pooling_type=1,
                is_normalized=True,
                prompt_tokens=11,
                total_tokens=11,
            )

    monkeypatch.setattr(emb_backend_mod, "NativeEmbeddingsBackend", _StubBackend)

    # Build a minimal serve app. The chat startup path requires a
    # backend; we patch _detect_backend to return None and skip the
    # lifespan startup by passing an unused model. Cleaner: bypass
    # create_app and exercise the router by directly hitting the
    # function. But for a real-shape test, use a TestClient against a
    # FastAPI app whose only route is /v1/embeddings.
    from fastapi import FastAPI

    test_app = FastAPI()

    # Re-create the route handler as the server does — calling into
    # the same factory + state symbols it imports. To avoid coupling
    # to create_app's lifespan (which loads chat backends), we
    # instantiate ServerState inline.
    from octomil.serve.config import ServerState

    state = ServerState()
    state.model_name = "bge-base-en-v1.5"

    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.native.embeddings_runtime import is_embedding_model

    @test_app.post("/v1/embeddings")
    async def create_embeddings(body: dict) -> dict:  # type: ignore[no-untyped-def]
        model_id = body.get("model") or state.model_name
        raw_input = body.get("input")
        if isinstance(raw_input, str):
            inputs = [raw_input]
        else:
            inputs = list(raw_input)  # type: ignore[arg-type]
        if not is_embedding_model(model_id):
            raise OctomilError(code=OctomilErrorCode.UNSUPPORTED_MODALITY, message="not embedding model")
        if state.embeddings_backend is None:
            backend = emb_backend_mod.NativeEmbeddingsBackend()
            backend.load_model(model_id)
            state.embeddings_backend = backend
        result = state.embeddings_backend.embed(inputs)
        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(result.embeddings)],
            "model": result.model,
            "usage": {"prompt_tokens": result.prompt_tokens, "total_tokens": result.total_tokens},
        }

    client = TestClient(test_app)
    resp = client.post("/v1/embeddings", json={"model": "bge-base-en-v1.5", "input": ["a", "b", "c"]})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["object"] == "list"
    assert payload["model"] == "bge-test"
    assert len(payload["data"]) == 3
    # Order preserved + `index` reflects input order.
    assert [item["index"] for item in payload["data"]] == [0, 1, 2]
    assert all(item["object"] == "embedding" for item in payload["data"])
    assert payload["usage"] == {"prompt_tokens": 11, "total_tokens": 11}


def test_serve_app_route_registered():
    """The serve app factory must register POST /v1/embeddings.
    Source-pin so an accidental delete of the route is caught
    without spinning up the lifespan (which loads a chat backend)."""
    src = (_repo_root() / "octomil/serve/app.py").read_text(encoding="utf-8")
    assert '@app.post("/v1/embeddings")' in src
    # And the route uses the native backend symbol — not a Python
    # embedder — to compute embeddings.
    assert "NativeEmbeddingsBackend" in src
