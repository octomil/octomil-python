"""v0.1.3 — Native embeddings hard-cutover tests.

Pins the cutover discipline that wires
:class:`NativeEmbeddingsBackend` into the kernel + local-runner
paths. Sibling to ``test_native_embeddings_backend.py`` (which pins
the backend itself); these tests pin the SDK boundary:

  1. Source-pin: no Python embedder import path
     (``sentence_transformers`` / ``llama_cpp.LlamaEmbedding`` /
     ``fastembed``) is referenced from the embedding code paths.
  2. Registry resolution: embedding-family ids resolve to
     :class:`NativeEmbeddingsRuntime` only when an artifact (real
     PrepareManager dir or env override) is available — capability
     honesty so cloud fallback engages when no artifact exists.
  3. Factory caches by model_id (mirrors chat path) so repeat
     resolutions reuse the warmed runtime.
  4. HuggingFace org-prefixed ids (``BAAI/bge-...``) match the
     family gate.
  5. ``_local_embed`` calls ``runtime.embed(inputs)`` and surfaces
     ``EmbeddingResult.embeddings`` + ``usage.total_tokens`` into
     :class:`ExecutionResult`.
  6. Batch order is preserved end-to-end through the bridge.
  7. Chat-only model id surfaces a bounded
     :class:`OctomilError(UNSUPPORTED_MODALITY)`.
  8. Adapter wraps backend's flat shape into the public nested
     :class:`EmbeddingResult` / :class:`EmbeddingUsage` shape.
  9. ``run`` and ``stream`` reject ``UNSUPPORTED_MODALITY``.
     ``stream`` is an async generator so the rejection is observed
     via ``async for`` (not a sync raise at call site).
 10. Concurrent ``embed()`` calls don't double-load the backend
     (lazy-load lock).
 11. ``/v1/embeddings`` server route registered, source-pinned.

Hard-cutover discipline: every reject path is a bounded
:class:`OctomilError`. There is no fallback.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixtures: keep the global registry + factory cache from leaking across
# tests. Both pieces are process-wide singletons, so without snapshot/restore
# a test that rebinds 'phi-4-mini' or warms a cache entry would change
# behavior in unrelated tests later in the same process.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_registry_and_cache():
    from octomil.runtime.core.registry import ModelRuntimeRegistry
    from octomil.runtime.native import embeddings_runtime as emb_rt

    registry = ModelRuntimeRegistry.shared()
    saved_families = dict(registry._families)
    saved_default = registry.default_factory
    saved_cache = dict(emb_rt._runtime_cache)
    try:
        yield
    finally:
        emb_rt.reset_runtime_cache()
        registry._families.clear()
        registry._families.update(saved_families)
        registry.default_factory = saved_default
        emb_rt._runtime_cache.clear()
        emb_rt._runtime_cache.update(saved_cache)


# ---------------------------------------------------------------------------
# (1) Source-pin: no Python embedder fallback in the cutover code paths.
# ---------------------------------------------------------------------------


_BANNED_EMBEDDER_IMPORTS = (
    "sentence_transformers",
    "fastembed",
    "llama_cpp.LlamaEmbedding",
)

_CUTOVER_SOURCES = (
    "octomil/runtime/native/embeddings_backend.py",
    "octomil/runtime/native/embeddings_runtime.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_no_python_embedder_fallback_in_native_paths():
    root = _repo_root()
    for src_relative in _CUTOVER_SOURCES:
        text = (root / src_relative).read_text(encoding="utf-8")
        for banned in _BANNED_EMBEDDER_IMPORTS:
            assert banned not in text, f"{src_relative} unexpectedly references {banned!r}"


def test_kernel_local_embed_does_not_import_python_embedder():
    src = (_repo_root() / "octomil/execution/kernel.py").read_text(encoding="utf-8")
    for banned in _BANNED_EMBEDDER_IMPORTS:
        assert banned not in src, f"octomil/execution/kernel.py unexpectedly references {banned!r}"


def test_serve_app_does_not_import_python_embedder():
    src = (_repo_root() / "octomil/serve/app.py").read_text(encoding="utf-8")
    for banned in _BANNED_EMBEDDER_IMPORTS:
        assert banned not in src, f"octomil/serve/app.py unexpectedly references {banned!r}"


# ---------------------------------------------------------------------------
# (2) Registry resolution + capability honesty.
# ---------------------------------------------------------------------------


def test_factory_returns_none_without_artifact(monkeypatch):
    """Capability honesty: with no PrepareManager artifact AND no env
    override, the factory MUST return None so ``_can_local`` reports
    False and configured cloud fallback engages."""
    from octomil.runtime.native import embeddings_runtime as emb_rt
    from octomil.runtime.native.embeddings_runtime import native_embeddings_factory

    monkeypatch.delenv("OCTOMIL_NATIVE_EMBEDDINGS_MODEL_DIR", raising=False)
    monkeypatch.setattr(emb_rt, "_resolve_prepared_model_dir", lambda _mid: None)

    assert native_embeddings_factory("nomic-embed-text-v1.5") is None
    assert native_embeddings_factory("bge-base-en-v1.5") is None


def test_factory_returns_runtime_with_env_override(monkeypatch, tmp_path):
    from octomil.runtime.native import embeddings_runtime as emb_rt
    from octomil.runtime.native.embeddings_runtime import (
        NativeEmbeddingsRuntime,
        native_embeddings_factory,
    )

    monkeypatch.setenv("OCTOMIL_NATIVE_EMBEDDINGS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(emb_rt, "_resolve_prepared_model_dir", lambda _mid: None)

    runtime = native_embeddings_factory("bge-base-en-v1.5")
    assert isinstance(runtime, NativeEmbeddingsRuntime)


def test_registry_resolves_embedding_family_with_env(monkeypatch, tmp_path):
    from octomil.runtime.core.registry import ModelRuntimeRegistry
    from octomil.runtime.native import embeddings_runtime as emb_rt
    from octomil.runtime.native.embeddings_runtime import (
        NativeEmbeddingsRuntime,
        register_native_embeddings_factory,
    )

    monkeypatch.setenv("OCTOMIL_NATIVE_EMBEDDINGS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(emb_rt, "_resolve_prepared_model_dir", lambda _mid: None)
    register_native_embeddings_factory()
    runtime = ModelRuntimeRegistry.shared().resolve("nomic-embed-text-v1.5")
    assert isinstance(runtime, NativeEmbeddingsRuntime)


def test_factory_caches_by_model_id(monkeypatch, tmp_path):
    """Repeat factory calls for the same model_id MUST return the
    same runtime instance — otherwise every embedding call re-loads
    the GGUF (multi-second perf regression)."""
    from octomil.runtime.native import embeddings_runtime as emb_rt
    from octomil.runtime.native.embeddings_runtime import native_embeddings_factory

    monkeypatch.setenv("OCTOMIL_NATIVE_EMBEDDINGS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(emb_rt, "_resolve_prepared_model_dir", lambda _mid: None)

    r1 = native_embeddings_factory("bge-base-en-v1.5")
    r2 = native_embeddings_factory("bge-base-en-v1.5")
    assert r1 is r2

    # Different model_id → different instance.
    r3 = native_embeddings_factory("nomic-embed-text-v1.5")
    assert r3 is not r1


def test_is_embedding_model_handles_hf_org_prefix():
    """HuggingFace ids carry an 'org/' prefix (BAAI, mixedbread-ai,
    Snowflake). The family gate must recognize them; otherwise a
    user passing the canonical HF id falls through to the chat
    default factory."""
    from octomil.runtime.native.embeddings_runtime import is_embedding_model

    assert is_embedding_model("BAAI/bge-base-en-v1.5")
    assert is_embedding_model("mixedbread-ai/mxbai-embed-large-v1")
    assert is_embedding_model("nomic-ai/nomic-embed-text-v1.5")
    assert is_embedding_model("intfloat/e5-mistral-7b-instruct")
    # Without org prefix, still works.
    assert is_embedding_model("bge-base-en-v1.5")
    # Filesystem paths do NOT match (more than one slash).
    assert not is_embedding_model("/local/path/to/bge.gguf")


def test_is_embedding_model_e5_prefix_is_specific():
    """Bare 'e5-' was too broad in round 1 — could match a future
    'e5-coder' chat model. Tightened to specific embedding shapes."""
    from octomil.runtime.native.embeddings_runtime import is_embedding_model

    assert is_embedding_model("e5-mistral-7b-instruct")
    assert is_embedding_model("e5-base-v2")
    assert is_embedding_model("e5-large-v2")
    assert is_embedding_model("e5-small")
    # A plausible future chat/coder name with the 'e5-' bare prefix
    # MUST NOT silently route through the embedding factory.
    assert not is_embedding_model("e5-coder")


def test_registry_returns_non_embedding_factory_to_none():
    from octomil.runtime.native.embeddings_runtime import native_embeddings_factory

    assert native_embeddings_factory("phi-4-mini") is None


# ---------------------------------------------------------------------------
# (5 + 6) Kernel _local_embed end-to-end through the runtime adapter.
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
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, inputs: list[str]) -> _FakeEmbeddingResult:
        self.calls.append(list(inputs))
        vectors = [[float(idx), float(len(text))] for idx, text in enumerate(inputs)]
        return _FakeEmbeddingResult(vectors, usage_total=sum(len(t.split()) for t in inputs), model="bge-test")


def _kernel_with_fake_embed_runtime(fake: _FakeEmbedRuntime, model_id: str) -> Any:
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.core.registry import ModelRuntimeRegistry

    registry = ModelRuntimeRegistry.shared()
    registry.register(model_id, lambda mid: fake)  # type: ignore[arg-type,return-value]
    return ExecutionKernel.__new__(ExecutionKernel)


def test_local_embed_routes_through_runtime_and_preserves_usage():
    from octomil.execution.kernel import LOCALITY_ON_DEVICE

    fake = _FakeEmbedRuntime()
    kernel = _kernel_with_fake_embed_runtime(fake, "bge-base-en-v1.5")
    inputs = ["hello world foo", "another input bar baz"]
    result = asyncio.run(kernel._local_embed(inputs, "bge-base-en-v1.5", fallback_used=False))

    assert result.locality == LOCALITY_ON_DEVICE
    assert result.embeddings is not None
    assert len(result.embeddings) == 2
    assert result.usage["total_tokens"] == 7
    assert result.usage["input_tokens"] == 7


def test_local_embed_preserves_batch_order():
    fake = _FakeEmbedRuntime()
    kernel = _kernel_with_fake_embed_runtime(fake, "nomic-embed-text-v1.5")
    inputs = ["zeroth", "first", "second", "third"]
    result = asyncio.run(kernel._local_embed(inputs, "nomic-embed-text-v1.5", fallback_used=False))
    assert fake.calls == [inputs]
    assert result.embeddings is not None
    assert [v[0] for v in result.embeddings] == [0.0, 1.0, 2.0, 3.0]


def test_local_embed_usage_shape_matches_public_contract():
    fake = _FakeEmbedRuntime()
    kernel = _kernel_with_fake_embed_runtime(fake, "bge-large-en-v1.5")
    result = asyncio.run(kernel._local_embed(["a b c"], "bge-large-en-v1.5", fallback_used=False))
    assert "input_tokens" in result.usage
    assert "total_tokens" in result.usage
    assert result.usage["total_tokens"] == 3


# ---------------------------------------------------------------------------
# (7) Bounded errors.
# ---------------------------------------------------------------------------


def test_local_embed_no_runtime_raises_unsupported_modality():
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.execution.kernel import ExecutionKernel
    from octomil.runtime.core.registry import ModelRuntimeRegistry

    registry = ModelRuntimeRegistry.shared()
    registry.register("not-a-real-embedding-model", lambda mid: None)
    registry.default_factory = None
    kernel = ExecutionKernel.__new__(ExecutionKernel)
    with pytest.raises(OctomilError) as exc_info:
        asyncio.run(kernel._local_embed(["x"], "not-a-real-embedding-model", fallback_used=False))
    assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY


class _FakeChatOnlyRuntime:
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
# (8 + 9) Adapter shape + run/stream rejection.
# ---------------------------------------------------------------------------


def test_native_embeddings_runtime_wraps_backend_shape():
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


def test_native_embeddings_runtime_run_rejects_chat():
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.core.types import RuntimeRequest
    from octomil.runtime.native.embeddings_runtime import NativeEmbeddingsRuntime

    runtime = NativeEmbeddingsRuntime(model_name="bge-test")
    with pytest.raises(OctomilError) as exc:
        asyncio.run(runtime.run(RuntimeRequest(messages=[])))
    assert exc.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY


def test_native_embeddings_runtime_stream_rejects_chat_via_async_for():
    """stream() is an async generator — the rejection must be
    observable via ``async for``, NOT a sync raise at call site.
    Callers wrap their iteration in try/except; sync raise breaks
    that pattern."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.core.types import RuntimeRequest
    from octomil.runtime.native.embeddings_runtime import NativeEmbeddingsRuntime

    runtime = NativeEmbeddingsRuntime(model_name="bge-test")

    async def _consume():
        async for _ in runtime.stream(RuntimeRequest(messages=[])):
            pass

    with pytest.raises(OctomilError) as exc:
        asyncio.run(_consume())
    assert exc.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY


# ---------------------------------------------------------------------------
# (10) Concurrent lazy-load doesn't double-load the backend.
# ---------------------------------------------------------------------------


def test_ensure_backend_serializes_concurrent_load():
    """Two threads racing through the lazy-load path must NOT both
    construct + load_model. The first wins; the second waits and
    reuses. Without the lock, the loser's backend leaks (cffi
    runtime + GGUF mmap never closed)."""
    from octomil.runtime.native.embeddings_runtime import NativeEmbeddingsRuntime

    construct_count = 0
    load_count = 0
    lock = threading.Lock()

    class _SlowBackend:
        def __init__(self, **_k: Any) -> None:
            nonlocal construct_count
            with lock:
                construct_count += 1

        def load_model(self, _name: str) -> None:
            nonlocal load_count
            time.sleep(0.05)  # simulate a slow GGUF load
            with lock:
                load_count += 1

        def close(self) -> None:
            pass

        def embed(self, inputs: Any) -> Any:
            raise NotImplementedError

    runtime = NativeEmbeddingsRuntime(model_name="bge-test")

    # Patch the backend class the runtime constructs.
    import octomil.runtime.native.embeddings_runtime as emb_rt

    original = emb_rt.NativeEmbeddingsBackend
    emb_rt.NativeEmbeddingsBackend = _SlowBackend  # type: ignore[misc,assignment]
    try:
        threads = [threading.Thread(target=runtime._ensure_backend) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        emb_rt.NativeEmbeddingsBackend = original  # type: ignore[misc]

    assert construct_count == 1, f"construct_count={construct_count}, expected 1"
    assert load_count == 1, f"load_count={load_count}, expected 1"


# ---------------------------------------------------------------------------
# (11) Server route registered + uses native backend symbol.
# ---------------------------------------------------------------------------


def test_serve_app_route_registered():
    src = (_repo_root() / "octomil/serve/app.py").read_text(encoding="utf-8")
    assert '@app.post("/v1/embeddings")' in src
    # Route must go through the registry (not direct backend
    # construction) so factory's prepared-artifact resolution is
    # used.
    assert "ModelRuntimeRegistry.shared().resolve" in src
    # Sync embed must run in a worker thread so it doesn't block
    # the asyncio event loop.
    assert "asyncio.to_thread(runtime.embed" in src


def test_serve_route_does_not_close_previous_runtime_in_request_path():
    """Round-3 fix: the route MUST NOT call ``previous.close()`` when
    swapping the active runtime. Closing while a concurrent request
    is mid-embed (in a worker thread) would free a backend that's
    actively serving inference. Cache reset happens once at lifespan
    teardown instead."""
    src = (_repo_root() / "octomil/serve/app.py").read_text(encoding="utf-8")
    # The literal call shape we removed.
    assert "previous.close()" not in src


def test_serve_lifespan_resets_runtime_cache():
    """Lifespan teardown MUST drain the runtime cache via
    ``reset_runtime_cache()`` so warmed cffi runtimes / GGUF mmaps
    are released at shutdown."""
    src = (_repo_root() / "octomil/serve/app.py").read_text(encoding="utf-8")
    assert "reset_runtime_cache" in src


# ---------------------------------------------------------------------------
# (12) Registry resolution for HF org-prefixed ids reaches the factory.
#      Round-3 fix: the registry's literal-prefix matcher needs the
#      ``<org>/<family>`` variants registered too — without them,
#      ``BAAI/bge-...`` passes ``is_embedding_model`` but never reaches
#      the factory and falls through to the chat default.
# ---------------------------------------------------------------------------


def test_registry_reaches_factory_for_hf_org_prefixed_ids(monkeypatch, tmp_path):
    from octomil.runtime.core.registry import ModelRuntimeRegistry
    from octomil.runtime.native import embeddings_runtime as emb_rt
    from octomil.runtime.native.embeddings_runtime import (
        NativeEmbeddingsRuntime,
        register_native_embeddings_factory,
    )

    monkeypatch.setenv("OCTOMIL_NATIVE_EMBEDDINGS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(emb_rt, "_resolve_prepared_model_dir", lambda _mid: None)
    register_native_embeddings_factory()

    for hf_id in (
        "BAAI/bge-base-en-v1.5",
        "nomic-ai/nomic-embed-text-v1.5",
        "intfloat/e5-mistral-7b-instruct",
        "mixedbread-ai/mxbai-embed-large-v1",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
    ):
        runtime = ModelRuntimeRegistry.shared().resolve(hf_id)
        assert isinstance(
            runtime, NativeEmbeddingsRuntime
        ), f"HF id {hf_id!r} did not resolve to NativeEmbeddingsRuntime"
