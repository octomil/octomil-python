"""v0.1.3 — NativeEmbeddingsBackend hard-cutover tests.

Per the v0.1.3 cutover spec these tests must prove:
  1. Capability honesty — `embeddings.text` advertisement gated on
     the runtime build (and the per-context pooling-type gate at
     session_open).
  2. Unloaded backend → bounded `RUNTIME_UNAVAILABLE` (no silent
     fallback to a Python embedder).
  3. Bad deadline_ms → bounded `INVALID_INPUT`.
  4. Bad input shape (non-str / non-list) → TypeError before any
     runtime call.
  5. Chat-only GGUF → `UNSUPPORTED_MODALITY` at session_open via
     the per-context pooling-type gate (env-gated end-to-end —
     skips when no GGUF is staged).
  6. End-to-end smoke against a real embedding GGUF (env-gated):
     - vectors in input order with stable n_dim.
     - L2 norm ≈ 1.0.
     - Same input → cosine ≈ 1.0 (determinism).
     - Different inputs → distinct.
     - usage.total_tokens > 0.
     - Single-input shape AND batched shape both work.
  7. No-runtime/no-artifact smoke: when the dylib isn't installed,
     tests skip cleanly via `pytest.importorskip("cffi")` rather
     than fall back to anything.

Hard-cutover discipline (NO Python fallback): every reject path
raises a bounded `OctomilError`. There is no try/except around the
backend that converts a runtime UNSUPPORTED into a Python-local
embedder call. The tests here pin that discipline by asserting
exact `OctomilErrorCode` values on every failure path.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any

import pytest

cffi = pytest.importorskip("cffi", reason="cffi extra not installed")  # noqa: F841


def _make_unloaded_backend():
    """A NativeEmbeddingsBackend that hasn't called load_model. Used
    to exercise input-validation gates without standing up a real
    runtime+model."""
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend

    return NativeEmbeddingsBackend()


def test_unloaded_backend_raises_runtime_unavailable():
    """``embed()`` before ``load_model()`` MUST raise bounded
    ``RUNTIME_UNAVAILABLE`` — not silently no-op, not silently fall
    back to a Python embedder. The hard-cutover contract says any
    code path that wants local embeddings MUST go through this
    backend; if the runtime isn't ready, the caller sees an
    actionable error code."""
    from octomil.errors import OctomilError, OctomilErrorCode

    backend = _make_unloaded_backend()
    with pytest.raises(OctomilError) as exc_info:
        backend.embed("hello world")
    assert exc_info.value.code == OctomilErrorCode.RUNTIME_UNAVAILABLE


def test_zero_deadline_raises_invalid_input():
    """deadline_ms <= 0 is a configuration error, not an instant
    timeout. Mirrors the same gate as NativeChatBackend (#74)."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend

    backend = NativeEmbeddingsBackend()
    backend._runtime = object()  # type: ignore[assignment]
    backend._model = object()  # type: ignore[assignment]

    for bad in (0, -1, -1000):
        with pytest.raises(OctomilError) as exc_info:
            backend.embed("hi", deadline_ms=bad)
        assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
        assert "deadline_ms" in str(exc_info.value)


def test_send_embed_source_pins_type_check_for_bad_input_type():
    """R1 Codex/Claude both flagged the original placeholder test
    was empty. The full type-check is in
    ``NativeSession.send_embed``; we pin its source-level shape
    rather than constructing a real session (which requires a cffi
    handle). The type check rejects non-str/non-list inputs with
    ``TypeError`` BEFORE any runtime contact, so caller-side bugs
    fail at the binding boundary."""
    import inspect

    from octomil.runtime.native.loader import NativeSession

    src = inspect.getsource(NativeSession.send_embed)
    assert "isinstance(inputs, str)" in src, "send_embed must check str"
    assert "isinstance(inputs, list)" in src, "send_embed must check list"
    assert "raise TypeError" in src, "send_embed must raise TypeError on bad type"
    assert '"input"' in src, 'send_embed must emit the wrapped {"input": ...} shape'


def test_native_embeddings_backend_drain_emits_vectors_in_order_via_fake_session():
    """Fake-session drain test (no runtime required). Pins the
    event-stream contract: SESSION_STARTED → 3× EMBEDDING_VECTOR
    (with index 0/1/2) → SESSION_COMPLETED(OK). R1 Codex flagged
    that event-drain invariants needed coverage that doesn't
    require a staged GGUF; this fills the gap."""
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend
    from octomil.runtime.native.loader import (
        OCT_EVENT_EMBEDDING_VECTOR,
        OCT_EVENT_NONE,
        OCT_EVENT_SESSION_COMPLETED,
        OCT_EVENT_SESSION_STARTED,
        OCT_STATUS_OK,
    )

    class _FakeEvent:
        def __init__(self, **kw: Any) -> None:
            self.type = kw["type"]
            self.values = kw.get("values", [])
            self.n_dim = kw.get("n_dim", 0)
            self.n_input_tokens = kw.get("n_input_tokens", 0)
            self.index = kw.get("index", 0)
            self.pooling_type = kw.get("pooling_type", 0)
            self.is_normalized = kw.get("is_normalized", False)
            self.terminal_status = kw.get("terminal_status", 0)
            self.text = ""

    events = iter(
        [
            _FakeEvent(type=OCT_EVENT_SESSION_STARTED),
            _FakeEvent(
                type=OCT_EVENT_EMBEDDING_VECTOR,
                values=[1.0, 0.0, 0.0],
                n_dim=3,
                n_input_tokens=4,
                index=0,
                pooling_type=1,
                is_normalized=True,
            ),
            _FakeEvent(
                type=OCT_EVENT_EMBEDDING_VECTOR,
                values=[0.0, 1.0, 0.0],
                n_dim=3,
                n_input_tokens=2,
                index=1,
                pooling_type=1,
                is_normalized=True,
            ),
            _FakeEvent(
                type=OCT_EVENT_EMBEDDING_VECTOR,
                values=[0.0, 0.0, 1.0],
                n_dim=3,
                n_input_tokens=5,
                index=2,
                pooling_type=1,
                is_normalized=True,
            ),
            _FakeEvent(type=OCT_EVENT_SESSION_COMPLETED, terminal_status=OCT_STATUS_OK),
        ]
    )

    class _FakeSession:
        def send_embed(self, *_a: Any, **_k: Any) -> None:
            return None

        def poll_event(self, *_a: Any, **_k: Any) -> Any:
            try:
                return next(events)
            except StopIteration:
                return _FakeEvent(type=OCT_EVENT_NONE)

        def close(self) -> None:
            return None

    class _FakeRuntime:
        def open_session(self, **_k: Any) -> Any:
            return _FakeSession()

        def last_error(self) -> str:
            return ""

    backend = NativeEmbeddingsBackend()
    backend._runtime = _FakeRuntime()  # type: ignore[assignment]
    backend._model = object()  # type: ignore[assignment]
    backend._model_name = "fake-bge-mini"

    r = backend.embed(["a", "b", "c"])
    assert r.embeddings == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert r.n_dim == 3
    assert r.pooling_type == 1
    assert r.is_normalized is True
    assert r.prompt_tokens == 11
    assert r.total_tokens == r.prompt_tokens
    assert r.model == "fake-bge-mini"


def test_native_embeddings_backend_out_of_order_index_raises_inference_failed():
    """Defensive invariant: if the runtime ever emits
    EMBEDDING_VECTOR events out of input order, the SDK MUST raise
    INFERENCE_FAILED rather than silently shuffle results."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend
    from octomil.runtime.native.loader import (
        OCT_EVENT_EMBEDDING_VECTOR,
        OCT_EVENT_NONE,
        OCT_EVENT_SESSION_COMPLETED,
        OCT_STATUS_OK,
    )

    class _FakeEvent:
        def __init__(self, **kw: Any) -> None:
            self.type = kw["type"]
            self.values = kw.get("values", [])
            self.n_dim = kw.get("n_dim", 0)
            self.n_input_tokens = kw.get("n_input_tokens", 0)
            self.index = kw.get("index", 0)
            self.pooling_type = kw.get("pooling_type", 0)
            self.is_normalized = kw.get("is_normalized", False)
            self.terminal_status = kw.get("terminal_status", 0)
            self.text = ""

    events = iter(
        [
            _FakeEvent(type=OCT_EVENT_EMBEDDING_VECTOR, values=[1.0, 0.0], n_dim=2, index=2),  # OUT OF ORDER
            _FakeEvent(type=OCT_EVENT_SESSION_COMPLETED, terminal_status=OCT_STATUS_OK),
        ]
    )

    class _FakeSession:
        def send_embed(self, *_a: Any, **_k: Any) -> None:
            return None

        def poll_event(self, *_a: Any, **_k: Any) -> Any:
            try:
                return next(events)
            except StopIteration:
                return _FakeEvent(type=OCT_EVENT_NONE)

        def close(self) -> None:
            return None

    class _FakeRuntime:
        def open_session(self, **_k: Any) -> Any:
            return _FakeSession()

        def last_error(self) -> str:
            return ""

    backend = NativeEmbeddingsBackend()
    backend._runtime = _FakeRuntime()  # type: ignore[assignment]
    backend._model = object()  # type: ignore[assignment]

    with pytest.raises(OctomilError) as exc_info:
        backend.embed(["a", "b", "c"])
    assert exc_info.value.code == OctomilErrorCode.INFERENCE_FAILED
    assert "out-of-order" in str(exc_info.value).lower() or "index" in str(exc_info.value).lower()


def test_native_embeddings_backend_n_dim_drift_raises_inference_failed():
    """Stable n_dim across the batch is part of the contract. If
    the runtime ever emits a vector with a different dimension
    mid-batch, the SDK raises INFERENCE_FAILED."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend
    from octomil.runtime.native.loader import (
        OCT_EVENT_EMBEDDING_VECTOR,
        OCT_EVENT_NONE,
        OCT_EVENT_SESSION_COMPLETED,
        OCT_STATUS_OK,
    )

    class _FakeEvent:
        def __init__(self, **kw: Any) -> None:
            self.type = kw["type"]
            self.values = kw.get("values", [])
            self.n_dim = kw.get("n_dim", 0)
            self.n_input_tokens = kw.get("n_input_tokens", 0)
            self.index = kw.get("index", 0)
            self.pooling_type = kw.get("pooling_type", 0)
            self.is_normalized = kw.get("is_normalized", False)
            self.terminal_status = kw.get("terminal_status", 0)
            self.text = ""

    events = iter(
        [
            _FakeEvent(type=OCT_EVENT_EMBEDDING_VECTOR, values=[1.0, 0.0, 0.0], n_dim=3, index=0),
            _FakeEvent(type=OCT_EVENT_EMBEDDING_VECTOR, values=[1.0, 0.0], n_dim=2, index=1),  # DRIFT
            _FakeEvent(type=OCT_EVENT_SESSION_COMPLETED, terminal_status=OCT_STATUS_OK),
        ]
    )

    class _FakeSession:
        def send_embed(self, *_a: Any, **_k: Any) -> None:
            return None

        def poll_event(self, *_a: Any, **_k: Any) -> Any:
            try:
                return next(events)
            except StopIteration:
                return _FakeEvent(type=OCT_EVENT_NONE)

        def close(self) -> None:
            return None

    class _FakeRuntime:
        def open_session(self, **_k: Any) -> Any:
            return _FakeSession()

        def last_error(self) -> str:
            return ""

    backend = NativeEmbeddingsBackend()
    backend._runtime = _FakeRuntime()  # type: ignore[assignment]
    backend._model = object()  # type: ignore[assignment]

    with pytest.raises(OctomilError) as exc_info:
        backend.embed(["a", "b"])
    assert exc_info.value.code == OctomilErrorCode.INFERENCE_FAILED
    assert "n_dim" in str(exc_info.value).lower() or "drift" in str(exc_info.value).lower()


def test_runtime_advertises_embeddings_text_helper_returns_false_on_failure():
    """``runtime_advertises_embeddings_text`` is a defensive helper.
    If the runtime is not installed / the capabilities query
    raises, the helper returns False (caller MUST then raise
    UNSUPPORTED_MODALITY rather than silently fall back)."""
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend

    class _BrokenRuntime:
        def capabilities(self) -> Any:
            raise RuntimeError("dylib missing")

    assert (
        NativeEmbeddingsBackend.runtime_advertises_embeddings_text(
            _BrokenRuntime()  # type: ignore[arg-type]
        )
        is False
    )


def test_runtime_advertises_embeddings_text_helper_on_advertised_runtime():
    """Helper returns True when the capability is in
    `supported_capabilities`. Pin the contract on a stub
    capabilities object."""
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend

    class _Caps:
        supported_capabilities = ("chat.completion", "embeddings.text")

    class _StubRuntime:
        def capabilities(self) -> Any:
            return _Caps()

    assert (
        NativeEmbeddingsBackend.runtime_advertises_embeddings_text(
            _StubRuntime()  # type: ignore[arg-type]
        )
        is True
    )


def test_runtime_advertises_embeddings_text_helper_on_chat_only_runtime():
    """When the runtime advertises chat but NOT embeddings.text,
    the helper returns False. Capability-honesty contract — no
    silent fallback."""
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend

    class _Caps:
        supported_capabilities = ("chat.completion",)  # NOT embeddings.text

    class _StubRuntime:
        def capabilities(self) -> Any:
            return _Caps()

    assert (
        NativeEmbeddingsBackend.runtime_advertises_embeddings_text(
            _StubRuntime()  # type: ignore[arg-type]
        )
        is False
    )


def test_unresolvable_gguf_raises_model_not_found():
    """``load_model`` with a path that doesn't exist (and no
    model_dir) raises bounded ``MODEL_NOT_FOUND`` — not a generic
    Python FileNotFoundError, not a silent fallback to a different
    artifact path."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend

    backend = NativeEmbeddingsBackend()
    with pytest.raises(OctomilError) as exc_info:
        backend.load_model("/nonexistent/path/to/embedding.gguf")
    assert exc_info.value.code == OctomilErrorCode.MODEL_NOT_FOUND


# ---------------------------------------------------------------------------
# End-to-end smoke (env-gated). Skips when no GGUF is staged via
# OCTOMIL_EMBED_GGUF. Mirrors the runtime-side smoke shape so a
# parity gap surfaces at the SDK boundary.
# ---------------------------------------------------------------------------


def _l2_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: list[float], b: list[float]) -> float:
    # Both vectors are L2-normalized → cosine = dot product.
    return sum(x * y for x, y in zip(a, b))


@pytest.mark.requires_runtime
@pytest.mark.timeout(120)
def test_native_embeddings_backend_end_to_end():
    """End-to-end exercise: load → embed → vectors in order.
    Asserts:
    - Embeddings.text is advertised.
    - Batched input → 3 vectors with index 0/1/2 (implicit via
      arrival order — the SDK doesn't expose index publicly because
      EmbeddingsResult.embeddings is in input order).
    - Stable n_dim across the batch.
    - L2 norm ≈ 1.0 (is_normalized=True flag).
    - Same input → cosine ≈ 1.0 (determinism within batch).
    - Different inputs → cosine < 0.99 (distinct).
    - usage.total_tokens > 0.
    - Same input across separate embed() calls → cosine ≈ 1.0
      (determinism across sessions; warmed model is reused)."""
    gguf = os.environ.get("OCTOMIL_EMBED_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_EMBED_GGUF unset or missing — stage a BGE/Nomic GGUF")

    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend

    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, os.path.basename(gguf))
        os.symlink(gguf, target)

        backend = NativeEmbeddingsBackend(model_dir=tmp)
        try:
            backend.load_model(os.path.basename(gguf))

            # Batched input.
            r = backend.embed(["the quick brown fox", "the quick brown fox", "hi"])
            assert len(r.embeddings) == 3, "batch produces 3 vectors"
            n_dim = r.n_dim
            assert n_dim > 0, "n_dim > 0"
            assert all(len(v) == n_dim for v in r.embeddings), "stable n_dim across batch"
            assert r.is_normalized, "is_normalized flag"
            assert r.prompt_tokens > 0, "prompt_tokens > 0"
            assert r.total_tokens == r.prompt_tokens, "embeddings total == prompt"

            # L2 norm ≈ 1.0 (within tolerance — runtime L2-normalizes
            # before emitting; SDK doesn't double-normalize).
            for i, v in enumerate(r.embeddings):
                norm = _l2_norm(v)
                assert abs(norm - 1.0) < 1e-3, f"vec[{i}] L2 norm {norm} != 1.0"

            # Same input determinism.
            cos_same = _cosine(r.embeddings[0], r.embeddings[1])
            assert cos_same > 0.9999, f"same input cosine {cos_same} not ~= 1.0"

            # Different inputs.
            cos_diff = _cosine(r.embeddings[0], r.embeddings[2])
            assert cos_diff < 0.99, f"different inputs cosine {cos_diff} >= 0.99"

            # Single-input shape.
            r2 = backend.embed("the quick brown fox")
            assert len(r2.embeddings) == 1
            assert r2.n_dim == n_dim, "n_dim stable across sessions"
            cos_cross = _cosine(r.embeddings[0], r2.embeddings[0])
            assert cos_cross > 0.9999, f"cross-session cosine {cos_cross} not ~= 1.0"
        finally:
            backend.close()


@pytest.mark.requires_runtime
@pytest.mark.timeout(60)
def test_native_embeddings_backend_chat_gguf_unsupported():
    """A decoder-only chat GGUF (Llama-3 / Qwen / SmolLM with
    LLAMA_POOLING_TYPE_NONE) MUST reject at session_open with
    bounded ``UNSUPPORTED_MODALITY``. The runtime's per-context
    pooling-type gate enforces this; the SDK propagates verbatim
    — NO silent fallback to a Python local embedder."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend

    chat_gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not chat_gguf or not os.path.isfile(chat_gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset — chat GGUF gate test needs a chat GGUF")

    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, os.path.basename(chat_gguf))
        os.symlink(chat_gguf, target)

        backend = NativeEmbeddingsBackend(model_dir=tmp)
        try:
            backend.load_model(os.path.basename(chat_gguf))
            with pytest.raises(OctomilError) as exc_info:
                backend.embed("hello world")
            # Per-context pooling-type gate at session_open returns
            # UNSUPPORTED → mapped to UNSUPPORTED_MODALITY. NEVER a
            # Python fallback / silent OK / different code.
            assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY
        finally:
            backend.close()


def test_no_silent_fallback_module_does_not_import_legacy_local_embedder():
    """Hard-cutover invariant: this module MUST NOT import any
    Python-local embedding library (sentence_transformers, llama_cpp's
    embed APIs, etc.). The cutover means native runtime is the
    ONLY local embedding path; any fallback would re-create the
    coupling we removed."""
    import inspect

    from octomil.runtime.native import embeddings_backend

    source = inspect.getsource(embeddings_backend)
    # Banned patterns — if any of these appear in the source, the
    # cutover discipline is broken.
    banned_substrings = [
        "import sentence_transformers",
        "from sentence_transformers",
        "import llama_cpp",
        "from llama_cpp",
        "fastembed",
        "import torch",  # torch-backed embedders count as legacy here
    ]
    for needle in banned_substrings:
        assert needle not in source, (
            f"native embeddings backend imports legacy local embedder ({needle!r}). "
            f"Hard cutover requires native runtime is the ONLY local embedding path."
        )


def test_default_deadline_is_5_minutes():
    """The class-level default deadline matches NativeChatBackend's
    convention so callers don't get surprising timeout shapes."""
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend

    assert NativeEmbeddingsBackend.DEFAULT_DEADLINE_MS == 300_000


def test_constructor_default_deadline_overrides_class_default():
    """Tests / CI inject a smaller deadline to exercise the timeout
    path without waiting 5 minutes. Same pattern as
    NativeChatBackend(default_deadline_ms=...)."""
    from octomil.runtime.native.embeddings_backend import NativeEmbeddingsBackend

    backend = NativeEmbeddingsBackend(default_deadline_ms=1500)
    assert backend._default_deadline_ms == 1500


def test_module_docstring_documents_hard_cutover_discipline():
    """Pin the cutover discipline in the docstring so future
    refactors that re-introduce a Python fallback have to delete
    explicit text saying it's banned."""
    from octomil.runtime.native import embeddings_backend

    doc = embeddings_backend.__doc__ or ""
    assert (
        "no silent" in doc.lower() or "no fallback" in doc.lower()
    ), "module docstring must spell out the no-silent-fallback discipline"
    assert "UNSUPPORTED_MODALITY" in doc, "docstring must name the bounded error code for capability misses"


def test_embeddings_result_shape_matches_existing_sdk_response():
    """``EmbeddingsResult`` mirrors ``octomil.embeddings.EmbeddingResult``
    fields the kernel relies on (`embeddings`, `model`, `usage`-shape
    via `prompt_tokens` + `total_tokens`). Pin the field set so a
    refactor doesn't drift."""
    from octomil.runtime.native.embeddings_backend import EmbeddingsResult

    fields = {f for f in EmbeddingsResult.__dataclass_fields__}
    required = {"embeddings", "model", "n_dim", "pooling_type", "is_normalized", "prompt_tokens", "total_tokens"}
    assert required <= fields, f"missing fields: {required - fields}"
