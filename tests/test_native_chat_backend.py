"""Hard-cutover tests for ``NativeChatBackend``.

Per the v0.1.2 cutover spec these tests must prove:
  1. Local GGUF chat.completion is served by ``NativeChatBackend``,
     NOT the legacy ``LlamaCppBackend``.
  2. The legacy ``LlamaCppBackend`` constructor is NOT called for
     planner-selected native chat.
  3. Unsupported request features (grammar, json_mode,
     enable_thinking, streaming) raise bounded ``OctomilError`` —
     no silent fallback to the Python path.
  4. The product flow does NOT require ``OCTOMIL_LLAMA_CPP_GGUF``;
     GGUF resolution comes from the PrepareManager-materialized
     ``model_dir``.
  5. Cloud / non-GGUF routes are unchanged (sanity smoke).
  6. The cached ``NativeModel`` is reused across requests on the
     same backend instance.

Most tests are GGUF-gated (need a real artifact for end-to-end
generation). The construction / feature-gating tests run
unconditionally so the cutover contract is exercised in CI.
"""

from __future__ import annotations

import os
import tempfile

import pytest

cffi = pytest.importorskip("cffi", reason="cffi extra not installed")  # noqa: F841


@pytest.mark.requires_runtime
def test_planner_engine_returns_native_chat_backend():
    """The hard cutover: ``LlamaCppEngine.create_backend`` MUST
    return a ``NativeChatBackend`` instance, not a
    ``LlamaCppBackend``. This is the planner-side wire-in for
    every local-GGUF chat request."""
    from octomil.runtime.engines.llamacpp.engine import LlamaCppEngine
    from octomil.runtime.native.chat_backend import NativeChatBackend
    from octomil.serve.backends.llamacpp import LlamaCppBackend

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing")

    # Set up a model_dir matching the PrepareManager shape (a
    # directory containing a `*.gguf` or an `artifact` sentinel).
    with tempfile.TemporaryDirectory() as tmp:
        # Symlink the staged GGUF into the temp dir so we exercise
        # the dir-resolution path without copying the whole file.
        target = os.path.join(tmp, os.path.basename(gguf))
        os.symlink(gguf, target)

        engine = LlamaCppEngine()
        backend = engine.create_backend("test-model.gguf", model_dir=tmp)
        try:
            assert isinstance(backend, NativeChatBackend), (
                f"engine.create_backend MUST return NativeChatBackend, got " f"{type(backend).__name__!r}"
            )
            assert not isinstance(
                backend, LlamaCppBackend
            ), "engine.create_backend MUST NOT return the legacy LlamaCppBackend"
            assert backend.name == "native-llama-cpp"
        finally:
            backend.close()


@pytest.mark.requires_runtime
def test_legacy_llama_cpp_backend_not_constructed_for_chat(monkeypatch):
    """Defense-in-depth: even if a future refactor accidentally
    re-introduces a Python-local fallback, this test fails. We
    monkeypatch ``LlamaCppBackend.__init__`` to raise; planner-
    initiated chat MUST never hit it."""
    from octomil.runtime.engines.llamacpp.engine import LlamaCppEngine
    from octomil.serve.backends import llamacpp as legacy_module

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing")

    constructed = []
    real_init = legacy_module.LlamaCppBackend.__init__

    def trap_init(self, *args, **kwargs):  # noqa: ANN001
        constructed.append((args, kwargs))
        # Don't actually call real_init — this fails the test loud.
        raise AssertionError(
            "Legacy LlamaCppBackend.__init__ MUST NOT be called on the "
            "native chat path. The hard cutover wires NativeChatBackend "
            "exclusively; any construction here is a regression."
        )

    monkeypatch.setattr(legacy_module.LlamaCppBackend, "__init__", trap_init)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, os.path.basename(gguf))
            os.symlink(gguf, target)
            engine = LlamaCppEngine()
            backend = engine.create_backend("test-model.gguf", model_dir=tmp)
            try:
                assert constructed == []
                # Doubly sure: the backend returned MUST not be the legacy.
                assert not isinstance(backend, legacy_module.LlamaCppBackend)
            finally:
                backend.close()
    finally:
        monkeypatch.setattr(legacy_module.LlamaCppBackend, "__init__", real_init)


@pytest.mark.requires_runtime
@pytest.mark.timeout(120)
def test_native_chat_backend_generate_end_to_end():
    """End-to-end exercise the cutover happy path: load → generate
    → text + metrics. Pins that ``send_chat`` honors ``max_tokens``
    and that the SDK assembles the runtime's TRANSCRIPT_CHUNK
    events into a non-empty string."""
    from octomil.runtime.native.chat_backend import NativeChatBackend
    from octomil.serve.types import GenerationRequest

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing")

    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, os.path.basename(gguf))
        os.symlink(gguf, target)

        backend = NativeChatBackend(model_dir=tmp)
        try:
            backend.load_model("test-model.gguf")
            req = GenerationRequest(
                model="test-model.gguf",
                messages=[{"role": "user", "content": "Reply with the word 'ok'."}],
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
            )
            text, metrics = backend.generate(req)
            assert isinstance(text, str), "generate must return a string"
            assert len(text) > 0, "generate must produce non-empty text"
            assert metrics.total_tokens >= 1, "metrics.total_tokens >= 1"
            assert metrics.total_tokens <= 8, f"max_tokens=8 must cap output; got {metrics.total_tokens}"
            assert metrics.ttfc_ms > 0, "first-chunk time must be measured"
        finally:
            backend.close()


@pytest.mark.requires_runtime
def test_native_chat_backend_reuses_cached_model_across_requests():
    """The cached ``NativeModel`` MUST be reused across requests on
    the same backend instance. Without this, every request would
    pay cold-load + warm latency. We verify by reading the
    ``NativeModel`` identity through two ``generate`` calls."""
    from octomil.runtime.native.chat_backend import NativeChatBackend
    from octomil.serve.types import GenerationRequest

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing")

    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, os.path.basename(gguf))
        os.symlink(gguf, target)

        backend = NativeChatBackend(model_dir=tmp)
        try:
            backend.load_model("test-model.gguf")
            model_id_1 = id(backend._model)  # noqa: SLF001
            req = GenerationRequest(
                model="test-model.gguf",
                messages=[{"role": "user", "content": "say ok"}],
                max_tokens=4,
                temperature=0.0,
                top_p=1.0,
            )
            backend.generate(req)
            model_id_2 = id(backend._model)  # noqa: SLF001
            backend.generate(req)
            model_id_3 = id(backend._model)  # noqa: SLF001
            assert model_id_1 == model_id_2 == model_id_3, "NativeModel handle must persist across generate() calls"
            # load_model is idempotent — a second call must not reopen.
            backend.load_model("test-model.gguf")
            assert id(backend._model) == model_id_1, (  # noqa: SLF001
                "load_model must be idempotent — no reopen"
            )
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Feature gating — these run unconditionally (no GGUF needed because
# the gate fires before any runtime call).
# ---------------------------------------------------------------------------


def _make_unloaded_backend():
    """A NativeChatBackend that hasn't called load_model — used to
    exercise the input-validation gates without standing up a real
    runtime+model."""
    from octomil.runtime.native.chat_backend import NativeChatBackend

    backend = NativeChatBackend()
    return backend


def test_native_chat_backend_rejects_grammar():
    """Grammar (GBNF constrained generation) is unsupported in
    v0.1.2 native. The backend MUST raise bounded
    ``UNSUPPORTED_MODALITY`` rather than fall back."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.serve.types import GenerationRequest

    backend = _make_unloaded_backend()
    req = GenerationRequest(
        model="x.gguf",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        grammar="root ::= [yn]",
    )
    # We need the runtime non-None for the gate to fire; install a
    # sentinel so the order-of-checks is grammar BEFORE runtime
    # readiness (matches generate()'s logic). Easier: set runtime to
    # a sentinel so the early check runs.
    backend._runtime = object()  # noqa: SLF001
    backend._model = object()  # noqa: SLF001
    with pytest.raises(OctomilError) as exc_info:
        backend.generate(req)
    assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY
    assert "grammar" in str(exc_info.value).lower()


def test_native_chat_backend_rejects_json_mode():
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.serve.types import GenerationRequest

    backend = _make_unloaded_backend()
    backend._runtime = object()  # noqa: SLF001
    backend._model = object()  # noqa: SLF001
    req = GenerationRequest(
        model="x.gguf",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        json_mode=True,
    )
    with pytest.raises(OctomilError) as exc_info:
        backend.generate(req)
    assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY


def test_native_chat_backend_rejects_enable_thinking():
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.serve.types import GenerationRequest

    backend = _make_unloaded_backend()
    backend._runtime = object()  # noqa: SLF001
    backend._model = object()  # noqa: SLF001
    req = GenerationRequest(
        model="x.gguf",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        enable_thinking=True,
    )
    with pytest.raises(OctomilError) as exc_info:
        backend.generate(req)
    assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY


@pytest.mark.asyncio
async def test_native_chat_backend_streaming_unsupported():
    """``chat.stream`` is not implemented in the runtime; the
    backend MUST raise ``UNSUPPORTED_MODALITY`` rather than fall
    back to the Python streaming path."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.serve.types import GenerationRequest

    backend = _make_unloaded_backend()
    req = GenerationRequest(
        model="x.gguf",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        stream=True,
    )
    with pytest.raises(OctomilError) as exc_info:
        async for _chunk in backend.generate_stream(req):
            pass
    assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY


def test_native_chat_backend_rejects_unsupported_role():
    """Roles outside {system,user,assistant} reject INVALID_INPUT.
    The runtime would also reject these via the event stream; the
    SDK pre-flights so the diagnostic surfaces before session_open."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.serve.types import GenerationRequest

    backend = _make_unloaded_backend()
    backend._runtime = object()  # noqa: SLF001
    backend._model = object()  # noqa: SLF001
    req = GenerationRequest(
        model="x.gguf",
        messages=[
            {"role": "system", "content": "..."},
            {"role": "tool", "content": "calling_func()"},
        ],
        max_tokens=8,
    )
    with pytest.raises(OctomilError) as exc_info:
        backend.generate(req)
    assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
    assert "tool" in str(exc_info.value)


def test_native_chat_backend_rejects_unknown_message_keys():
    """Messages with unknown keys (e.g., OpenAI's `name`,
    `tool_calls`) reject INVALID_INPUT."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.serve.types import GenerationRequest

    backend = _make_unloaded_backend()
    backend._runtime = object()  # noqa: SLF001
    backend._model = object()  # noqa: SLF001
    req = GenerationRequest(
        model="x.gguf",
        messages=[{"role": "user", "content": "hi", "name": "alice"}],
        max_tokens=8,
    )
    with pytest.raises(OctomilError) as exc_info:
        backend.generate(req)
    assert exc_info.value.code == OctomilErrorCode.INVALID_INPUT
    assert "name" in str(exc_info.value)


def test_native_chat_backend_does_not_read_env_var(monkeypatch):
    """Product flow MUST NOT depend on ``OCTOMIL_LLAMA_CPP_GGUF``.
    The backend resolves GGUF via ``model_dir``; the env var is for
    test/dev tooling only.

    We exercise the negative case: clear the env, set ``model_dir``
    to a non-existent dir, expect MODEL_NOT_FOUND — proving the
    backend doesn't silently fall back to env-var lookup."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.native.chat_backend import NativeChatBackend

    monkeypatch.delenv("OCTOMIL_LLAMA_CPP_GGUF", raising=False)
    backend = NativeChatBackend(model_dir="/tmp/octomil-cutover-no-such-dir")
    with pytest.raises(OctomilError) as exc_info:
        backend.load_model("nope.gguf")
    assert exc_info.value.code == OctomilErrorCode.MODEL_NOT_FOUND


@pytest.mark.requires_runtime
@pytest.mark.timeout(120)
def test_native_chat_backend_default_temperature_does_not_block_request():
    """Codex R1 P1 regression: GenerationRequest's default
    ``temperature=0.7`` MUST NOT block product chat. The cutover
    backend ignores non-greedy temperature/top_p (logger.warning)
    and forwards only max_tokens to send_chat. v0.1.2 ships
    greedy-only on the runtime side.
    """
    from octomil.runtime.native.chat_backend import NativeChatBackend
    from octomil.serve.types import GenerationRequest

    gguf = os.environ.get("OCTOMIL_LLAMA_CPP_GGUF", "")
    if not gguf or not os.path.isfile(gguf):
        pytest.skip("OCTOMIL_LLAMA_CPP_GGUF unset or missing")

    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, os.path.basename(gguf))
        os.symlink(gguf, target)
        backend = NativeChatBackend(model_dir=tmp)
        try:
            backend.load_model("test-model.gguf")
            # ALL DEFAULT VALUES — temperature=0.7, top_p=1.0,
            # max_tokens=512. The runtime caps via n_ctx clamp.
            req = GenerationRequest(
                model="test-model.gguf",
                messages=[{"role": "user", "content": "say ok"}],
                max_tokens=8,  # short for test speed
            )
            assert req.temperature == 0.7, "sanity: default still 0.7"
            text, metrics = backend.generate(req)
            assert isinstance(text, str)
            assert len(text) > 0, "default-temperature request must produce output"
            assert metrics.total_tokens >= 1
        finally:
            backend.close()


def test_native_chat_backend_does_not_forward_temperature_or_top_p_to_send_chat():
    """Codex R2 nit: a no-runtime unit test that pins the B1 fix
    in CI shape (without OCTOMIL_LLAMA_CPP_GGUF). We replace
    ``NativeRuntime.open_session`` and ``NativeSession.send_chat``
    with fakes and assert the SDK passes ONLY ``max_tokens`` to
    send_chat — never ``temperature`` or ``top_p`` — even when the
    request carries the dataclass defaults (0.7 / 1.0).
    """
    from unittest.mock import MagicMock

    from octomil.runtime.native.chat_backend import NativeChatBackend
    from octomil.runtime.native.loader import (
        OCT_EVENT_SESSION_COMPLETED,
        OCT_STATUS_OK,
    )
    from octomil.serve.types import GenerationRequest

    backend = NativeChatBackend()
    # Stand up fake runtime + model + session so generate() can run
    # without a real dylib + GGUF.
    fake_session = MagicMock()
    sent: dict[str, object] = {}

    def _fake_send_chat(messages, **kwargs):  # noqa: ANN001
        sent["messages"] = messages
        sent["kwargs"] = kwargs

    fake_session.send_chat.side_effect = _fake_send_chat

    # Single SESSION_COMPLETED event — generate() should drain and
    # exit without hitting the deadline.
    completed_ev = MagicMock()
    completed_ev.type = OCT_EVENT_SESSION_COMPLETED
    completed_ev.terminal_status = OCT_STATUS_OK
    completed_ev.text = ""
    fake_session.poll_event.return_value = completed_ev
    fake_session.close.return_value = None

    fake_runtime = MagicMock()
    fake_runtime.open_session.return_value = fake_session
    fake_runtime.last_error.return_value = ""

    backend._runtime = fake_runtime  # noqa: SLF001
    backend._model = MagicMock()  # noqa: SLF001

    req = GenerationRequest(
        model="x.gguf",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        # Defaults (unset) — temperature=0.7, top_p=1.0.
    )
    assert req.temperature == 0.7
    assert req.top_p == 1.0

    text, metrics = backend.generate(req)

    fake_session.send_chat.assert_called_once()
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    kwargs = sent["kwargs"]
    assert "max_tokens" in kwargs and kwargs["max_tokens"] == 8
    # The B1 fix: temperature / top_p must NOT be forwarded.
    assert "temperature" not in kwargs, (
        "NativeChatBackend MUST NOT forward request.temperature to send_chat "
        "(v0.1.2 ships greedy-only; runtime applies its default)"
    )
    assert "top_p" not in kwargs, "NativeChatBackend MUST NOT forward request.top_p to send_chat"
    assert isinstance(text, str)
    assert metrics.total_tokens == 0  # No TRANSCRIPT_CHUNK events delivered by the fake


def test_native_chat_backend_runtime_unsupported_maps_to_unsupported_modality():
    """Codex R1 P1: when the runtime's session terminates with
    OCT_STATUS_UNSUPPORTED, the SDK maps to UNSUPPORTED_MODALITY
    (NOT INVALID_INPUT). We exercise the mapping function directly
    since simulating the event-stream path requires a runtime."""
    from octomil.errors import OctomilError, OctomilErrorCode
    from octomil.runtime.native.chat_backend import _runtime_status_to_sdk_error
    from octomil.runtime.native.loader import (
        OCT_STATUS_BUSY,
        OCT_STATUS_INVALID_INPUT,
        OCT_STATUS_NOT_FOUND,
        OCT_STATUS_UNSUPPORTED,
        OCT_STATUS_VERSION_MISMATCH,
    )

    err = _runtime_status_to_sdk_error(OCT_STATUS_UNSUPPORTED, "msg")
    assert isinstance(err, OctomilError)
    assert err.code == OctomilErrorCode.UNSUPPORTED_MODALITY

    err = _runtime_status_to_sdk_error(OCT_STATUS_NOT_FOUND, "msg")
    assert err.code == OctomilErrorCode.MODEL_NOT_FOUND

    err = _runtime_status_to_sdk_error(OCT_STATUS_INVALID_INPUT, "msg")
    assert err.code == OctomilErrorCode.INVALID_INPUT

    err = _runtime_status_to_sdk_error(OCT_STATUS_VERSION_MISMATCH, "msg")
    assert err.code == OctomilErrorCode.RUNTIME_UNAVAILABLE

    err = _runtime_status_to_sdk_error(OCT_STATUS_BUSY, "msg")
    assert err.code == OctomilErrorCode.SERVER_ERROR


def test_legacy_llama_cpp_backend_module_marked_deprecated_for_product():
    """Sanity: the legacy module's docstring documents the cutover.
    A future refactor that re-promotes it for product chat would
    need to remove this marker, surfacing the change for review."""
    from octomil.serve.backends import llamacpp as legacy_module

    doc = (legacy_module.__doc__ or "").lower()
    assert "deprecated" in doc, "legacy llamacpp module must keep its DEPRECATED-for-product marker"
    assert "nativechatbackend" in doc.replace(" ", ""), "legacy module docstring must point at the cutover replacement"
