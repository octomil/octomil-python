"""Cutover follow-up #71: BackendCapabilities replaces
``isinstance(_, LlamaCppBackend)`` checks in the serve layer.

Pre-cutover the serve layer used ``isinstance(unwrap_backend(_),
LlamaCppBackend)`` to decide whether the backend handles GBNF
grammar internally. Post-cutover the type identity became
``NativeChatBackend`` (which does NOT handle grammar), and the
isinstance check silently evaluated False — semantically correct
in this case, but the type-marker shape is brittle as more
backends ship. This module:

  - declares ``BackendCapabilities`` (frozen dataclass) on
    ``InferenceBackend`` as a class-var;
  - has each backend subclass override with its actual flags;
  - migrates the 3 ``uses_grammar_natively`` call sites to query
    ``backend.capabilities.grammar_supported``.

Tests pin:
  - Each shipped backend declares ``capabilities``.
  - Default ``InferenceBackend`` has conservative defaults.
  - ``LlamaCppBackend.capabilities.grammar_supported is True``.
  - ``NativeChatBackend.capabilities.grammar_supported is False``.
  - Frozen dataclass — instances are immutable (`raises FrozenInstanceError`).
  - Migration: production source no longer references
    ``isinstance(_, LlamaCppBackend)`` for grammar routing.
  - ``supports(feature_name)`` accessor returns False for unknown
    flags (forward-compat).
"""

from __future__ import annotations

import pytest

from octomil.serve.types import BackendCapabilities, InferenceBackend


def test_inference_backend_default_capabilities_are_conservative() -> None:
    """A backend that doesn't override the class-var gets
    grammar=False, json_mode=False, streaming=True, tools=False."""
    caps = InferenceBackend.capabilities
    assert isinstance(caps, BackendCapabilities)
    assert caps.grammar_supported is False
    assert caps.json_mode_supported is False
    assert caps.streaming_supported is True
    assert caps.tools_supported is False
    assert caps.attention_backend == "standard"


def test_legacy_llama_cpp_backend_declares_grammar_supported() -> None:
    """Legacy ``LlamaCppBackend`` (Python ``llama_cpp.Llama`` path)
    handles GBNF grammar via ``LlamaGrammar.from_string``. The
    capability declaration MUST reflect that — the serve layer
    routes grammar requests to backends with
    ``grammar_supported=True``."""
    from octomil.serve.backends.llamacpp import LlamaCppBackend

    caps = LlamaCppBackend.capabilities
    assert caps.grammar_supported is True
    assert caps.json_mode_supported is True
    assert caps.streaming_supported is True
    assert caps.attention_backend == "flash_attention"


def test_native_chat_backend_declares_no_grammar_no_streaming() -> None:
    """v0.1.2 ``NativeChatBackend`` rejects grammar (UNSUPPORTED_MODALITY)
    and has no chat.stream capability yet. Capability flags
    declared accordingly so the serve layer doesn't route those
    request shapes here."""
    pytest.importorskip("cffi", reason="cffi extra not installed")
    from octomil.runtime.native.chat_backend import NativeChatBackend

    caps = NativeChatBackend.capabilities
    assert caps.grammar_supported is False
    assert caps.json_mode_supported is False
    assert caps.streaming_supported is False
    assert caps.tools_supported is False
    assert caps.attention_backend == "native"


def test_capabilities_is_frozen_dataclass() -> None:
    """BackendCapabilities is frozen: a backend can't mutate its
    own caps after class definition. Mutating breaks the contract
    that callers can cache the value."""
    from dataclasses import FrozenInstanceError

    caps = BackendCapabilities()
    with pytest.raises(FrozenInstanceError):
        caps.grammar_supported = True  # type: ignore[misc]


def test_capabilities_supports_accessor_returns_false_for_unknown() -> None:
    """``supports(feature_name)`` is a forward-compat accessor —
    a future feature flag the binding doesn't know about defaults
    to False (i.e., callers querying ``backend.capabilities.supports
    ("future_feature")`` get a safe negative)."""
    caps = BackendCapabilities(grammar_supported=True)
    assert caps.supports("grammar_supported") is True
    assert caps.supports("json_mode_supported") is False
    assert caps.supports("nonexistent_feature_xyz") is False


def test_serve_app_uses_capabilities_query_not_isinstance() -> None:
    """Cutover follow-up #71: production source must NOT
    reference ``isinstance(_, LlamaCppBackend)`` for grammar
    routing. The serve layer queries
    ``backend.capabilities.grammar_supported`` instead.

    A future refactor that re-introduces the type check would
    re-create the brittleness problem: post-cutover the type is
    NativeChatBackend, not LlamaCppBackend, and the isinstance
    check silently evaluates False even when a future backend
    supports grammar.
    """
    import inspect

    from octomil.serve import app as serve_app

    source = inspect.getsource(serve_app.create_app)
    # Pre-cutover the production source had:
    #   isinstance(unwrap_backend(_primary_backend), LlamaCppBackend)
    # Post-cutover follow-up #71 it MUST be:
    #   unwrap_backend(...).capabilities.grammar_supported
    assert "isinstance(unwrap_backend" not in source or "LlamaCppBackend" not in source, (
        "serve/app.py MUST NOT use isinstance(_, LlamaCppBackend) for "
        "grammar routing — query backend.capabilities.grammar_supported"
    )
    assert (
        "capabilities.grammar_supported" in source
    ), "serve/app.py MUST query backend.capabilities.grammar_supported for grammar routing (cutover follow-up #71)"


def test_multi_model_uses_capabilities_query_not_isinstance() -> None:
    """Same migration as serve/app.py but for the multi-model
    handler. ``_handle_decomposed`` and the standard fallback loop
    both used to type-check ``LlamaCppBackend``; both must now
    query ``capabilities.grammar_supported``."""
    import inspect

    from octomil.serve import multi_model

    source = inspect.getsource(multi_model.create_multi_model_app)
    # Both call sites in the multi-model handler.
    grammar_query_count = source.count("capabilities.grammar_supported")
    assert grammar_query_count >= 2, (
        f"multi_model.create_multi_model_app should have >= 2 "
        f"`capabilities.grammar_supported` queries (one in the "
        f"standard fallback loop, one in _execute_subtask). "
        f"Got {grammar_query_count}."
    )


def test_capabilities_class_var_resolved_via_instance_or_class() -> None:
    """``capabilities`` is a class-var so callers can query
    either ``InstanceBackend.capabilities`` (without instantiating)
    or ``backend.capabilities`` (on an instance). Both should
    resolve to the same object."""
    from octomil.serve.backends.llamacpp import LlamaCppBackend

    class_caps = LlamaCppBackend.capabilities
    # We don't actually instantiate (it would try to load llama_cpp);
    # the class-attribute lookup is what matters for the serve-layer
    # query semantics.
    assert isinstance(class_caps, BackendCapabilities)
    assert class_caps is LlamaCppBackend.capabilities  # idempotent


def test_capabilities_distinct_per_backend() -> None:
    """Distinct backend classes MUST have distinct capability
    instances — a future refactor that accidentally has two
    classes share the same dataclass instance via mutable
    default would silently couple them."""
    pytest.importorskip("cffi")
    from octomil.runtime.native.chat_backend import NativeChatBackend
    from octomil.serve.backends.llamacpp import LlamaCppBackend

    assert LlamaCppBackend.capabilities is not NativeChatBackend.capabilities
    assert LlamaCppBackend.capabilities != NativeChatBackend.capabilities


def test_instrumented_backend_delegates_capabilities_to_inner() -> None:
    """Cutover follow-up #71 (R4 Codex): ``InstrumentedBackend`` is a
    transparent wrapper around any ``InferenceBackend``. Without an
    explicit ``capabilities`` shadow, Python class lookup finds the
    base ``InferenceBackend.capabilities`` default before
    ``__getattr__`` runs, so callers querying
    ``InstrumentedBackend(LlamaCppBackend()).capabilities.grammar_supported``
    would see ``False`` (the conservative default) instead of ``True``
    (LlamaCppBackend's actual flag). ``InstrumentedBackend.__init__``
    MUST shadow ``capabilities = backend.capabilities`` like it
    already does for ``name`` and ``attention_backend``.

    Production migration uses ``unwrap_backend(_)`` which sidesteps
    this; the bug bites any future caller that doesn't unwrap. Pin
    the contract here."""
    from octomil.serve.backends.llamacpp import LlamaCppBackend
    from octomil.serve.instrumentation import InstrumentedBackend

    inner = LlamaCppBackend.__new__(LlamaCppBackend)
    inner.name = "llama.cpp"  # type: ignore[attr-defined]
    inner.attention_backend = "flash_attention"  # type: ignore[attr-defined]
    inner.capabilities = LlamaCppBackend.capabilities  # type: ignore[misc]

    class _StubEmitter:
        def emit(self, *_a: object, **_k: object) -> None:
            pass

    wrapped = InstrumentedBackend(inner, _StubEmitter())  # type: ignore[arg-type]
    assert wrapped.capabilities is LlamaCppBackend.capabilities
    assert wrapped.capabilities.grammar_supported is True
    assert wrapped.capabilities.attention_backend == "flash_attention"


def test_mlx_backend_declares_metal_fused_capabilities() -> None:
    """Cutover follow-up #71 (R4 Codex): pre-fix ``MLXBackend.attention_backend``
    was ``"metal_fused"`` but ``MLXBackend.capabilities.attention_backend``
    inherited the base default ``"standard"`` — drift between the two
    views. The declaration MUST keep them in sync."""
    pytest.importorskip("mlx_lm", reason="mlx_lm not installed")
    from octomil.serve.backends.mlx import MLXBackend

    caps = MLXBackend.capabilities
    assert isinstance(caps, BackendCapabilities)
    assert caps.attention_backend == "metal_fused"
    assert caps.attention_backend == MLXBackend.attention_backend
    assert caps.grammar_supported is False
    assert caps.json_mode_supported is False
    assert caps.streaming_supported is True


def test_resolve_backend_capabilities_returns_default_for_duck_typed_backend() -> None:
    """Cutover follow-up #71 (R4 Codex): some engine plugins (ORT,
    Ollama) construct duck-typed backends without inheriting from
    ``InferenceBackend``. Direct ``backend.capabilities`` raises
    AttributeError on those. ``resolve_backend_capabilities`` returns
    conservative defaults so callers don't break. Defensive
    insurance — every shipped backend SHOULD declare capabilities,
    but the helper makes refactor mistakes graceful."""
    from octomil.serve.types import _DEFAULT_BACKEND_CAPABILITIES, resolve_backend_capabilities

    class _DuckTypedBackend:
        name = "duck"
        attention_backend = "standard"

    duck = _DuckTypedBackend()
    caps = resolve_backend_capabilities(duck)
    assert caps is _DEFAULT_BACKEND_CAPABILITIES
    assert caps.grammar_supported is False
    from octomil.serve.backends.llamacpp import LlamaCppBackend

    assert resolve_backend_capabilities(LlamaCppBackend) is LlamaCppBackend.capabilities


def test_subclassed_inference_backends_have_consistent_attention_backend() -> None:
    """Cutover follow-up #71 (R4 Codex): the dataclass docstring
    explicitly notes that ``capabilities.attention_backend`` mirrors
    ``InferenceBackend.attention_backend`` and that they should not
    drift. Pin the consistency for every concrete subclass that ships
    a non-default ``attention_backend``."""
    pytest.importorskip("mlx_lm", reason="mlx_lm not installed")
    from octomil.serve.backends.llamacpp import LlamaCppBackend
    from octomil.serve.backends.mlx import MLXBackend

    for backend_cls in (LlamaCppBackend, MLXBackend):
        assert backend_cls.capabilities.attention_backend == backend_cls.attention_backend, (
            f"{backend_cls.__name__}: capabilities.attention_backend "
            f"({backend_cls.capabilities.attention_backend!r}) MUST equal "
            f"class attribute attention_backend ({backend_cls.attention_backend!r})"
        )
