"""Cutover follow-up #71 (R1 Codex): grammar-capability gating in the
serve layer.

R1 Codex review of the BackendCapabilities migration surfaced two
silent semantic violations the original isinstance(LlamaCppBackend)
shape papered over:

  1. **Explicit caller-supplied GBNF silently stripped.** A request
     with ``body.grammar`` set was forwarded to the planner-routed
     backend with ``grammar=None`` whenever the backend's
     ``capabilities.grammar_supported=False``. Pre-cutover this was a
     dead code path (LlamaCppBackend always supported grammar);
     post-cutover (chat→native) every explicit GBNF request returned
     200 OK with text that ignored the requested constraint. Should
     raise ``UNSUPPORTED_MODALITY`` instead.

  2. **json_mode forwarded after system-prompt fallback.** When the
     serve layer falls back to system-prompt JSON nudging (because
     the backend can't constrain grammar internally), it then ALSO
     forwarded ``json_mode=True`` to the backend. NativeChatBackend's
     ``_gate_unsupported_request_features`` raises
     ``UNSUPPORTED_MODALITY`` on ``json_mode=True``, so every JSON
     request to native turned into a 422 even though the fallback
     was supposed to make it work. Should send ``json_mode=False``
     after fallback (the system prompt does the constraining).

This module pins the helper-level gating contract. Integration via
the FastAPI test client lives in test_serve_error_mapping.py /
test_serve.py.
"""

from __future__ import annotations

import pytest

from octomil.errors import OctomilError, OctomilErrorCode
from octomil.serve.grammar_helpers import _reject_explicit_grammar_on_non_grammar_backend


def test_explicit_grammar_against_non_grammar_backend_raises_unsupported_modality() -> None:
    """When ``body.grammar`` is set explicitly (not derived from
    response_format) and the routed backend declares
    ``grammar_supported=False``, the helper raises
    ``OctomilError(UNSUPPORTED_MODALITY)``. Pre-fix the serve layer
    silently stripped the grammar, returning 200 OK with output that
    ignored the constraint."""
    with pytest.raises(OctomilError) as exc_info:
        _reject_explicit_grammar_on_non_grammar_backend(
            backend_name="native",
            grammar_str='root ::= "yes" | "no"',
            is_json=False,
            uses_grammar_natively=False,
        )
    assert exc_info.value.code == OctomilErrorCode.UNSUPPORTED_MODALITY
    msg = str(exc_info.value)
    assert "grammar_supported=False" in msg
    assert "native" in msg


def test_explicit_grammar_against_grammar_capable_backend_does_not_raise() -> None:
    """The legacy ``LlamaCppBackend`` declares
    ``grammar_supported=True``; explicit GBNF is forwarded
    transparently. The helper is a no-op in that branch."""
    _reject_explicit_grammar_on_non_grammar_backend(
        backend_name="llama.cpp",
        grammar_str='root ::= "yes" | "no"',
        is_json=False,
        uses_grammar_natively=True,
    )


def test_json_mode_against_non_grammar_backend_does_not_raise() -> None:
    """JSON-mode requests (``is_json=True``, grammar derived from
    ``response_format``) fall back to system-prompt nudging on non-
    grammar backends — the helper MUST NOT reject them. Only explicit
    caller-supplied GBNF triggers the rejection.

    Pre-fix the gating logic conflated the two: any non-empty
    grammar_str + grammar_supported=False raised. That broke JSON
    mode for native, the post-cutover default chat backend."""
    _reject_explicit_grammar_on_non_grammar_backend(
        backend_name="native",
        grammar_str='root ::= "{" string ":" value "}"',  # derived JSON GBNF
        is_json=True,
        uses_grammar_natively=False,
    )


def test_no_grammar_no_json_no_op() -> None:
    """The common path — no body.grammar, no response_format — calls
    the helper with grammar_str=None. Helper is a no-op."""
    _reject_explicit_grammar_on_non_grammar_backend(
        backend_name="native",
        grammar_str=None,
        is_json=False,
        uses_grammar_natively=False,
    )


def test_grammar_helpers_imports_helper_from_grammar_helpers_module() -> None:
    """Importability guard: the helper lives in serve.grammar_helpers
    so both serve/app.py and serve/multi_model.py can import it
    without a circular dependency back through serve/types.py."""
    from octomil.serve import grammar_helpers

    assert hasattr(grammar_helpers, "_reject_explicit_grammar_on_non_grammar_backend")
    assert callable(grammar_helpers._reject_explicit_grammar_on_non_grammar_backend)


def test_serve_app_calls_grammar_capability_gate() -> None:
    """Textual migration guard: ``create_app`` must call the gating
    helper in the chat-completions handler so explicit caller GBNF
    against a non-grammar backend raises rather than being silently
    stripped. A future refactor that forgets the call would re-create
    the silent-200 bug Codex caught at R1."""
    import inspect

    from octomil.serve import app as serve_app

    source = inspect.getsource(serve_app.create_app)
    assert "_reject_explicit_grammar_on_non_grammar_backend" in source, (
        "serve/app.py MUST call _reject_explicit_grammar_on_non_grammar_backend "
        "to surface UNSUPPORTED_MODALITY for explicit body.grammar against "
        "non-grammar backends (cutover follow-up #71 R1 Codex)"
    )


def test_serve_multi_model_calls_grammar_capability_gate_at_both_sites() -> None:
    """Both grammar-routing call sites in the multi-model handler
    (standard fallback loop + ``_execute_subtask`` decomposed path)
    must invoke the gating helper. >=2 occurrences pinned."""
    import inspect

    from octomil.serve import multi_model

    source = inspect.getsource(multi_model.create_multi_model_app)
    gate_count = source.count("_reject_explicit_grammar_on_non_grammar_backend")
    assert gate_count >= 2, (
        f"multi_model.create_multi_model_app should call "
        f"_reject_explicit_grammar_on_non_grammar_backend at >=2 sites "
        f"(standard fallback loop + _execute_subtask). Got {gate_count}."
    )


def test_serve_app_does_not_forward_json_mode_after_prompt_fallback() -> None:
    """Textual migration guard: ``create_app`` must gate
    ``json_mode=`` on ``uses_grammar_natively`` so that after the
    system-prompt fallback (which fires when the backend can't
    constrain grammar), we don't ALSO send ``json_mode=True`` to the
    backend. Native rejects ``json_mode=True`` with
    UNSUPPORTED_MODALITY, so forwarding the flag turned every JSON
    request into a 422 post-cutover."""
    import inspect

    from octomil.serve import app as serve_app

    source = inspect.getsource(serve_app.create_app)
    # Must NOT pass plain `json_mode=is_json` — that's the bug.
    # Must pass `json_mode=is_json and uses_grammar_natively`.
    assert "json_mode=is_json and uses_grammar_natively" in source, (
        "serve/app.py MUST forward json_mode only when "
        "uses_grammar_natively=True; after system-prompt fallback the "
        "backend should generate without a json_mode flag (cutover "
        "follow-up #71 R1 Codex)"
    )


def _extract_retry_req_block(source: str) -> str:
    """Slice the ``retry_req = GenerationRequest(...)`` constructor body
    by paren-balancing — the literal first ``)`` belongs to an inner
    call (e.g., ``max(gen_req.temperature - 0.2, 0.0)``)."""
    start = source.find("retry_req = GenerationRequest(")
    assert start > -1, "retry_req block must exist"
    open_paren = source.find("(", start)
    depth = 0
    for i, ch in enumerate(source[open_paren:], start=open_paren):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError("unbalanced parens in retry_req constructor")


def test_serve_app_retry_path_does_not_forward_json_mode() -> None:
    """Cutover follow-up #71 (R2 Codex): the JSON-mode retry block in
    ``serve/app.py`` only fires in the ``is_json and not
    uses_grammar_natively`` branch — by construction the non-grammar
    fallback. Pre-fix the retry hard-coded ``json_mode=True``, which
    forwarded the flag to a backend that just rejected it on the
    first attempt — every JSON request that needed a retry on native
    would 422. The retry MUST send ``json_mode=False`` so the system-
    prompt nudging carries the constraint."""
    import inspect

    from octomil.serve import app as serve_app

    source = inspect.getsource(serve_app.create_app)
    retry_block = _extract_retry_req_block(source)
    assert "json_mode=False" in retry_block, (
        "serve/app.py retry block MUST send json_mode=False — the retry only "
        "fires for non-grammar backends, and json_mode=True 422s native "
        "(cutover follow-up #71 R2 Codex)"
    )
    assert "json_mode=True" not in retry_block, "serve/app.py retry block MUST NOT send json_mode=True"


def test_serve_multi_model_retry_path_does_not_forward_json_mode() -> None:
    """Same retry-block gating in multi_model.py's standard fallback
    loop — the only place a retry is wired in the multi-model
    handler. ``_execute_subtask`` (decomposed path) does not have a
    retry block; the standard fallback loop does."""
    import inspect

    from octomil.serve import multi_model

    source = inspect.getsource(multi_model.create_multi_model_app)
    retry_block = _extract_retry_req_block(source)
    assert (
        "json_mode=False" in retry_block
    ), "multi_model.py retry block MUST send json_mode=False (cutover follow-up #71 R2 Codex)"
    assert "json_mode=True" not in retry_block


def test_serve_multi_model_does_not_forward_json_mode_after_prompt_fallback_at_both_sites() -> None:
    """Same json_mode gating must apply in both multi-model call
    sites. >=2 occurrences pinned."""
    import inspect

    from octomil.serve import multi_model

    source = inspect.getsource(multi_model.create_multi_model_app)
    gate_count = source.count("json_mode=is_json and uses_grammar_natively")
    assert gate_count >= 2, (
        f"multi_model.create_multi_model_app should gate "
        f"json_mode=is_json on uses_grammar_natively at >=2 sites. "
        f"Got {gate_count}."
    )
