"""Hard-cutover invariants — enforces the matrix in
``docs/native-cutover-matrix.md``.

Two classes of guard:

1. **Static** (always run): pin the runtime version, verify the SDK's
   side of the wiring is intact, verify the contract-side capability
   enum still names every BLOCKED capability so the runtime's strict-
   reject path on session-open keeps the bounded-error shape.

2. **`requires_runtime`** (skip when no dylib available): assert the
   actual ABI behavior — advertised set is exactly the DONE rows;
   any BLOCKED capability rejects at ``oct_session_open`` with
   ``OCT_STATUS_UNSUPPORTED`` and a non-empty ``last_error``.

Failure of any of these means the matrix has drifted from reality
and a cutover PR must update either the implementation or the matrix.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Expected matrix state. Keep in sync with docs/native-cutover-matrix.md.
# Updating either side without the other should fail one of the tests below.
# ---------------------------------------------------------------------------

EXPECTED_RUNTIME_VERSION = "v0.1.4"

# Capabilities the runtime advertises today. The runtime advertises chat
# streaming via ``OCT_EVENT_TRANSCRIPT_CHUNK`` events on the same
# ``chat.completion`` session, so ``chat.stream`` does NOT appear here.
DONE_CAPABILITIES: frozenset[str] = frozenset(
    {
        "chat.completion",
        "embeddings.text",
    }
)

# Capabilities that have a contract enum entry but no native adapter.
# Opening a session against any of these MUST return UNSUPPORTED.
BLOCKED_CAPABILITIES: frozenset[str] = frozenset(
    {
        "audio.diarization",
        "audio.realtime.session",
        "audio.speaker.embedding",
        "audio.stt.batch",
        "audio.stt.stream",
        "audio.transcription",
        "audio.tts.batch",
        "audio.tts.stream",
        "audio.vad",
        "chat.stream",
        "embeddings.image",
        "index.vector.query",
    }
)


# ---------------------------------------------------------------------------
# Static guards (run unconditionally)
# ---------------------------------------------------------------------------


def test_fetch_runtime_dev_pins_expected_version() -> None:
    """`scripts/fetch_runtime_dev.py` is the only place the dev binding
    pulls a runtime release from. The matrix doc claims v0.1.4 is the
    consumed version; this test pins it so a silent bump in the fetch
    script doesn't drift the matrix."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "fetch_runtime_dev.py"
    assert script.is_file(), f"fetch_runtime_dev.py missing at {script}"
    text = script.read_text(encoding="utf-8")
    match = re.search(r'^DEFAULT_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "DEFAULT_VERSION assignment not found in fetch_runtime_dev.py"
    assert match.group(1) == EXPECTED_RUNTIME_VERSION, (
        f"docs/native-cutover-matrix.md pins runtime {EXPECTED_RUNTIME_VERSION}, "
        f"but fetch_runtime_dev.py pins {match.group(1)!r}. Either update the matrix "
        f"+ this test (after running the parity gate against the new runtime), or "
        f"revert the script."
    )


def test_capabilities_enum_covers_done_and_blocked_sets() -> None:
    """The SDK-side capability allowlist (``RUNTIME_CAPABILITIES``) is the
    forward-compat reader filter. It must include every capability the
    matrix mentions — DONE rows so the parsed view doesn't silently
    drop them, BLOCKED rows so request-side strict-reject names the
    capability instead of failing some unrelated validator."""
    from octomil.runtime.native.capabilities import RUNTIME_CAPABILITIES

    expected = DONE_CAPABILITIES | BLOCKED_CAPABILITIES
    missing = expected - RUNTIME_CAPABILITIES
    assert not missing, (
        f"RUNTIME_CAPABILITIES is missing {sorted(missing)!r}; either the matrix is "
        f"out of date or the contract enum was reduced. Bump capabilities.py "
        f"and the contract schema together."
    )


def test_native_embeddings_factory_registered_on_sdk_import() -> None:
    """``octomil/runtime/__init__.py:_connect_native_embeddings`` runs
    at import time. Its job is to register a runtime factory for every
    embedding-family prefix so ``ModelRuntimeRegistry.resolve(model)``
    returns a ``NativeEmbeddingsRuntime`` for those ids — that is the
    embeddings.text product wiring. If this hook silently no-ops the
    cutover regresses.

    Resolving the runtime end-to-end requires a PrepareManager-materialized
    artifact (capability honesty: the factory returns ``None`` without one),
    so we instead pin the factory registration directly: every embedding
    prefix from ``_EMBEDDING_FAMILY_PREFIXES`` must map to
    ``native_embeddings_factory`` in the shared registry.
    """
    import octomil.runtime as _runtime  # noqa: F401  (triggers _connect calls)
    from octomil.runtime.core.registry import ModelRuntimeRegistry
    from octomil.runtime.native.embeddings_runtime import (
        _EMBEDDING_FAMILY_PREFIXES,
        native_embeddings_factory,
    )

    families = ModelRuntimeRegistry.shared()._families  # noqa: SLF001
    missing = [p for p in _EMBEDDING_FAMILY_PREFIXES if p.lower() not in families]
    assert not missing, (
        f"Embedding family prefixes not registered in ModelRuntimeRegistry: "
        f"{missing!r}. _connect_native_embeddings ran but did not install the "
        f"factory — likely an import-time exception was swallowed by the "
        f"try/except in octomil/runtime/__init__.py."
    )
    for prefix in _EMBEDDING_FAMILY_PREFIXES:
        registered = families[prefix.lower()]
        assert registered is native_embeddings_factory, (
            f"prefix {prefix!r} is bound to {registered!r}, not "
            f"native_embeddings_factory. The cutover requires the native "
            f"runtime to be the exclusive product binding for embeddings.text."
        )


# ---------------------------------------------------------------------------
# Runtime-bound guards (skip without a dylib)
# ---------------------------------------------------------------------------


@pytest.mark.requires_runtime
def test_runtime_advertises_only_done_capabilities() -> None:
    """``oct_runtime_capabilities`` MUST report exactly the DONE rows.

    This is the ``no fake native capability advertisement`` rule
    (cutover spec hard rule 3). If the runtime ever advertises a
    capability without a real adapter, ``open_session`` would happily
    accept it and produce undefined behavior; this test trips first.
    """
    from octomil.runtime.native.loader import NativeRuntime

    with NativeRuntime.open() as rt:
        caps = rt.capabilities()
        advertised = frozenset(caps.supported_capabilities)

    extra = advertised - DONE_CAPABILITIES
    missing = DONE_CAPABILITIES - advertised
    assert not extra, (
        f"runtime advertises capabilities outside the DONE_NATIVE_CUTOVER set: "
        f"{sorted(extra)!r}. Either a real adapter landed (in which case update "
        f"DONE_CAPABILITIES + the matrix doc + run the parity gate) or the "
        f"runtime is fabricating advertisement."
    )
    assert not missing, (
        f"runtime is missing capabilities documented as DONE_NATIVE_CUTOVER: "
        f"{sorted(missing)!r}. Likely a runtime regression or a build that "
        f"didn't link the llama.cpp adapter."
    )


@pytest.mark.requires_runtime
@pytest.mark.parametrize("capability", sorted(BLOCKED_CAPABILITIES))
def test_blocked_capability_rejects_with_bounded_unsupported(capability: str) -> None:
    """Every BLOCKED capability MUST reject at ``oct_session_open`` with
    ``OCT_STATUS_UNSUPPORTED`` and a non-empty runtime ``last_error``.

    This is the ``unsupported features must reject with bounded errors``
    rule (cutover spec hard rule 4). Silent fall-through, generic
    ``OCT_STATUS_INTERNAL``, or empty last_error all fail.
    """
    from octomil.runtime.native.loader import (
        OCT_STATUS_UNSUPPORTED,
        NativeRuntime,
        NativeRuntimeError,
    )

    with NativeRuntime.open() as rt:
        with pytest.raises(NativeRuntimeError) as excinfo:
            rt.open_session(capability=capability)

    err = excinfo.value
    assert err.status == OCT_STATUS_UNSUPPORTED, (
        f"open_session(capability={capability!r}) returned status "
        f"{err.status}; expected OCT_STATUS_UNSUPPORTED. The runtime is "
        f"either crashing or returning a non-bounded code for an "
        f"unimplemented capability."
    )
    assert err.last_error, (
        f"open_session(capability={capability!r}) returned UNSUPPORTED "
        f"with empty runtime last_error. Bounded-error rejection requires "
        f"a diagnostic string per the v0.4 ABI contract."
    )
