"""SDK-internal runtime wiring guards.

These tests cover **SDK-owned** invariants only:

1. The runtime version the SDK fetches matches what the SDK is built
   against. ``scripts/fetch_runtime_dev.py`` is owned by this repo.
2. The native runtime factories the SDK installs at import time wire
   up correctly so embedding-family model ids resolve to
   ``NativeEmbeddingsRuntime`` instead of falling through to the
   chat-oriented default factory.

What does **not** belong here (and lives in the right repo instead):

* Cutover matrix / capability classification — internal team artifact;
  lives in workspace ``strategy/`` (octomil monorepo root).
* Runtime advertisement and bounded-UNSUPPORTED guards on the C ABI —
  the dylib is the system under test; lives in ``octomil-runtime``.
* Contract-enum partition completeness — tests the
  ``runtime_capability.json`` schema; lives in ``octomil-contracts``.

The "no Python local fallback reachable" guard for chat / embeddings
is already pinned in this repo by:

* ``tests/test_native_chat_backend.py:71`` —
  ``test_legacy_llama_cpp_backend_not_constructed_for_chat`` traps
  ``LlamaCppBackend.__init__`` and fails loudly if the planner ever
  re-introduces a Python-local fallback for chat.
* ``tests/test_native_embeddings_cutover.py`` (520 lines) — pins the
  embeddings cutover end-to-end.

This file does not duplicate those guards.
"""

from __future__ import annotations

import re
from pathlib import Path

EXPECTED_RUNTIME_VERSION = "v0.1.16"


def test_fetch_runtime_dev_pins_expected_version() -> None:
    """The SDK's dev-cache fetcher pins the runtime version the SDK is
    built and tested against. A silent bump in the script would mean
    the SDK starts running against an unverified runtime build; this
    test is the chokepoint that requires an explicit, reviewed
    version change."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "fetch_runtime_dev.py"
    assert script.is_file(), f"fetch_runtime_dev.py missing at {script}"
    text = script.read_text(encoding="utf-8")
    match = re.search(r'^DEFAULT_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "DEFAULT_VERSION assignment not found in fetch_runtime_dev.py"
    assert match.group(1) == EXPECTED_RUNTIME_VERSION, (
        f"SDK expects runtime {EXPECTED_RUNTIME_VERSION}, but "
        f"fetch_runtime_dev.py pins {match.group(1)!r}. Bump "
        f"EXPECTED_RUNTIME_VERSION in this test only after the parity "
        f"gate has been run against the new runtime."
    )


def test_native_embeddings_factory_registers_every_family_prefix() -> None:
    """``register_native_embeddings_factory`` is what
    ``octomil/runtime/__init__.py:_connect_native_embeddings`` calls at
    SDK import to wire the embeddings.text product binding. The function
    must install ``native_embeddings_factory`` for every prefix in
    ``_EMBEDDING_FAMILY_PREFIXES``; otherwise
    ``ModelRuntimeRegistry.resolve("bge-…")`` would fall through to the
    default factory (the chat path), silently mis-routing embedding
    requests.

    The test calls ``register_native_embeddings_factory`` explicitly
    rather than relying on the import-time side effect, because 11 other
    tests in the suite call ``ModelRuntimeRegistry.shared().clear()`` —
    in pytest-xdist that wipes the import-time registrations on the
    worker before this test runs. Re-invoking the function is the same
    call ``_connect_native_embeddings`` makes (re-registration is
    idempotent per the function's docstring), so the assertion still
    proves the wiring contract.
    """
    from octomil.runtime.core.registry import ModelRuntimeRegistry
    from octomil.runtime.native.embeddings_runtime import (
        _EMBEDDING_FAMILY_PREFIXES,
        native_embeddings_factory,
        register_native_embeddings_factory,
    )

    register_native_embeddings_factory()

    families = ModelRuntimeRegistry.shared()._families  # noqa: SLF001
    missing = [p for p in _EMBEDDING_FAMILY_PREFIXES if p.lower() not in families]
    assert not missing, (
        f"Embedding family prefixes not registered in ModelRuntimeRegistry: "
        f"{missing!r}. register_native_embeddings_factory ran but did not "
        f"install the factory for every prefix in _EMBEDDING_FAMILY_PREFIXES."
    )
    for prefix in _EMBEDDING_FAMILY_PREFIXES:
        registered = families[prefix.lower()]
        assert registered is native_embeddings_factory, (
            f"prefix {prefix!r} is bound to {registered!r}, not "
            f"native_embeddings_factory. The cutover requires the native "
            f"runtime to be the exclusive product binding for embeddings.text."
        )
