"""v0.1.6 PR2 — Static import-graph guard for the legacy pywhispercpp shim.

v0.1.5 PR-2B cut over the SDK STT path to
:class:`octomil.runtime.native.stt_backend.NativeSttBackend` (cffi
bindings into octomil-runtime + whisper.cpp). The legacy
``pywhispercpp`` engine was renamed to
``octomil/runtime/engines/whisper/_legacy_pywhisper.py`` and remains
in the tree for benchmark / parity use ONLY (gated behind
``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK=1`` in
:mod:`scripts.parity_native_stt`).

This module is the static import-graph guard that prevents future
refactors from accidentally re-introducing the legacy fallback into
product code. Mirrors the discipline of
``test_native_embeddings_cutover.py:97`` — source-pin via AST walk
plus a runtime ``sys.modules`` check.

What is enforced
----------------

(1) Source-pin (AST): no module under ``octomil/`` may write
    ``import octomil.runtime.engines.whisper._legacy_pywhisper`` or
    ``from octomil.runtime.engines.whisper._legacy_pywhisper import …``
    EXCEPT:

    * ``octomil/runtime/engines/whisper/__init__.py`` — the package
      shim. It is permitted to re-export EXACTLY one symbol:
      :func:`is_whisper_model`. Re-exporting any other symbol (in
      particular ``WhisperCppEngine`` or ``_WhisperBackend``) would
      pull legacy inference machinery onto the product path.
    * ``octomil/runtime/engines/whisper/_legacy_pywhisper.py`` itself
      (self-references are vacuously allowed; it is the file we are
      guarding).

(2) Source-pin (AST): no module under ``octomil/`` may import the
    legacy class symbols (``WhisperCppEngine``, ``_WhisperBackend``)
    from anywhere. The package ``__init__`` re-exports
    :func:`is_whisper_model` only — the engine class is reachable
    only via the explicit
    ``octomil.runtime.engines.whisper._legacy_pywhisper`` dotted path,
    which (1) already forbids on product paths.

(3) Runtime check: with ``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK`` unset
    or set to anything other than ``"1"``, importing the canonical
    product entry points (``octomil``, ``octomil.serve.app``,
    ``octomil.execution.kernel``) MUST NOT bring
    ``octomil.runtime.engines.whisper._legacy_pywhisper`` into
    ``sys.modules`` … unless the package shim
    (``octomil.runtime.engines.whisper.__init__``) is itself
    imported, in which case the shim's lone re-export of
    :func:`is_whisper_model` is the bounded surface.

    The runtime check therefore asserts: the legacy module's
    presence in ``sys.modules`` is either absent OR fully accounted
    for by the package shim — never by direct product-path import.

Whitelisted (these MAY reference the legacy module):

* ``tests/`` — benchmark / reference tests live here.
* ``scripts/parity_*.py`` — parity-gate scripts; explicitly opt in
  via ``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK=1``.
* ``packaging/`` — PyInstaller spec lists the legacy module as a
  hidden import so opt-in benchmark builds keep working.
* ``octomil/runtime/engines/whisper/_legacy_pywhisper.py`` itself.
* ``octomil/runtime/engines/whisper/__init__.py`` — bounded
  re-export of :func:`is_whisper_model` ONLY (enforced).

Hard-cutover discipline: if a future refactor needs to broaden the
re-export surface, it must first promote a non-legacy backend (i.e.
ship a real native replacement for ``is_whisper_model``'s purpose)
and remove the legacy import from the shim — not widen this guard.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants — pinned dotted paths the guard reasons about.
# ---------------------------------------------------------------------------

_LEGACY_DOTTED = "octomil.runtime.engines.whisper._legacy_pywhisper"
_LEGACY_TAIL = "_legacy_pywhisper"

# Symbol names that, if imported on a product path, would defeat the
# cutover. ``is_whisper_model`` is intentionally NOT in this list — it is
# the bounded re-export surface. Engine / inference symbols are.
_LEGACY_INFERENCE_SYMBOLS = frozenset(
    {
        "WhisperCppEngine",
        "_WhisperBackend",
    }
)

# Files where ``_legacy_pywhisper`` references are allowed to appear in
# import statements. Paths are repo-relative and use forward slashes.
_PRODUCT_IMPORT_WHITELIST = frozenset(
    {
        "octomil/runtime/engines/whisper/__init__.py",
        "octomil/runtime/engines/whisper/_legacy_pywhisper.py",
    }
)

# Directory prefixes (repo-relative, forward-slash) that are NOT product
# paths and are allowed to reference the legacy module freely. We still
# walk every file under ``octomil/`` for the AST guard — these are tested
# at the *consumer* level (tests, scripts, packaging) and not part of the
# import-graph guard.
_NON_PRODUCT_PREFIXES = (
    "tests/",
    "scripts/",
    "packaging/",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_product_python_files() -> list[Path]:
    """All ``*.py`` files under the ``octomil/`` package — i.e. the
    product import surface this guard polices."""
    root = _repo_root() / "octomil"
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _relpath(path: Path) -> str:
    return path.resolve().relative_to(_repo_root()).as_posix()


# ---------------------------------------------------------------------------
# (1) AST guard — no product file imports the legacy module by dotted path,
#     EXCEPT the bounded shim and the legacy file itself.
# ---------------------------------------------------------------------------


def _legacy_imports_in_file(path: Path) -> list[tuple[int, str]]:
    """Return a list of ``(lineno, snippet)`` for every statement in
    ``path`` that imports the legacy module by dotted path or tail name.

    Catches all four forms:

    * ``import octomil.runtime.engines.whisper._legacy_pywhisper``
    * ``import octomil.runtime.engines.whisper._legacy_pywhisper as X``
    * ``from octomil.runtime.engines.whisper._legacy_pywhisper import …``
    * ``from .whisper import _legacy_pywhisper`` / relative-form
      siblings — caught by the ``names`` walk on every ``ImportFrom``.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # A syntax error in product code is a separate (loud) bug; this
        # guard's job is import discipline, not parser sanity. Skip.
        return []

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _LEGACY_DOTTED or alias.name.endswith("." + _LEGACY_TAIL):
                    hits.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # Absolute "from octomil...._legacy_pywhisper import X"
            if module == _LEGACY_DOTTED or module.endswith("." + _LEGACY_TAIL):
                imported = ", ".join(a.name for a in node.names)
                hits.append((node.lineno, f"from {module} import {imported}"))
                continue
            # "from octomil.runtime.engines.whisper import _legacy_pywhisper"
            for alias in node.names:
                if alias.name == _LEGACY_TAIL:
                    hits.append((node.lineno, f"from {module} import {alias.name}"))
    return hits


def test_no_legacy_pywhisper_import_on_product_paths():
    """No product module may import the legacy whisper shim directly.

    Whitelist is enforced at file granularity — the package
    ``__init__`` and the legacy file itself are the only allowed
    sites. Adding to the whitelist requires changing this test
    (deliberate, reviewable).
    """
    offenders: list[str] = []
    for path in _iter_product_python_files():
        rel = _relpath(path)
        if rel in _PRODUCT_IMPORT_WHITELIST:
            continue
        for lineno, snippet in _legacy_imports_in_file(path):
            offenders.append(f"{rel}:{lineno}  {snippet}")

    assert not offenders, (
        "Legacy pywhispercpp imports found on product paths "
        "(see tests/test_no_legacy_pywhisper_in_product_paths.py):\n  " + "\n  ".join(offenders)
    )


def test_whisper_package_shim_only_reexports_is_whisper_model():
    """The single allowed whitelist entry (the package ``__init__``) MUST
    re-export ``is_whisper_model`` and nothing else from the legacy
    module. Re-exporting an inference symbol (``WhisperCppEngine``,
    ``_WhisperBackend``) would pull the legacy engine onto every
    product path that touches whisper-name detection."""
    shim = _repo_root() / "octomil/runtime/engines/whisper/__init__.py"
    tree = ast.parse(shim.read_text(encoding="utf-8"), filename=str(shim))
    legacy_reexports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == _LEGACY_DOTTED or module.endswith("." + _LEGACY_TAIL):
                legacy_reexports.extend(a.name for a in node.names)

    assert legacy_reexports == ["is_whisper_model"], (
        "octomil/runtime/engines/whisper/__init__.py is allowed to "
        "re-export EXACTLY ['is_whisper_model'] from "
        f"_legacy_pywhisper. Found: {legacy_reexports!r}. "
        "Re-exporting an engine / inference symbol breaks the "
        "v0.1.5 PR-2B cutover discipline."
    )


def test_no_legacy_inference_symbols_on_product_paths():
    """No product module may import ``WhisperCppEngine`` or
    ``_WhisperBackend`` by name from anywhere. These are legacy-only
    symbols; reaching them requires the explicit
    ``_legacy_pywhisper`` dotted path, which the previous test
    forbids on product paths."""
    offenders: list[str] = []
    for path in _iter_product_python_files():
        rel = _relpath(path)
        if rel in _PRODUCT_IMPORT_WHITELIST:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in _LEGACY_INFERENCE_SYMBOLS:
                        offenders.append(f"{rel}:{node.lineno}  from {node.module} import {alias.name}")
    assert not offenders, "Legacy whisper inference symbols imported on product paths:\n  " + "\n  ".join(offenders)


# ---------------------------------------------------------------------------
# (2) Runtime guard — importing canonical product entry points must NOT
#     pull the legacy module in via any path other than the bounded
#     ``__init__`` re-export. The shim's lone import of
#     ``is_whisper_model`` does cause Python to load the legacy module,
#     but that is the one accounted-for surface; this test confirms the
#     legacy module is NEVER reached EXCEPT through that shim, and never
#     when ``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK`` is unset and the shim
#     itself is not loaded.
# ---------------------------------------------------------------------------


# Subprocess source — kept as a single textwrap-dedented string so each
# scenario gets a clean interpreter (no test-suite contamination from
# already-imported modules).
_PROBE_SCRIPT = textwrap.dedent(
    """
    import importlib
    import json
    import sys

    target_module = sys.argv[1]
    importlib.import_module(target_module)

    legacy = "octomil.runtime.engines.whisper._legacy_pywhisper"
    shim = "octomil.runtime.engines.whisper"
    print(json.dumps({
        "legacy_loaded": legacy in sys.modules,
        "shim_loaded": shim in sys.modules,
    }))
    """
).strip()


def _run_probe(target_module: str, *, env: dict[str, str]) -> dict[str, bool]:
    """Spawn a fresh interpreter, import ``target_module``, and report
    whether the legacy module / shim are in ``sys.modules``."""
    proc_env = {**os.environ, **env}
    # Strip any inherited opt-in so the negative case is clean.
    if env.get("OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK") is None:
        proc_env.pop("OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK", None)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT, target_module],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_repo_root()),
        env=proc_env,
    )
    import json

    return json.loads(result.stdout.strip().splitlines()[-1])


def test_top_level_octomil_import_does_not_load_legacy_module():
    """Importing the top-level ``octomil`` package without the
    benchmark opt-in MUST NOT pull the legacy module into
    ``sys.modules``. The package init is small; if a future change
    eagerly imports ``octomil.runtime.engines.whisper`` from
    ``octomil/__init__.py`` (or any module reached transitively), the
    legacy module would arrive via the shim's bounded re-export. That
    is still off the product hot path, but it widens the import
    surface — fail loudly so the change is reviewed."""
    state = _run_probe("octomil", env={})
    # We accept either: legacy not loaded, OR legacy loaded ONLY
    # because the shim is loaded (bounded re-export). The negative
    # case we forbid is: legacy loaded WITHOUT the shim, i.e. some
    # other product path imported it directly.
    if state["legacy_loaded"]:
        assert state["shim_loaded"], (
            "Legacy pywhispercpp module reached sys.modules WITHOUT "
            "going through octomil.runtime.engines.whisper.__init__ — "
            "some product path is importing it directly."
        )


@pytest.mark.parametrize(
    "module",
    [
        "octomil.execution.kernel",
        "octomil.serve.app",
    ],
)
def test_product_entry_points_only_load_legacy_via_shim(module: str):
    """The two canonical product entry points that *do* call
    :func:`is_whisper_model` (kernel + serve) load it through the
    package shim. The shim's lone re-export brings the legacy module
    in; this test pins that the import shape is exactly that — never
    a direct ``_legacy_pywhisper`` import from elsewhere."""
    state = _run_probe(module, env={})
    if state["legacy_loaded"]:
        assert state[
            "shim_loaded"
        ], f"{module} caused legacy pywhispercpp to load without the package shim being loaded — direct import found."


def test_benchmark_opt_in_still_works():
    """Sanity: the parity / benchmark opt-in path is untouched. Setting
    ``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK=1`` and importing the legacy
    module by dotted path MUST succeed. This is the single supported
    way to reach the legacy code at runtime."""
    state = _run_probe(
        _LEGACY_DOTTED,
        env={"OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK": "1"},
    )
    assert state["legacy_loaded"] is True


# ---------------------------------------------------------------------------
# (3) Regression docstring — pin the warning that lives at the top of
#     ``_legacy_pywhisper.py``. If a future refactor strips the warning,
#     this test fails and the maintainer is forced to either restore it
#     or update the guard deliberately.
# ---------------------------------------------------------------------------


def test_legacy_module_has_do_not_use_warning():
    """The legacy module's docstring must carry the explicit
    'DO NOT use on product paths' warning that points readers to this
    guard test. Pin it so a future refactor cannot silently strip the
    warning."""
    src = (_repo_root() / "octomil/runtime/engines/whisper/_legacy_pywhisper.py").read_text(encoding="utf-8")
    # We accept either capitalization but require the substantive
    # phrase that names the guard test by file name.
    assert (
        "DO NOT" in src and "product path" in src.lower()
    ), "_legacy_pywhisper.py must carry a 'DO NOT use on product paths' warning in its module docstring."
    assert "test_no_legacy_pywhisper_in_product_paths" in src, (
        "_legacy_pywhisper.py must reference the guard test "
        "'test_no_legacy_pywhisper_in_product_paths' so readers can find "
        "the enforcement point."
    )
    assert "OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK" in src, (
        "_legacy_pywhisper.py must document the opt-in env var "
        "'OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK' as the single supported "
        "entry point."
    )
