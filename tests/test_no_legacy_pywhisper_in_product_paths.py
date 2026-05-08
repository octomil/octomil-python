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

(1) Source-pin (AST): no module under ``octomil/`` may write any of
    these import shapes EXCEPT the legacy file itself
    (``_legacy_pywhisper.py`` is allowed to self-reference):

    * Absolute import:
      ``import octomil.runtime.engines.whisper._legacy_pywhisper``
    * Absolute from-import:
      ``from octomil.runtime.engines.whisper._legacy_pywhisper import …``
    * Sibling form:
      ``from octomil.runtime.engines.whisper import _legacy_pywhisper``
    * Relative form (any depth):
      ``from . import _legacy_pywhisper`` or
      ``from ._legacy_pywhisper import …`` (or ``..``, ``...``, etc.)
    * Star form:
      ``from …._legacy_pywhisper import *``
    * Dynamic form:
      ``importlib.import_module("…._legacy_pywhisper")`` /
      ``__import__("…._legacy_pywhisper")`` /
      ``__import__("…<whisper-package>", fromlist=["_legacy_pywhisper"])``
      (Python loads each fromlist entry as a submodule of the first
      arg, so this shape DOES pull the legacy module in even though
      the call returns the parent package.)

    v0.1.6 PR2 deliberately moved :func:`is_whisper_model` (and
    ``_WHISPER_MODELS``) into the non-legacy module
    :mod:`octomil.runtime.engines.whisper.model_names` precisely so the
    package shim no longer has to import the legacy module at all.
    The package ``__init__`` is therefore NOT whitelisted — it is
    expected to import only from ``model_names``, and the runtime
    probe asserts that.

(2) Source-pin (AST): no module under ``octomil/`` may import the
    legacy class symbols (``WhisperCppEngine``, ``_WhisperBackend``)
    by name from anywhere — neither absolute nor relative nor star.
    The legacy file itself is the only allowed site.

(3) Shim discipline: the package ``__init__`` MUST re-export
    :func:`is_whisper_model` from
    :mod:`octomil.runtime.engines.whisper.model_names` (the
    non-legacy source of truth) and MUST NOT import anything from
    ``_legacy_pywhisper``. Pinned via AST.

(4) Runtime check (strict): with ``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK``
    unset, importing ANY of the canonical product entry points
    (``octomil``, ``octomil.serve.app``, ``octomil.execution.kernel``,
    ``octomil.runtime.engines.whisper``) MUST NOT cause
    ``octomil.runtime.engines.whisper._legacy_pywhisper`` to enter
    ``sys.modules``. After the v0.1.6 PR2 refactor, ``model_names``
    is the only module the shim touches, so the legacy module is
    fully off the product import graph.

(5) Sanity: with ``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK=1`` set and the
    legacy dotted path imported directly, the legacy module DOES
    load — the parity / benchmark opt-in is untouched.

Whitelisted (these MAY reference the legacy module):

* ``tests/`` — benchmark / reference tests live here.
* ``scripts/parity_*.py`` — parity-gate scripts; explicitly opt in
  via ``OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK=1``.
* ``packaging/`` — PyInstaller spec lists the legacy module as a
  hidden import so opt-in benchmark builds keep working.
* ``octomil/runtime/engines/whisper/_legacy_pywhisper.py`` itself.

Hard-cutover discipline: if a future refactor needs the legacy
module on a product path, it must remove the legacy module entirely
(or build a real native replacement) — not widen this guard.
"""

from __future__ import annotations

import ast
import json
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
_WHISPER_PACKAGE = "octomil.runtime.engines.whisper"
_MODEL_NAMES_DOTTED = "octomil.runtime.engines.whisper.model_names"
_MODEL_NAMES_TAIL = "model_names"

# Symbol names that, if imported on a product path, would defeat the
# cutover. ``is_whisper_model`` is intentionally NOT in this list — it now
# lives in the non-legacy ``model_names`` module. Engine / inference
# symbols are.
_LEGACY_INFERENCE_SYMBOLS = frozenset(
    {
        "WhisperCppEngine",
        "_WhisperBackend",
    }
)

# Files where ``_legacy_pywhisper`` references are allowed to appear in
# import statements. v0.1.6 PR2: only the legacy file itself. The package
# shim no longer imports from the legacy module — it imports from the
# non-legacy ``model_names`` module instead.
_PRODUCT_IMPORT_WHITELIST = frozenset(
    {
        "octomil/runtime/engines/whisper/_legacy_pywhisper.py",
    }
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
# AST helpers.
# ---------------------------------------------------------------------------


def _parse(path: Path) -> ast.AST | None:
    """Best-effort parse. Returns ``None`` on read / parse error so a
    syntax error in product code is a separate (loud) bug — not this
    guard's responsibility."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        return None


def _module_dotted(path: Path) -> str:
    """Return the dotted module path for ``path`` under the ``octomil``
    package — used to resolve relative imports."""
    rel = path.resolve().relative_to(_repo_root()).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(module: str, level: int, anchor: str) -> str:
    """Resolve a relative ``ImportFrom`` (``from .x.y import …``) into
    its absolute dotted path, given the importing module's dotted path
    (``anchor``). Returns ``""`` if the level overshoots the anchor."""
    if level <= 0:
        return module or ""
    anchor_parts = anchor.split(".")
    # `from . import x` inside a package module ``a.b.c`` resolves to
    # ``a.b`` (drop ``level`` trailing parts). For an ``__init__`` the
    # anchor is already the package, so level=1 means "this package".
    if level > len(anchor_parts):
        return ""
    base = anchor_parts[: len(anchor_parts) - (level - 1)]
    if module:
        base.append(module)
    return ".".join(p for p in base if p)


# ---------------------------------------------------------------------------
# (1) AST guard — every shape that could re-introduce ``_legacy_pywhisper``
#     into a product path's import graph.
# ---------------------------------------------------------------------------


def _legacy_imports_in_file(path: Path) -> list[tuple[int, str]]:
    """Return a list of ``(lineno, snippet)`` for every statement in
    ``path`` that imports the legacy module — covering absolute,
    relative, sibling, star, and dynamic forms."""
    tree = _parse(path)
    if tree is None:
        return []

    anchor = _module_dotted(path)
    hits: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # ``import a.b.c._legacy_pywhisper`` (and aliased form).
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _LEGACY_DOTTED or alias.name.endswith("." + _LEGACY_TAIL):
                    hits.append((node.lineno, f"import {alias.name}"))
            continue

        # ``from … import …`` — handles absolute, relative, sibling, star.
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            absolute = _resolve_relative(module, level, anchor) if level else module

            # Form A: from <legacy dotted> import …
            if absolute == _LEGACY_DOTTED or absolute.endswith("." + _LEGACY_TAIL):
                imported = ", ".join(a.name for a in node.names)
                src_repr = ("." * level) + module if level else module
                hits.append((node.lineno, f"from {src_repr} import {imported}"))
                continue

            # Form B: from <something> import _legacy_pywhisper
            for alias in node.names:
                if alias.name == _LEGACY_TAIL:
                    src_repr = ("." * level) + module if level else module
                    hits.append((node.lineno, f"from {src_repr} import {alias.name}"))
            continue

        # Form C: dynamic imports of the legacy module — string-literal
        # arguments only. Three sub-shapes are detected:
        #
        #   importlib.import_module("…._legacy_pywhisper")
        #   __import__("…._legacy_pywhisper")
        #   __import__("…<whisper-package>", fromlist=["_legacy_pywhisper"])
        #     (or any literal-list `fromlist` containing the tail name —
        #      Python actually loads the submodule in this shape, even
        #      though the call returns the parent package.)
        #
        # Dynamic-runtime forms with non-literal arguments are out of
        # scope (string analysis would always lose to a determined
        # adversary; the strict runtime probe catches those).
        if isinstance(node, ast.Call):
            for hit in _legacy_dynamic_import_hits(node):
                hits.append((node.lineno, hit))

    return hits


def _legacy_dynamic_import_hits(node: ast.Call) -> list[str]:
    """Detect string-literal dynamic imports that load the legacy module.

    Returns one snippet per legacy reference in this call. Empty list
    if the call is unrelated (or arguments are non-literal).
    """
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "import_module":
        call_label = "importlib.import_module"
    elif isinstance(func, ast.Name) and func.id == "__import__":
        call_label = "__import__"
    else:
        return []

    out: list[str] = []

    # Sub-shape (a): first positional arg is the legacy dotted path or
    # ends with the legacy tail.
    if node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            value = first.value
            if value == _LEGACY_DOTTED or value.endswith("." + _LEGACY_TAIL):
                out.append(f'{call_label}("{value}")')

    # Sub-shape (b): __import__("…whisper", fromlist=["_legacy_pywhisper"]).
    # Python loads each name in fromlist as a submodule of the first
    # arg, so a literal "_legacy_pywhisper" entry pulls the legacy
    # module in even though the call returns the package. Accept any
    # literal-list / literal-tuple in either positional 4th arg or
    # ``fromlist`` keyword.
    fromlist_node: ast.AST | None = None
    if call_label == "__import__":
        if len(node.args) >= 4:
            fromlist_node = node.args[3]
        for kw in node.keywords:
            if kw.arg == "fromlist":
                fromlist_node = kw.value
    elif call_label == "importlib.import_module":
        # importlib.import_module has no fromlist. The first arg is
        # already covered.
        pass

    if isinstance(fromlist_node, (ast.List, ast.Tuple)):
        first_arg_repr = ""
        if node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                first_arg_repr = first.value
        for elt in fromlist_node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str) and elt.value == _LEGACY_TAIL:
                out.append(f'{call_label}("{first_arg_repr}", fromlist=[…, "{_LEGACY_TAIL}", …])')

    return out


def test_no_legacy_pywhisper_import_on_product_paths():
    """No product module may import the legacy whisper shim by ANY
    static form — absolute, relative, sibling, star, or dynamic
    string-literal call.

    Whitelist is enforced at file granularity. v0.1.6 PR2 narrowed it
    to just ``_legacy_pywhisper.py`` itself (self-references); the
    package ``__init__`` no longer imports from the legacy module at
    all.
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


def test_no_legacy_inference_symbols_on_product_paths():
    """No product module may import ``WhisperCppEngine`` or
    ``_WhisperBackend`` by name from anywhere (absolute, relative,
    star). These symbols only live in the legacy file, and the
    previous test already forbids importing that file from product
    paths — this is a defense-in-depth pin against the symbol names
    themselves."""
    offenders: list[str] = []
    for path in _iter_product_python_files():
        rel = _relpath(path)
        if rel in _PRODUCT_IMPORT_WHITELIST:
            continue
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                src_repr = "." * (node.level or 0) + (node.module or "")
                for alias in node.names:
                    if alias.name in _LEGACY_INFERENCE_SYMBOLS:
                        offenders.append(f"{rel}:{node.lineno}  from {src_repr} import {alias.name}")
    assert not offenders, "Legacy whisper inference symbols imported on product paths:\n  " + "\n  ".join(offenders)


def test_whisper_package_shim_imports_from_model_names_not_legacy():
    """The package shim (``__init__``) MUST source
    :func:`is_whisper_model` from the non-legacy ``model_names``
    module and MUST NOT import anything from ``_legacy_pywhisper``.
    This is the structural change v0.1.6 PR2 codifies — the shim is
    the single product-path entry to whisper-name detection, and it
    must not drag the legacy module along."""
    shim = _repo_root() / "octomil/runtime/engines/whisper/__init__.py"
    tree = ast.parse(shim.read_text(encoding="utf-8"), filename=str(shim))

    legacy_imports: list[str] = []
    model_names_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            level = node.level or 0
            module = node.module or ""
            absolute = _resolve_relative(module, level, _WHISPER_PACKAGE) if level else module
            if absolute == _LEGACY_DOTTED or absolute.endswith("." + _LEGACY_TAIL):
                legacy_imports.extend(a.name for a in node.names)
            if absolute == _MODEL_NAMES_DOTTED or absolute.endswith("." + _MODEL_NAMES_TAIL):
                model_names_imports.extend(a.name for a in node.names)
            for alias in node.names:
                if alias.name == _LEGACY_TAIL:
                    legacy_imports.append(alias.name)

    assert not legacy_imports, (
        "octomil/runtime/engines/whisper/__init__.py must NOT import "
        "from _legacy_pywhisper. v0.1.6 PR2 moved is_whisper_model "
        "into the non-legacy 'model_names' module; the shim should "
        f"import from there. Found legacy imports: {legacy_imports!r}."
    )
    assert "is_whisper_model" in model_names_imports, (
        "octomil/runtime/engines/whisper/__init__.py must re-export "
        "is_whisper_model from the non-legacy 'model_names' module."
    )


# ---------------------------------------------------------------------------
# (2) Runtime guard — strict. After v0.1.6 PR2 the legacy module is fully
#     off the product import graph; importing any product entry point
#     without the benchmark opt-in MUST leave it absent from sys.modules.
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
    model_names = "octomil.runtime.engines.whisper.model_names"
    print(json.dumps({
        "legacy_loaded": legacy in sys.modules,
        "shim_loaded": shim in sys.modules,
        "model_names_loaded": model_names in sys.modules,
    }))
    """
).strip()


def _run_probe(target_module: str, *, env: dict[str, str]) -> dict[str, bool]:
    """Spawn a fresh interpreter, import ``target_module``, and report
    whether each candidate module is in ``sys.modules``."""
    proc_env = {**os.environ, **env}
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
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "module",
    [
        "octomil",
        "octomil.execution.kernel",
        "octomil.serve.app",
        "octomil.runtime.engines.whisper",
    ],
)
def test_product_entry_points_do_not_load_legacy_module(module: str):
    """Strict: importing any canonical product entry point with the
    benchmark opt-in unset MUST NOT cause ``_legacy_pywhisper`` to
    enter ``sys.modules``. v0.1.6 PR2 made this assertion strict — the
    legacy module is fully off the product import graph."""
    state = _run_probe(module, env={})
    assert state["legacy_loaded"] is False, (
        f"Importing {module!r} loaded the legacy pywhispercpp module — "
        "the v0.1.6 PR2 cutover discipline says it must stay off the "
        "product import graph until OCTOMIL_USE_PY_WHISPERCPP_BENCHMARK=1."
    )


def test_whisper_package_loads_model_names_not_legacy():
    """When the whisper package is imported, ``model_names`` MUST be
    loaded (it is the non-legacy source of :func:`is_whisper_model`)
    and the legacy module MUST NOT be."""
    state = _run_probe(_WHISPER_PACKAGE, env={})
    assert (
        state["model_names_loaded"] is True
    ), "Importing the whisper package did not load 'model_names' — the shim's re-export of is_whisper_model is broken."
    assert state["legacy_loaded"] is False, "Importing the whisper package loaded the legacy pywhispercpp module."


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


# ---------------------------------------------------------------------------
# (4) Self-test — assert the AST guard catches the relative + dynamic
#     forms it claims to catch. We synthesize tiny snippets and feed them
#     to the parser directly (no temp files) so the guard's own logic is
#     verified against forbidden shapes without polluting the source tree.
# ---------------------------------------------------------------------------


def _hits_for_snippet(snippet: str, *, anchor_path: str = "octomil/foo/bar.py") -> list[str]:
    """Run the AST guard's matcher against an in-memory snippet,
    pretending it lives at ``anchor_path`` for relative-import
    resolution. Returns the list of detected snippets."""
    tree = ast.parse(snippet)
    anchor_dotted = _module_dotted(_repo_root() / anchor_path)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _LEGACY_DOTTED or alias.name.endswith("." + _LEGACY_TAIL):
                    hits.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            absolute = _resolve_relative(module, level, anchor_dotted) if level else module
            if absolute == _LEGACY_DOTTED or absolute.endswith("." + _LEGACY_TAIL):
                imported = ", ".join(a.name for a in node.names)
                src_repr = ("." * level) + module if level else module
                hits.append(f"from {src_repr} import {imported}")
                continue
            for alias in node.names:
                if alias.name == _LEGACY_TAIL:
                    src_repr = ("." * level) + module if level else module
                    hits.append(f"from {src_repr} import {alias.name}")
        elif isinstance(node, ast.Call):
            # Reuse the production matcher so the self-test pins the
            # same code path it claims to verify.
            hits.extend(_legacy_dynamic_import_hits(node))
    return hits


@pytest.mark.parametrize(
    "snippet,anchor_path",
    [
        # Absolute dotted-path forms.
        ("import octomil.runtime.engines.whisper._legacy_pywhisper", "octomil/foo/bar.py"),
        (
            "import octomil.runtime.engines.whisper._legacy_pywhisper as legacy",
            "octomil/foo/bar.py",
        ),
        (
            "from octomil.runtime.engines.whisper._legacy_pywhisper import is_whisper_model",
            "octomil/foo/bar.py",
        ),
        (
            "from octomil.runtime.engines.whisper._legacy_pywhisper import *",
            "octomil/foo/bar.py",
        ),
        # Sibling absolute form.
        (
            "from octomil.runtime.engines.whisper import _legacy_pywhisper",
            "octomil/foo/bar.py",
        ),
        # Relative forms — anchored inside the whisper package.
        (
            "from . import _legacy_pywhisper",
            "octomil/runtime/engines/whisper/__init__.py",
        ),
        (
            "from ._legacy_pywhisper import is_whisper_model",
            "octomil/runtime/engines/whisper/__init__.py",
        ),
        (
            "from ._legacy_pywhisper import *",
            "octomil/runtime/engines/whisper/__init__.py",
        ),
        # Relative form one level deeper.
        (
            "from .._legacy_pywhisper import is_whisper_model",
            "octomil/runtime/engines/whisper/sub/sub.py",
        ),
        # Dynamic forms with literal string.
        (
            'import importlib\nimportlib.import_module("octomil.runtime.engines.whisper._legacy_pywhisper")',
            "octomil/foo/bar.py",
        ),
        (
            '__import__("octomil.runtime.engines.whisper._legacy_pywhisper")',
            "octomil/foo/bar.py",
        ),
        # __import__ fromlist shape — Python loads the listed submodule
        # of the first arg, so this DOES bring _legacy_pywhisper in even
        # though the call returns the parent package.
        (
            '__import__("octomil.runtime.engines.whisper", fromlist=["_legacy_pywhisper"])',
            "octomil/foo/bar.py",
        ),
        # Same shape, fromlist as positional 4th arg, tuple form.
        (
            '__import__("octomil.runtime.engines.whisper", None, None, ("_legacy_pywhisper",))',
            "octomil/foo/bar.py",
        ),
    ],
)
def test_ast_guard_self_test_catches_forbidden_shape(snippet: str, anchor_path: str):
    """Confirm the AST guard's matcher detects every forbidden shape
    enumerated in the module docstring. Without these self-tests, a
    future refactor that breaks the matcher would silently pass."""
    hits = _hits_for_snippet(snippet, anchor_path=anchor_path)
    assert hits, f"AST guard did not catch forbidden snippet: {snippet!r}"


def test_ast_guard_self_test_ignores_unrelated_imports():
    """Negative control: imports that share no part of the legacy
    dotted path / tail name MUST NOT register as hits. Without this
    pin, an over-broad matcher (e.g. substring match) could
    spuriously flag innocent code."""
    benign_snippets = [
        "import octomil.runtime.engines.whisper.model_names",
        "from octomil.runtime.engines.whisper import is_whisper_model",
        "from octomil.runtime.engines.whisper.model_names import is_whisper_model",
        "from . import model_names",
        'importlib.import_module("octomil.runtime.engines.whisper")',
    ]
    for snippet in benign_snippets:
        # Wrap the importlib snippet with the import statement so it parses.
        full = ("import importlib\n" + snippet) if "importlib." in snippet else snippet
        hits = _hits_for_snippet(full, anchor_path="octomil/runtime/engines/whisper/__init__.py")
        assert not hits, f"AST guard spuriously matched benign snippet: {snippet!r} -> {hits!r}"
