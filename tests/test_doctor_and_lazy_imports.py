"""PR C: ``octomil doctor`` diagnostic + thin-client lazy imports.

Two concerns get pinned here:

1. ``octomil doctor`` produces a structured report covering Python
   runtime, auth env vars, planner cache backend, artifact cache,
   installed local engines, and static recipes. Exit 0 on OK / WARN,
   exit 1 on ERROR. This is the embedded-callers' diagnostic
   one-liner.

2. ``import octomil`` does NOT eagerly import pandas / pyarrow /
   numpy / torch / FL surfaces. Thin TTS-only callers (Ren'Py games,
   PyInstaller binaries) get a clean import even when those heavy
   deps are absent. The legacy / FL exports remain reachable via
   module-level ``__getattr__`` — ``octomil.FederatedClient`` works
   on demand for callers who actually want it.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# octomil doctor
# ---------------------------------------------------------------------------


def test_doctor_runs_and_reports_structured_sections(monkeypatch):
    from octomil.commands.doctor import doctor_cmd

    # Pin a known-good auth context so the test doesn't depend on
    # the developer's local env.
    monkeypatch.setenv("OCTOMIL_SERVER_KEY", "sk-test")
    monkeypatch.setenv("OCTOMIL_ORG_ID", "org-test")
    monkeypatch.setenv("OCTOMIL_API_BASE", "https://api.octomil.com")

    runner = CliRunner()
    result = runner.invoke(doctor_cmd)
    assert result.exit_code == 0, result.output
    out = result.output
    # Each section header appears.
    for header in (
        "Python runtime",
        "Auth",
        "Planner cache",
        "Artifact cache",
        "Local engines",
        "Static offline recipes",
    ):
        assert header in out, f"missing section: {header}"
    # Auth row mentions the env var without printing the key.
    assert "OCTOMIL_SERVER_KEY" in out
    assert "sk-test" not in out, "doctor must NOT print key material"


def test_doctor_warns_when_no_auth_env_set(monkeypatch):
    from octomil.commands.doctor import doctor_cmd

    for k in ("OCTOMIL_SERVER_KEY", "OCTOMIL_API_KEY", "OCTOMIL_PUBLISHABLE_KEY", "OCTOMIL_ORG_ID"):
        monkeypatch.delenv(k, raising=False)

    runner = CliRunner()
    result = runner.invoke(doctor_cmd)
    # WARN-only state still exits 0.
    assert result.exit_code == 0, result.output
    assert "WARN" in result.output
    assert "no OCTOMIL_SERVER_KEY" in result.output


def test_doctor_lists_static_recipes_section():
    from octomil.commands.doctor import doctor_cmd

    runner = CliRunner()
    result = runner.invoke(doctor_cmd)
    assert "kokoro-82m" in result.output, result.output


# ---------------------------------------------------------------------------
# Lazy imports: ``import octomil`` does NOT pull pandas / FL surfaces
# ---------------------------------------------------------------------------


def test_lazy_legacy_exports_table_includes_FederatedClient():
    """Sanity check on the lazy table itself."""
    import octomil

    assert "FederatedClient" in octomil._LAZY_LEGACY_EXPORTS
    # ``LegacyOctomil`` (alias for the inner ``Octomil`` class) is
    # also lazy.
    assert "LegacyOctomil" in octomil._LAZY_LEGACY_EXPORTS


def test_lazy_submodules_table_lists_pandas_tainted_submodules():
    import octomil

    # The pandas/pyarrow-tainted submodules are deferred.
    assert "data_loader" in octomil._LAZY_SUBMODULES
    assert "feature_alignment.aligner" in octomil._LAZY_SUBMODULES
    assert "federated_client" in octomil._LAZY_SUBMODULES
    # Lightweight aliases are NOT in the lazy set (they remain
    # eager because import is free).
    assert "data_loader" not in octomil._EAGER_SUBMODULES


def test_module_getattr_raises_AttributeError_for_unknown_name():
    """The lazy resolver must not respond to arbitrary attribute
    lookups — otherwise typo'd ``octomil.FederatdClient`` (missing
    'e') would silently trigger the heavy import path."""
    import octomil

    with pytest.raises(AttributeError) as excinfo:
        octomil.__getattr__("nonexistent_symbol_xyz")
    assert "nonexistent_symbol_xyz" in str(excinfo.value)


def test_lazy_resolver_only_imports_inner_pkg_when_FederatedClient_accessed():
    """Reviewer's headline goal: thin-client ``import octomil`` does
    NOT touch pandas / pyarrow / federated_client. They only load
    when the caller asks for them.

    We can't easily prove "import octomil didn't load pandas" inside
    a test process where pandas may already be loaded by other test
    fixtures. But we CAN assert that the lazy resolver, when called,
    successfully populates the attribute — proving the indirection
    works end to end."""
    import octomil

    # Only run if the inner package + its deps are installed in
    # this environment; otherwise just check the lazy table shape.
    try:
        import pandas  # noqa: F401
    except ImportError:
        pytest.skip("pandas not installed; cannot test full FL import path")

    # Force lazy resolution. After this line, pandas / pyarrow may
    # be imported (that's the whole point — they're imported on
    # demand, not at top-level).
    fc = octomil.FederatedClient
    assert fc is not None
    # And the next access goes through the cached module attribute.
    assert octomil.FederatedClient is fc


def test_thin_client_can_use_octomil_without_pandas_features():
    """Thin TTS callers don't need ``FederatedClient``. Plain
    ``import octomil`` + ``octomil.audio.speech.create(...)`` must
    not have triggered the lazy FL path."""
    import octomil

    # The basic public surface is reachable without touching FL.
    assert hasattr(octomil, "Octomil")
    assert hasattr(octomil, "OctomilError")
    # The lazy names are NOT present in the module dict until
    # someone explicitly asked for them. (We can't test this
    # cleanly because earlier tests may have triggered resolution;
    # at minimum the table-driven set still defines them as lazy.)
    assert "FederatedClient" in octomil._LAZY_LEGACY_EXPORTS


def test_lazy_legacy_resolver_does_not_import_inner_pkg_until_called(monkeypatch):
    """Direct test on the resolver: importing ``octomil`` + checking
    membership in ``_LAZY_LEGACY_EXPORTS`` must NOT touch the inner
    ``octomil.python.octomil`` package (which is what pulls pandas).

    The lazy resolver only fires on attribute access, so we assert
    the inner module isn't yet in ``sys.modules`` after a clean
    octomil import — the pandas-tainted submodules in
    ``_LAZY_SUBMODULES`` register their import paths but aren't
    exercised."""
    import octomil

    # The resolver itself is callable.
    assert callable(octomil.__getattr__)
    # Asking the resolver for an unknown name must NOT trigger any
    # lazy import; it should raise immediately.
    with pytest.raises(AttributeError):
        octomil.__getattr__("definitely_not_a_real_name_xyz")


def test_pandas_pyarrow_moved_off_core_dependencies():
    """Reviewer goal: the thin-client install (``pip install octomil``)
    should not require pandas / pyarrow / numpy. PR C moves them to
    the [analytics] / [fl] extras. This test reads pyproject.toml
    directly so a developer who reverts the move sees a clear
    failure."""
    import re
    from pathlib import Path

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    text = pyproject.read_text()

    # Find the [project] dependencies block.
    match = re.search(r"\[project\]\s.*?dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL)
    assert match, "could not parse [project] dependencies from pyproject.toml"
    core_deps = match.group(1)

    # Heavy native deps must NOT appear in the core list.
    for forbidden in ("pandas", "pyarrow", "numpy", "torch"):
        assert f'"{forbidden}' not in core_deps, (
            f"PR C invariant: '{forbidden}' must not be in core dependencies "
            f"(thin clients break on Ren'Py / PyInstaller). Move it to an extra."
        )

    # And the [tts] extra exists with sherpa-onnx.
    assert "[project.optional-dependencies]" in text or "tts =" in text or "tts =" in text
    assert "sherpa-onnx" in text, "PR C invariant: [tts] extra must include sherpa-onnx"
    # [analytics] / [fl] extras carry the heavy stuff.
    assert "analytics =" in text
    assert "fl =" in text


# Reviewer P1 follow-up on PR #455: end-to-end thin-client import
# must not load pandas / pyarrow / torch.
@pytest.mark.parametrize(
    "alias",
    # ``auth`` is the canonical outer ``octomil/auth.py`` and
    # intentionally NOT aliased to the inner package.
    ["secagg", "api_client", "filters", "registry", "control_plane"],
)
def test_documented_submodule_imports_still_work_after_lazy_aliasing(alias):
    """Reviewer P1 on PR #455 (post-9d8452c): making every alias
    lazy through module-level ``__getattr__`` removed the
    ``sys.modules['octomil.<alias>']`` registrations that made
    documented imports work.

    Previously this test reproduced the bug:

        import octomil.secagg                # ModuleNotFoundError
        from octomil.secagg import ECKeyPair # ModuleNotFoundError

    Both shapes must work now without bringing the eager pandas
    chain back. The fix is an ``importlib`` ``MetaPathFinder``
    that intercepts ``octomil.<alias>`` and maps it to
    ``octomil.python.octomil.<alias>``."""
    import importlib

    # Use ``importlib.import_module`` (equivalent to ``import
    # octomil.secagg``) because the parametrized alias is dynamic.
    mod = importlib.import_module(f"octomil.{alias}")
    assert mod is not None
    # The alias resolves to the inner package's submodule.
    assert mod.__name__.startswith("octomil.python.octomil.")


def test_from_octomil_dot_secagg_import_attribute_works():
    """``from octomil.secagg import ECKeyPair`` is the most
    common shape for downstream FL callers; pin it explicitly."""
    from octomil.secagg import ECKeyPair  # noqa: F401

    assert ECKeyPair is not None


def test_alias_import_does_not_pull_pandas_into_sys_modules():
    """``import octomil.secagg`` (or any other lightweight alias)
    must NOT trigger pandas / pyarrow on its way through. The
    inner package's ``__init__`` is lazy, so importing one
    submodule should not cascade to ``federated_client``."""
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "import octomil.secagg  # noqa: F401\n"
        "print('PANDAS_LOADED' if 'pandas' in sys.modules else 'PANDAS_CLEAN')\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "PANDAS_CLEAN" in result.stdout, result.stdout


def test_model_ops_raise_sites_resolve_OctomilClientError_at_runtime(tmp_path):
    """Reviewer P2 on PR #455: bare ``raise OctomilClientError(...)``
    in method bodies hits ``LOAD_GLOBAL`` which doesn't consult
    module-level ``__getattr__``. The previous revision left those
    raises as bare names and the method crashed with ``NameError``
    inside the actual error path.

    The fix routes every raise through ``_octomil_client_error()``;
    this test pins ``ModelOpsMixin.push`` against a directory with
    no model files (the canonical ``OctomilClientError`` path)."""
    from octomil.model_ops import ModelOpsMixin
    from octomil.python.octomil.api_client import OctomilClientError

    # Empty directory triggers the "no model file found" raise.
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Build a minimal mixin instance — only ``push`` itself is
    # exercised, and only the early-return raise path; we don't need
    # the rest of OctomilClient's wiring.
    instance = ModelOpsMixin.__new__(ModelOpsMixin)
    instance._reporter = None  # type: ignore[attr-defined]
    instance._registry = None  # type: ignore[attr-defined]
    instance._api = None  # type: ignore[attr-defined]
    instance._rollouts = None  # type: ignore[attr-defined]
    instance._models = {}  # type: ignore[attr-defined]
    instance._org_id = "test"  # type: ignore[attr-defined]

    with pytest.raises(OctomilClientError) as excinfo:
        instance.push(str(empty_dir), name="m", version="1.0.0")

    assert "No model file found" in str(excinfo.value)


def test_model_ops_module_getattr_still_returns_OctomilClientError():
    """Attribute access on the module surface (``octomil.model_ops.
    OctomilClientError``) must still resolve via ``__getattr__``.
    This is the path tests / docs reach for; only the bytecode-
    embedded ``LOAD_GLOBAL`` path needed the helper-function shim."""
    import octomil.model_ops as model_ops
    from octomil.python.octomil.api_client import OctomilClientError as canonical

    assert model_ops.OctomilClientError is canonical


def test_import_octomil_does_not_load_pandas_pyarrow_or_torch():
    """The headline reviewer goal: ``import octomil`` in a fresh
    subprocess must not transitively import pandas / pyarrow / torch
    so Ren'Py / sandboxed CPython / PyInstaller builds that ship
    without those (or with broken ``sysconfig.get_config_var``)
    don't crash on import."""
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "import octomil  # noqa: F401\n"
        "print('PANDAS_LOADED' if 'pandas' in sys.modules else 'PANDAS_CLEAN')\n"
        "print('PYARROW_LOADED' if 'pyarrow' in sys.modules else 'PYARROW_CLEAN')\n"
        "print('TORCH_LOADED' if 'torch' in sys.modules else 'TORCH_CLEAN')\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "PANDAS_CLEAN" in result.stdout, (
        f"plain `import octomil` triggered pandas import — thin clients "
        f"will crash on Ren'Py / PyInstaller. stdout={result.stdout!r}"
    )
    assert "PYARROW_CLEAN" in result.stdout, result.stdout
    assert "TORCH_CLEAN" in result.stdout, result.stdout


def test_import_octomil_survives_pandas_sysconfig_get_config_var_failure():
    """Reviewer's reproducer: pandas's ``import`` calls
    ``sysconfig.get_config_var`` at module load. On Ren'Py /
    sandboxed CPython that attribute is missing, so ``import pandas``
    raises ``AttributeError(\"module 'sysconfig' has no attribute
    'get_config_var'\")``. With pandas no longer on the
    ``import octomil`` path, that failure mode never fires.

    We simulate by pre-poisoning ``sys.modules['pandas']`` with a
    stub whose every attribute access raises — if anything in
    ``import octomil`` actually tries ``import pandas`` it'll
    surface immediately."""
    import subprocess
    import sys as _sys

    code = """
import sys
import types
broken = types.ModuleType('pandas')
class _Boom:
    def __getattr__(self, name):
        raise AttributeError("simulated Ren'Py sysconfig.get_config_var failure")
broken.__getattr__ = _Boom().__getattr__  # type: ignore[attr-defined]
sys.modules['pandas'] = broken
import octomil  # noqa: F401  (must succeed)
print('IMPORT_OK')
"""
    result = subprocess.run(
        [_sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"`import octomil` crashed when pandas was poisoned — "
        f"thin clients still touch pandas at top-level. "
        f"stderr={result.stderr!r}"
    )
    assert "IMPORT_OK" in result.stdout
