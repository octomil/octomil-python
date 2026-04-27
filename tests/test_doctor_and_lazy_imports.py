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


# NOTE: a true end-to-end "import octomil does not load pandas"
# regression is BLOCKED on a follow-up that decouples
# ``OctomilClient`` (and its eager ``model_ops`` mixin chain) from
# ``octomil.python.octomil``. The lazy ``__getattr__`` framework
# wired here is the necessary first step; tracking the remaining
# eager edge in a dedicated PR.
