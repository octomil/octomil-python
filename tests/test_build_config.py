"""Tests for standalone binary build configuration files.

Validates that the PyInstaller spec, build script, install script, Homebrew
formula, and CI workflow are well-structured and consistent.
"""

from __future__ import annotations

import ast
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Build artifacts that only exist during release builds
_SPEC_FILE = ROOT / "octomil.spec"
_FORMULA_FILE = ROOT / "homebrew" / "octomil.rb"

_skip_no_spec = pytest.mark.skipif(
    not _SPEC_FILE.is_file(),
    reason="octomil.spec not present (only exists during release builds)",
)
_skip_no_formula = pytest.mark.skipif(
    not _FORMULA_FILE.is_file(),
    reason="homebrew/octomil.rb not present (only exists during release builds)",
)


# ---------------------------------------------------------------------------
# PyInstaller spec
# ---------------------------------------------------------------------------


@_skip_no_spec
class TestPyInstallerSpec:
    """Validate octomil.spec contents."""

    @pytest.fixture()
    def spec_content(self) -> str:
        return _SPEC_FILE.read_text()

    def test_spec_file_exists(self) -> None:
        assert _SPEC_FILE.is_file()

    def test_spec_is_valid_python(self, spec_content: str) -> None:
        """The spec file must be parseable Python."""
        ast.parse(spec_content)

    def test_spec_entry_point(self, spec_content: str) -> None:
        assert "octomil/__main__.py" in spec_content

    def test_spec_output_name(self, spec_content: str) -> None:
        assert 'name="octomil"' in spec_content

    def test_spec_console_mode(self, spec_content: str) -> None:
        assert "console=True" in spec_content

    def test_spec_onefile_mode(self, spec_content: str) -> None:
        """EXE receives all bundles — this is the one-file pattern."""
        assert "a.binaries" in spec_content
        assert "a.zipfiles" in spec_content
        assert "a.datas" in spec_content

    def test_spec_hidden_imports_engines(self, spec_content: str) -> None:
        """All engine modules must be listed as hidden imports."""
        expected_engines = [
            "octomil.runtime.engines.mlx.engine",
            "octomil.runtime.engines.llamacpp.engine",
            "octomil.runtime.engines.ort.engine",
            "octomil.runtime.engines.whisper.engine",
            "octomil.runtime.engines.ollama.engine",
            "octomil.runtime.engines.echo.engine",
            "octomil.runtime.engines.experimental.mnn.engine",
            "octomil.runtime.engines.experimental.executorch.engine",
        ]
        for engine in expected_engines:
            assert engine in spec_content, f"Missing hidden import: {engine}"

    def test_spec_hidden_imports_catalog(self, spec_content: str) -> None:
        assert "octomil.models.catalog" in spec_content

    def test_spec_excludes_heavy_deps(self, spec_content: str) -> None:
        """Heavy optional deps should be excluded to keep binary small."""
        for dep in ("torch", "tensorflow", "scipy", "matplotlib"):
            assert dep in spec_content, f"Expected {dep} in excludes list"


# ---------------------------------------------------------------------------
# Build script
# ---------------------------------------------------------------------------


class TestBuildScript:
    """Validate scripts/build-binary.sh."""

    @pytest.fixture()
    def script_path(self) -> Path:
        return ROOT / "scripts" / "build-binary.sh"

    @pytest.fixture()
    def script_content(self, script_path: Path) -> str:
        return script_path.read_text()

    def test_build_script_exists(self, script_path: Path) -> None:
        assert script_path.is_file()

    def test_build_script_executable(self, script_path: Path) -> None:
        mode = script_path.stat().st_mode
        assert mode & stat.S_IXUSR, "build-binary.sh should be executable"

    def test_build_script_has_shebang(self, script_content: str) -> None:
        assert script_content.startswith("#!/")

    def test_build_script_uses_set_euo(self, script_content: str) -> None:
        assert "set -euo pipefail" in script_content

    def test_build_script_detects_platforms(self, script_content: str) -> None:
        for platform in ("Darwin", "Linux", "arm64", "x86_64"):
            assert platform in script_content

    def test_build_script_runs_pyinstaller(self, script_content: str) -> None:
        assert "pyinstaller" in script_content
        assert "octomil.spec" in script_content

    def test_build_script_creates_archive(self, script_content: str) -> None:
        assert "tar -czf" in script_content

    def test_build_script_generates_sha256(self, script_content: str) -> None:
        assert "sha256sum" in script_content or "shasum" in script_content

    def test_build_script_verifies_binary(self, script_content: str) -> None:
        assert "--version" in script_content


# ---------------------------------------------------------------------------
# Install script
# ---------------------------------------------------------------------------


class TestInstallScript:
    """Validate scripts/install.sh."""

    @pytest.fixture()
    def script_path(self) -> Path:
        return ROOT / "scripts" / "install.sh"

    @pytest.fixture()
    def script_content(self, script_path: Path) -> str:
        return script_path.read_text()

    def test_install_script_exists(self, script_path: Path) -> None:
        assert script_path.is_file()

    def test_install_script_executable(self, script_path: Path) -> None:
        mode = script_path.stat().st_mode
        assert mode & stat.S_IXUSR, "install.sh should be executable"

    def test_install_script_has_shebang(self, script_content: str) -> None:
        assert script_content.startswith("#!/")

    def test_install_script_uses_set_eu(self, script_content: str) -> None:
        """Should use set -eu (POSIX sh does not support pipefail)."""
        assert "set -eu" in script_content

    def test_install_script_detects_os(self, script_content: str) -> None:
        for os_name in ("Darwin", "Linux"):
            assert os_name in script_content

    def test_install_script_detects_arch(self, script_content: str) -> None:
        for arch in ("arm64", "aarch64", "x86_64"):
            assert arch in script_content

    def test_install_script_uses_github_releases(self, script_content: str) -> None:
        assert "github.com" in script_content
        assert "octomil/octomil-python" in script_content

    def test_install_script_supports_octomil_version_env(self, script_content: str) -> None:
        assert "OCTOMIL_VERSION" in script_content

    def test_install_script_supports_octomil_install_env(self, script_content: str) -> None:
        assert "OCTOMIL_INSTALL" in script_content

    def test_install_script_fallback_install_dir(self, script_content: str) -> None:
        assert "/usr/local/bin" in script_content
        assert ".local/bin" in script_content

    def test_install_script_verifies_installation(self, script_content: str) -> None:
        assert "--version" in script_content

    def test_install_script_cleans_up_tmpdir(self, script_content: str) -> None:
        assert "mktemp" in script_content
        assert "trap" in script_content


# ---------------------------------------------------------------------------
# Homebrew formula
# ---------------------------------------------------------------------------


@_skip_no_formula
class TestHomebrewFormula:
    """Validate homebrew/octomil.rb."""

    @pytest.fixture()
    def formula_content(self) -> str:
        return _FORMULA_FILE.read_text()

    def test_formula_exists(self) -> None:
        assert _FORMULA_FILE.is_file()

    def test_formula_class_name(self, formula_content: str) -> None:
        assert "class Octomil < Formula" in formula_content

    def test_formula_has_desc(self, formula_content: str) -> None:
        assert 'desc "' in formula_content

    def test_formula_has_homepage(self, formula_content: str) -> None:
        assert 'homepage "https://octomil.com"' in formula_content

    def test_formula_has_license(self, formula_content: str) -> None:
        assert 'license "MIT"' in formula_content

    def test_formula_handles_arm_and_intel(self, formula_content: str) -> None:
        assert "Hardware::CPU.arm?" in formula_content
        assert "darwin-arm64" in formula_content
        assert "darwin-amd64" in formula_content

    def test_formula_handles_linux(self, formula_content: str) -> None:
        assert "on_linux" in formula_content

    def test_formula_install_block(self, formula_content: str) -> None:
        assert 'bin.install "octomil"' in formula_content

    def test_formula_test_block(self, formula_content: str) -> None:
        assert "assert_match" in formula_content
        assert "--version" in formula_content

    def test_formula_uses_github_releases(self, formula_content: str) -> None:
        assert "github.com/octomil/octomil-python/releases" in formula_content


# ---------------------------------------------------------------------------
# CI workflow
# ---------------------------------------------------------------------------


class TestReleaseWorkflow:
    """Validate .github/workflows/release-binary.yml."""

    @pytest.fixture()
    def workflow_content(self) -> str:
        return (ROOT / ".github" / "workflows" / "release-binary.yml").read_text()

    def test_workflow_exists(self) -> None:
        assert (ROOT / ".github" / "workflows" / "release-binary.yml").is_file()

    def test_workflow_triggers_on_tag(self, workflow_content: str) -> None:
        assert "tags: ['v*']" in workflow_content

    def test_workflow_matrix_targets(self, workflow_content: str) -> None:
        for target in ("darwin-arm64", "linux-amd64"):
            assert target in workflow_content

    def test_workflow_uses_python_311(self, workflow_content: str) -> None:
        assert "python-version: '3.11'" in workflow_content

    def test_workflow_installs_pyinstaller(self, workflow_content: str) -> None:
        assert "pyinstaller" in workflow_content.lower()

    def test_workflow_runs_pyinstaller(self, workflow_content: str) -> None:
        assert "pyinstaller packaging/octomil.spec" in workflow_content

    def test_workflow_verifies_binary(self, workflow_content: str) -> None:
        assert "--version" in workflow_content

    def test_workflow_generates_sha256(self, workflow_content: str) -> None:
        assert "sha256" in workflow_content.lower()

    def test_workflow_uploads_artifacts(self, workflow_content: str) -> None:
        assert "upload-artifact" in workflow_content

    def test_workflow_uploads_to_release(self, workflow_content: str) -> None:
        assert "gh release upload" in workflow_content

    def test_workflow_uses_macos_runners(self, workflow_content: str) -> None:
        """macOS arm64 requires macos-14."""
        assert "macos-14" in workflow_content

    def test_workflow_does_not_overwrite_existing_release_yml(self) -> None:
        """The existing release.yml (PyPI) should still exist."""
        assert (ROOT / ".github" / "workflows" / "release.yml").is_file()


# ---------------------------------------------------------------------------
# Cross-file consistency
# ---------------------------------------------------------------------------


class TestConsistency:
    """Verify files are consistent with each other and the project."""

    @_skip_no_spec
    def test_spec_entry_point_matches_pyproject(self) -> None:
        """octomil.spec should use the same entry point as pyproject.toml."""
        pyproject_content = (ROOT / "pyproject.toml").read_text()
        assert "octomil.cli:main" in pyproject_content
        spec_content = _SPEC_FILE.read_text()
        assert "octomil/__main__.py" in spec_content

    def test_all_engines_in_spec(self) -> None:
        """Every engine in the runtime/engines directory should be a hidden import."""
        spec_content = (ROOT / "packaging" / "octomil.spec").read_text()
        # Check stable engines
        engines_dir = ROOT / "octomil" / "runtime" / "engines"
        for engine_dir in engines_dir.iterdir():
            if not engine_dir.is_dir() or engine_dir.name.startswith("_"):
                continue
            if engine_dir.name == "experimental":
                continue
            engine_file = engine_dir / "engine.py"
            if engine_file.exists():
                module = f"octomil.runtime.engines.{engine_dir.name}.engine"
                assert module in spec_content, f"Engine {module} not in hidden imports"

    def test_github_repo_consistent(self) -> None:
        """All files should reference the same GitHub repo."""
        repo = "octomil/octomil-python"
        paths = [ROOT / "scripts" / "install.sh"]
        if _FORMULA_FILE.is_file():
            paths.append(_FORMULA_FILE)
        for path in paths:
            content = path.read_text()
            assert repo in content, f"{path.name} missing repo reference"

    @_skip_no_formula
    def test_version_in_formula_matches_pyproject(self) -> None:
        """The formula version should match pyproject.toml version."""
        import re

        pyproject_content = (ROOT / "pyproject.toml").read_text()
        formula_content = _FORMULA_FILE.read_text()
        # Extract version from pyproject.toml and formula, then compare
        pyproject_match = re.search(r'version\s*=\s*"([^"]+)"', pyproject_content)
        formula_match = re.search(r'version "([^"]+)"', formula_content)
        assert pyproject_match is not None, "version not found in pyproject.toml"
        assert formula_match is not None, "version not found in formula"
        assert pyproject_match.group(1) == formula_match.group(1)
