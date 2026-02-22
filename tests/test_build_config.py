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


# ---------------------------------------------------------------------------
# PyInstaller spec
# ---------------------------------------------------------------------------


class TestPyInstallerSpec:
    """Validate edgeml.spec contents."""

    @pytest.fixture()
    def spec_content(self) -> str:
        return (ROOT / "edgeml.spec").read_text()

    def test_spec_file_exists(self) -> None:
        assert (ROOT / "edgeml.spec").is_file()

    def test_spec_is_valid_python(self, spec_content: str) -> None:
        """The spec file must be parseable Python."""
        ast.parse(spec_content)

    def test_spec_entry_point(self, spec_content: str) -> None:
        assert "edgeml/cli.py" in spec_content

    def test_spec_output_name(self, spec_content: str) -> None:
        assert 'name="edgeml"' in spec_content

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
            "edgeml.engines.mlx_engine",
            "edgeml.engines.llamacpp_engine",
            "edgeml.engines.mnn_engine",
            "edgeml.engines.executorch_engine",
            "edgeml.engines.ort_engine",
            "edgeml.engines.whisper_engine",
            "edgeml.engines.echo_engine",
        ]
        for engine in expected_engines:
            assert engine in spec_content, f"Missing hidden import: {engine}"

    def test_spec_hidden_imports_catalog(self, spec_content: str) -> None:
        assert "edgeml.models.catalog" in spec_content

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
        assert "edgeml.spec" in script_content

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
        assert "edgeml-ai/edgeml-python" in script_content

    def test_install_script_supports_edgeml_version_env(
        self, script_content: str
    ) -> None:
        assert "EDGEML_VERSION" in script_content

    def test_install_script_supports_edgeml_install_env(
        self, script_content: str
    ) -> None:
        assert "EDGEML_INSTALL" in script_content

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


class TestHomebrewFormula:
    """Validate homebrew/edgeml.rb."""

    @pytest.fixture()
    def formula_content(self) -> str:
        return (ROOT / "homebrew" / "edgeml.rb").read_text()

    def test_formula_exists(self) -> None:
        assert (ROOT / "homebrew" / "edgeml.rb").is_file()

    def test_formula_class_name(self, formula_content: str) -> None:
        assert "class Edgeml < Formula" in formula_content

    def test_formula_has_desc(self, formula_content: str) -> None:
        assert 'desc "' in formula_content

    def test_formula_has_homepage(self, formula_content: str) -> None:
        assert 'homepage "https://edgeml.io"' in formula_content

    def test_formula_has_license(self, formula_content: str) -> None:
        assert 'license "MIT"' in formula_content

    def test_formula_handles_arm_and_intel(self, formula_content: str) -> None:
        assert "Hardware::CPU.arm?" in formula_content
        assert "darwin-arm64" in formula_content
        assert "darwin-amd64" in formula_content

    def test_formula_handles_linux(self, formula_content: str) -> None:
        assert "on_linux" in formula_content

    def test_formula_install_block(self, formula_content: str) -> None:
        assert 'bin.install "edgeml"' in formula_content

    def test_formula_test_block(self, formula_content: str) -> None:
        assert "assert_match" in formula_content
        assert "--version" in formula_content

    def test_formula_uses_github_releases(self, formula_content: str) -> None:
        assert "github.com/edgeml-ai/edgeml-python/releases" in formula_content


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
        for target in ("darwin-arm64", "darwin-amd64", "linux-amd64"):
            assert target in workflow_content

    def test_workflow_uses_python_311(self, workflow_content: str) -> None:
        assert "python-version: '3.11'" in workflow_content

    def test_workflow_installs_pyinstaller(self, workflow_content: str) -> None:
        assert "pyinstaller" in workflow_content.lower()

    def test_workflow_runs_pyinstaller(self, workflow_content: str) -> None:
        assert "pyinstaller edgeml.spec" in workflow_content

    def test_workflow_verifies_binary(self, workflow_content: str) -> None:
        assert "--version" in workflow_content

    def test_workflow_generates_sha256(self, workflow_content: str) -> None:
        assert "sha256" in workflow_content.lower()

    def test_workflow_uploads_artifacts(self, workflow_content: str) -> None:
        assert "upload-artifact" in workflow_content

    def test_workflow_uploads_to_release(self, workflow_content: str) -> None:
        assert "gh release upload" in workflow_content

    def test_workflow_uses_macos_runners(self, workflow_content: str) -> None:
        """macOS arm64 requires macos-14, intel uses macos-13."""
        assert "macos-14" in workflow_content
        assert "macos-13" in workflow_content

    def test_workflow_does_not_overwrite_existing_release_yml(self) -> None:
        """The existing release.yml (PyPI) should still exist."""
        assert (ROOT / ".github" / "workflows" / "release.yml").is_file()


# ---------------------------------------------------------------------------
# Cross-file consistency
# ---------------------------------------------------------------------------


class TestConsistency:
    """Verify files are consistent with each other and the project."""

    def test_spec_entry_point_matches_setup_py(self) -> None:
        """edgeml.spec should use the same entry point as setup.py."""
        setup_content = (ROOT / "setup.py").read_text()
        assert "edgeml.cli:main" in setup_content
        spec_content = (ROOT / "edgeml.spec").read_text()
        assert "edgeml/cli.py" in spec_content

    def test_all_engines_in_spec(self) -> None:
        """Every engine in the engines directory should be a hidden import."""
        spec_content = (ROOT / "edgeml.spec").read_text()
        engines_dir = ROOT / "edgeml" / "engines"
        for py_file in engines_dir.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name == "base.py":
                continue
            module = f"edgeml.engines.{py_file.stem}"
            assert module in spec_content, f"Engine {module} not in hidden imports"

    def test_github_repo_consistent(self) -> None:
        """All files should reference the same GitHub repo."""
        repo = "edgeml-ai/edgeml-python"
        for path in [
            ROOT / "scripts" / "install.sh",
            ROOT / "homebrew" / "edgeml.rb",
        ]:
            content = path.read_text()
            assert repo in content, f"{path.name} missing repo reference"

    def test_version_in_formula_matches_setup(self) -> None:
        """The formula version should match setup.py version."""
        setup_content = (ROOT / "setup.py").read_text()
        formula_content = (ROOT / "homebrew" / "edgeml.rb").read_text()
        # Both should have version 1.0.0
        assert 'version="1.0.0"' in setup_content
        assert 'version "1.0.0"' in formula_content
