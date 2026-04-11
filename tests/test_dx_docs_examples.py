"""DX documentation validation — ensures CLI help and docs tell a coherent story."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from octomil.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCliHelpDx:
    """CLI --help shows the right top-level commands."""

    def test_top_level_help_shows_core_commands(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        for cmd in ("run", "embed", "transcribe", "serve"):
            assert cmd in result.output, f"'{cmd}' not in top-level --help"

    def test_run_help_says_local(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        # Should mention local / on-device / no server
        output_lower = result.output.lower()
        assert any(
            word in output_lower for word in ("local", "on-device", "no server")
        ), "run --help should mention local execution"

    def test_serve_help_says_openai_compatible(self):
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert (
            "openai" in output_lower or "compatible" in output_lower
        ), "serve --help should mention OpenAI compatibility"

    def test_embed_help_exists(self):
        runner = CliRunner()
        result = runner.invoke(main, ["embed", "--help"])
        assert result.exit_code == 0

    def test_transcribe_help_exists(self):
        runner = CliRunner()
        result = runner.invoke(main, ["transcribe", "--help"])
        assert result.exit_code == 0


class TestDocSnippetConsistency:
    """Verify doc files don't use banned placeholders in inference examples."""

    def _read_file(self, path: str) -> str:
        return (REPO_ROOT / path).read_text()

    def test_readme_no_your_api_key_in_inference(self):
        content = self._read_file("README.md")
        # YOUR_API_KEY should not appear — should be YOUR_SERVER_KEY or YOUR_CLIENT_KEY
        assert "YOUR_API_KEY" not in content, "README uses deprecated YOUR_API_KEY placeholder"

    def test_readme_has_local_cli_examples(self):
        content = self._read_file("README.md")
        assert "octomil run" in content
        assert "octomil embed" in content
        assert "octomil transcribe" in content

    def test_readme_has_serve_example(self):
        content = self._read_file("README.md")
        assert "octomil serve" in content

    def test_quickstart_no_your_api_key(self):
        content = self._read_file("docs/quickstart.md")
        assert "YOUR_API_KEY" not in content

    def test_routing_doc_no_quality_first_as_preset(self):
        content = self._read_file("docs/routing.md")
        # quality_first should not be listed as a primary preset
        # It can appear as a legacy alias mention
        lines = content.split("\n")
        for line in lines:
            if "quality_first" in line.lower():
                assert (
                    "legacy" in line.lower() or "alias" in line.lower() or "deprecated" in line.lower()
                ), f"quality_first appears without legacy/alias context: {line}"
