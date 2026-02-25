"""Tests for ollama:// URI scheme in octomil deploy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from octomil.cli import main
from octomil.sources.base import SourceResult
from octomil.sources.ollama import OllamaSource, _parse_ollama_ref


# ---------------------------------------------------------------------------
# _parse_ollama_ref
# ---------------------------------------------------------------------------


class TestParseOllamaRef:
    def test_with_tag(self):
        assert _parse_ollama_ref("gemma:2b") == ("gemma", "2b")

    def test_without_tag_defaults_latest(self):
        assert _parse_ollama_ref("llama3.2") == ("llama3.2", "latest")

    def test_preserves_dotted_names(self):
        assert _parse_ollama_ref("llama3.2:3b") == ("llama3.2", "3b")

    def test_preserves_complex_tag(self):
        assert _parse_ollama_ref("gemma:2b-instruct-q4") == ("gemma", "2b-instruct-q4")


# ---------------------------------------------------------------------------
# OllamaSource.check_cache
# ---------------------------------------------------------------------------


class TestOllamaSourceCheckCache:
    def test_returns_blob_path_when_cached(self, tmp_path):
        # Set up fake ollama cache structure
        # The manifest path is: manifests/registry.ollama.ai/library/<model>/<tag>
        # where <tag> is a FILE, not a directory.
        manifest_parent = (
            tmp_path / "manifests" / "registry.ollama.ai" / "library" / "gemma"
        )
        manifest_parent.mkdir(parents=True)
        manifest_file = manifest_parent / "2b"
        manifest_file.write_text(
            '{"layers": [{"mediaType": "application/vnd.ollama.image.model",'
            '"digest": "sha256:abc123"}]}'
        )

        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()
        blob_file = blob_dir / "sha256-abc123"
        blob_file.write_bytes(b"fake gguf data")

        source = OllamaSource(models_dir=str(tmp_path))
        result = source.check_cache("gemma:2b")
        assert result == str(blob_file)

    def test_returns_none_when_no_manifest(self, tmp_path):
        source = OllamaSource(models_dir=str(tmp_path))
        assert source.check_cache("nonexistent:7b") is None

    def test_returns_none_when_blob_missing(self, tmp_path):
        manifest_parent = (
            tmp_path / "manifests" / "registry.ollama.ai" / "library" / "gemma"
        )
        manifest_parent.mkdir(parents=True)
        manifest_file = manifest_parent / "2b"
        manifest_file.write_text(
            '{"layers": [{"mediaType": "application/vnd.ollama.image.model",'
            '"digest": "sha256:missing"}]}'
        )
        # No blob directory
        source = OllamaSource(models_dir=str(tmp_path))
        assert source.check_cache("gemma:2b") is None

    def test_default_tag_is_latest(self, tmp_path):
        manifest_parent = (
            tmp_path / "manifests" / "registry.ollama.ai" / "library" / "llama3"
        )
        manifest_parent.mkdir(parents=True)
        manifest_file = manifest_parent / "latest"
        manifest_file.write_text(
            '{"layers": [{"mediaType": "application/vnd.ollama.image.model",'
            '"digest": "sha256:def456"}]}'
        )
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()
        blob_file = blob_dir / "sha256-def456"
        blob_file.write_bytes(b"fake gguf")

        source = OllamaSource(models_dir=str(tmp_path))
        result = source.check_cache("llama3")
        assert result == str(blob_file)


# ---------------------------------------------------------------------------
# OllamaSource.resolve
# ---------------------------------------------------------------------------


class TestOllamaSourceResolve:
    def test_resolve_cached_model(self, tmp_path):
        manifest_parent = (
            tmp_path / "manifests" / "registry.ollama.ai" / "library" / "gemma"
        )
        manifest_parent.mkdir(parents=True)
        (manifest_parent / "2b").write_text(
            '{"layers": [{"mediaType": "application/vnd.ollama.image.model",'
            '"digest": "sha256:aaa111"}]}'
        )
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()
        blob_file = blob_dir / "sha256-aaa111"
        blob_file.write_bytes(b"gguf data")

        source = OllamaSource(models_dir=str(tmp_path))
        result = source.resolve("gemma:2b")
        assert isinstance(result, SourceResult)
        assert result.path == str(blob_file)
        assert result.source_type == "ollama"
        assert result.cached is True

    @patch("octomil.sources.ollama.shutil.which", return_value=None)
    def test_resolve_not_cached_no_cli_raises(self, mock_which, tmp_path):
        source = OllamaSource(models_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="not found in local cache"):
            source.resolve("nonexistent:7b")

    @patch("octomil.sources.ollama.subprocess.run")
    @patch("octomil.sources.ollama.shutil.which", return_value="/usr/bin/ollama")
    def test_resolve_pulls_and_succeeds(self, mock_which, mock_run, tmp_path):
        # First call to check_cache returns None (not cached yet)
        # After pull, we need the cache to exist
        manifest_parent = (
            tmp_path / "manifests" / "registry.ollama.ai" / "library" / "phi"
        )
        manifest_parent.mkdir(parents=True)
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()

        def fake_pull(*args, **kwargs):
            # Simulate ollama pull populating the cache
            (manifest_parent / "latest").write_text(
                '{"layers": [{"mediaType": "application/vnd.ollama.image.model",'
                '"digest": "sha256:pulled123"}]}'
            )
            (blob_dir / "sha256-pulled123").write_bytes(b"pulled model data")

        mock_run.side_effect = fake_pull

        source = OllamaSource(models_dir=str(tmp_path))
        result = source.resolve("phi")
        assert result.path == str(blob_dir / "sha256-pulled123")
        assert result.cached is False
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# CLI: ollama:// prefix stripping
# ---------------------------------------------------------------------------


class TestOllamaUriPrefixStripping:
    def test_strips_prefix_correctly(self):
        uri = "ollama://llama3.2:3b"
        ref = uri[len("ollama://") :]
        assert ref == "llama3.2:3b"

    def test_strips_prefix_no_tag(self):
        uri = "ollama://gemma"
        ref = uri[len("ollama://") :]
        assert ref == "gemma"

    def test_base_name_extraction(self):
        ref = "llama3.2:3b"
        base = ref.split(":")[0]
        assert base == "llama3.2"

    def test_base_name_no_tag(self):
        ref = "gemma"
        base = ref.split(":")[0]
        assert base == "gemma"


# ---------------------------------------------------------------------------
# CLI: octomil deploy ollama:// --phone
# ---------------------------------------------------------------------------


class TestDeployOllamaUri:
    @patch("octomil.commands.deploy.webbrowser.open")
    @patch("octomil.ollama.get_ollama_model")
    @patch("octomil.sources.ollama.OllamaSource.resolve")
    def test_deploy_ollama_uri_resolves_and_deploys(
        self, mock_resolve, mock_get_model, mock_open, monkeypatch
    ):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_resolve.return_value = SourceResult(
            path="/fake/.ollama/models/blobs/sha256-abc123",
            source_type="ollama",
            cached=True,
        )
        mock_get_model.return_value = MagicMock(
            name="gemma",
            size_display="1.6 GB",
            quantization="Q4_K_M",
        )

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "code": "OLL123",
            "expires_at": "2026-02-25T12:00:00Z",
        }

        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "done"}

        with (
            patch("httpx.post", return_value=mock_post_resp),
            patch("httpx.get", return_value=mock_poll_resp),
            patch("time.sleep"),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["deploy", "ollama://gemma:2b", "--phone"])

        assert result.exit_code == 0
        assert "Resolving ollama model: gemma:2b" in result.output
        assert "Found:" in result.output
        mock_resolve.assert_called_once_with("gemma:2b")

    @patch("octomil.sources.ollama.OllamaSource.resolve")
    def test_deploy_ollama_uri_resolve_error(self, mock_resolve, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")
        mock_resolve.side_effect = RuntimeError(
            "Ollama model 'bad:model' not found in local cache "
            "and ollama CLI is not installed."
        )

        runner = CliRunner()
        result = runner.invoke(main, ["deploy", "ollama://bad:model", "--phone"])

        assert result.exit_code != 0
        assert "Error:" in result.output
        assert "not found" in result.output

    def test_deploy_ollama_uri_empty_ref(self, monkeypatch):
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        runner = CliRunner()
        result = runner.invoke(main, ["deploy", "ollama://", "--phone"])

        assert result.exit_code != 0
        assert "requires a model name" in result.output

    @patch("octomil.commands.deploy.webbrowser.open")
    @patch("octomil.ollama.get_ollama_model")
    @patch("octomil.sources.ollama.OllamaSource.resolve")
    def test_deploy_ollama_uri_without_phone_flag(
        self, mock_resolve, mock_get_model, mock_open, monkeypatch
    ):
        """ollama:// URI should work even without --phone (resolves model first)."""
        monkeypatch.setenv("OCTOMIL_API_KEY", "test-key")

        mock_resolve.return_value = SourceResult(
            path="/fake/.ollama/models/blobs/sha256-abc123",
            source_type="ollama",
            cached=True,
        )

        runner = CliRunner()
        # Without --phone, the deploy goes through the non-phone path.
        # It will still resolve the ollama URI first, then hit the
        # normal deploy flow (which will likely fail due to no client).
        # We just verify the resolution happens.
        result = runner.invoke(main, ["deploy", "ollama://llama3.2"])

        assert "Resolving ollama model: llama3.2" in result.output
        assert "Found:" in result.output
        mock_resolve.assert_called_once_with("llama3.2")


# ---------------------------------------------------------------------------
# CLI: octomil models — deploy URI column
# ---------------------------------------------------------------------------


class TestModelsCommandDeployUri:
    @patch("octomil.ollama.is_ollama_running", return_value=True)
    @patch("octomil.ollama.list_ollama_models")
    def test_models_shows_deploy_uri(self, mock_list, mock_running):
        from octomil.ollama import OllamaModel

        mock_list.return_value = [
            OllamaModel(
                name="gemma:2b",
                size=1_678_000_000,
                family="gemma",
                quantization="Q4_K_M",
                parameter_size="2B",
                modified_at="2026-01-15T10:30:00Z",
                digest="sha256:abc123",
            ),
        ]
        runner = CliRunner()
        result = runner.invoke(main, ["models", "--source", "ollama"])
        assert result.exit_code == 0
        assert "ollama://gemma:2b" in result.output
        assert "DEPLOY URI" in result.output

    @patch("octomil.ollama.is_ollama_running", return_value=True)
    @patch("octomil.ollama.list_ollama_models")
    def test_models_shows_header_row(self, mock_list, mock_running):
        from octomil.ollama import OllamaModel

        mock_list.return_value = [
            OllamaModel(
                name="llama3.2:3b",
                size=2_048_000_000,
                family="llama",
                quantization="Q4_K_M",
                parameter_size="3B",
                modified_at="2026-01-10T08:00:00Z",
                digest="sha256:789abc",
            ),
        ]
        runner = CliRunner()
        result = runner.invoke(main, ["models", "--source", "ollama"])
        assert result.exit_code == 0
        assert "NAME" in result.output
        assert "SIZE" in result.output
        assert "QUANT" in result.output
        assert "FAMILY" in result.output
