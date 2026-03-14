"""Tests for the HuggingFace checkpoint resolver.

Mocks HfApi.list_repo_tree() and download functions to test resolution
logic without network calls or requiring huggingface_hub to be installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from octomil.sources.hf_resolver import (
    _SHARD_RE,
    resolve_hf_checkpoint,
)

# Module path for patching
_MOD = "octomil.sources.hf_resolver"


# ---------------------------------------------------------------------------
# Fake HF types — mimics huggingface_hub.hf_api.RepoFile
# ---------------------------------------------------------------------------


@dataclass
class FakeRepoFile:
    """Mimics huggingface_hub.hf_api.RepoFile for testing."""

    path: str
    size: int
    lfs: Optional[dict] = None


@dataclass
class FakeRepoFolder:
    """Mimics a non-file tree entry (folder) — no .size attribute."""

    path: str


# ---------------------------------------------------------------------------
# Tests: shard regex
# ---------------------------------------------------------------------------


class TestShardRegex:
    def test_matches_shard_pattern(self):
        m = _SHARD_RE.match("qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
        assert m is not None
        assert m.group(1) == "qwen2.5-7b-instruct-q4_k_m"
        assert m.group(2) == "00001"
        assert m.group(3) == "00002"

    def test_does_not_match_single_file(self):
        assert _SHARD_RE.match("qwen2.5-3b-instruct-q4_k_m.gguf") is None

    def test_does_not_match_non_gguf(self):
        assert _SHARD_RE.match("model-00001-of-00002.safetensors") is None


# ---------------------------------------------------------------------------
# Tests: single GGUF resolution
# ---------------------------------------------------------------------------


class TestResolveSingleGGUF:
    @patch(f"{_MOD}._hf_hub_download", return_value="/cache/model.gguf")
    @patch(f"{_MOD}._list_repo_files")
    def test_single_gguf_file(self, mock_list, mock_download):
        """Single GGUF file in repo -> kind='single_file', correct path."""
        # _list_repo_files already filters to file entries (has .path + .size)
        mock_list.return_value = [
            FakeRepoFile(
                path="qwen2.5-3b-instruct-q4_k_m.gguf",
                size=2_000_000_000,
                lfs={"sha256": "abc123"},
            ),
            FakeRepoFile(path="README.md", size=1000),
        ]

        result = resolve_hf_checkpoint(
            "Qwen/Qwen2.5-3B-Instruct-GGUF",
            artifact_format="gguf",
            quantization_hint="q4_k_m",
        )

        assert result.kind == "single_file"
        assert result.path == "/cache/model.gguf"
        assert len(result.files) == 1
        assert result.files[0].sha256 == "abc123"
        assert result.total_size_bytes == 2_000_000_000
        mock_download.assert_called_once_with(
            "Qwen/Qwen2.5-3B-Instruct-GGUF",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf",
            revision=None,
        )

    @patch(f"{_MOD}._list_repo_files")
    def test_no_gguf_files_raises(self, mock_list):
        """Repo with no GGUF files -> RuntimeError."""
        mock_list.return_value = [
            FakeRepoFile(path="README.md", size=1000),
            FakeRepoFile(path="config.json", size=500),
        ]

        with pytest.raises(RuntimeError, match="No GGUF files found"):
            resolve_hf_checkpoint("some/repo", artifact_format="gguf")


# ---------------------------------------------------------------------------
# Tests: sharded GGUF resolution
# ---------------------------------------------------------------------------


class TestResolveShardedGGUF:
    @patch(f"{_MOD}._snapshot_download", return_value="/cache/snapshots/abc")
    @patch(f"{_MOD}._list_repo_files")
    def test_sharded_gguf(self, mock_list, mock_snapshot):
        """Sharded GGUF -> kind='sharded', all shards in files[]."""
        mock_list.return_value = [
            FakeRepoFile(
                path="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                size=4_000_000_000,
                lfs={"sha256": "sha1"},
            ),
            FakeRepoFile(
                path="qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
                size=3_500_000_000,
                lfs={"sha256": "sha2"},
            ),
            FakeRepoFile(path="README.md", size=1000),
        ]

        result = resolve_hf_checkpoint(
            "Qwen/Qwen2.5-7B-Instruct-GGUF",
            artifact_format="gguf",
            quantization_hint="q4_k_m",
        )

        assert result.kind == "sharded"
        assert len(result.files) == 2
        assert result.total_size_bytes == 7_500_000_000
        assert result.path == os.path.join(
            "/cache/snapshots/abc",
            "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        )
        mock_snapshot.assert_called_once_with(
            "Qwen/Qwen2.5-7B-Instruct-GGUF",
            revision=None,
            allow_patterns=[
                "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
            ],
        )

    @patch(f"{_MOD}._snapshot_download", return_value="/cache/snapshots/abc")
    @patch(f"{_MOD}._list_repo_files")
    def test_sharded_with_multiple_quants(self, mock_list, mock_snapshot):
        """Sharded repo with multiple quants -> filters to correct quant."""
        mock_list.return_value = [
            # q4_k_m shards
            FakeRepoFile(path="model-q4_k_m-00001-of-00002.gguf", size=4_000_000_000),
            FakeRepoFile(path="model-q4_k_m-00002-of-00002.gguf", size=3_500_000_000),
            # q8_0 shards (should be filtered out)
            FakeRepoFile(path="model-q8_0-00001-of-00003.gguf", size=7_000_000_000),
            FakeRepoFile(path="model-q8_0-00002-of-00003.gguf", size=7_000_000_000),
            FakeRepoFile(path="model-q8_0-00003-of-00003.gguf", size=7_000_000_000),
        ]

        result = resolve_hf_checkpoint(
            "org/multi-quant-sharded",
            artifact_format="gguf",
            quantization_hint="q4_k_m",
        )

        assert result.kind == "sharded"
        assert len(result.files) == 2
        assert result.total_size_bytes == 7_500_000_000


# ---------------------------------------------------------------------------
# Tests: multi-quant filtering
# ---------------------------------------------------------------------------


class TestMultiQuantFiltering:
    @patch(f"{_MOD}._hf_hub_download", return_value="/cache/q4_k_m.gguf")
    @patch(f"{_MOD}._list_repo_files")
    def test_filters_by_quant_hint(self, mock_list, mock_download):
        """Multi-quant repo + hint -> filters to correct quantization."""
        mock_list.return_value = [
            FakeRepoFile(path="model-q4_k_m.gguf", size=2_000_000_000),
            FakeRepoFile(path="model-q8_0.gguf", size=4_000_000_000),
            FakeRepoFile(path="model-fp16.gguf", size=8_000_000_000),
        ]

        result = resolve_hf_checkpoint(
            "org/multi-quant-repo",
            artifact_format="gguf",
            quantization_hint="q4_k_m",
        )

        assert result.kind == "single_file"
        assert len(result.files) == 1
        mock_download.assert_called_once_with(
            "org/multi-quant-repo",
            filename="model-q4_k_m.gguf",
            revision=None,
        )

    @patch(f"{_MOD}._list_repo_files")
    def test_no_matching_quant_raises(self, mock_list):
        """No GGUF files matching quant hint -> RuntimeError."""
        mock_list.return_value = [
            FakeRepoFile(path="model-q8_0.gguf", size=4_000_000_000),
        ]

        with pytest.raises(RuntimeError, match="No GGUF files matching"):
            resolve_hf_checkpoint(
                "org/repo",
                artifact_format="gguf",
                quantization_hint="q4_k_m",
            )


# ---------------------------------------------------------------------------
# Tests: directory (MLX/safetensors) resolution
# ---------------------------------------------------------------------------


class TestResolveDirectory:
    @patch(f"{_MOD}._snapshot_download", return_value="/cache/snapshots/mlx")
    @patch(f"{_MOD}._list_repo_files")
    def test_mlx_directory(self, mock_list, mock_snapshot):
        """MLX directory repo -> kind='directory', only safe patterns."""
        mock_list.return_value = [
            FakeRepoFile(path="model.safetensors", size=2_000_000_000, lfs={"sha256": "st1"}),
            FakeRepoFile(path="config.json", size=500),
            FakeRepoFile(path="tokenizer.json", size=2000),
            FakeRepoFile(path="tokenizer.model", size=3000),
            # These should NOT be included in resolved files
            FakeRepoFile(path="README.md", size=1000),
            FakeRepoFile(path="convert.py", size=5000),
        ]

        result = resolve_hf_checkpoint(
            "mlx-community/gemma-3-1b-it-4bit",
            artifact_format="directory",
        )

        assert result.kind == "directory"
        assert result.path == "/cache/snapshots/mlx"

        file_paths = [f.path for f in result.files]
        assert "model.safetensors" in file_paths
        assert "config.json" in file_paths
        assert "tokenizer.json" in file_paths
        assert "tokenizer.model" in file_paths
        assert "README.md" not in file_paths
        assert "convert.py" not in file_paths

        # snapshot_download called with tight allow_patterns
        call_args = mock_snapshot.call_args
        patterns = call_args.kwargs.get("allow_patterns", [])
        assert "*.py" not in patterns
        assert "*.safetensors" in patterns


# ---------------------------------------------------------------------------
# Tests: unknown artifact format
# ---------------------------------------------------------------------------


class TestUnknownFormat:
    @patch(f"{_MOD}._list_repo_files")
    def test_unknown_format_raises(self, mock_list):
        """Unknown artifact_format -> RuntimeError."""
        mock_list.return_value = []

        with pytest.raises(RuntimeError, match="Unknown artifact_format"):
            resolve_hf_checkpoint("org/repo", artifact_format="unknown_format")


# ---------------------------------------------------------------------------
# Tests: revision pinning
# ---------------------------------------------------------------------------


class TestRevisionPinning:
    @patch(f"{_MOD}._hf_hub_download", return_value="/cache/model.gguf")
    @patch(f"{_MOD}._list_repo_files")
    def test_revision_passed_to_download(self, mock_list, mock_download):
        """Revision SHA is forwarded to hf_hub_download."""
        mock_list.return_value = [
            FakeRepoFile(path="model-q4_k_m.gguf", size=2_000_000_000),
        ]

        resolve_hf_checkpoint(
            "org/repo",
            revision="abc123def",
            artifact_format="gguf",
            quantization_hint="q4_k_m",
        )

        mock_download.assert_called_once_with(
            "org/repo",
            filename="model-q4_k_m.gguf",
            revision="abc123def",
        )

    @patch(f"{_MOD}._snapshot_download", return_value="/cache/snapshots/abc")
    @patch(f"{_MOD}._list_repo_files")
    def test_revision_passed_to_snapshot(self, mock_list, mock_snapshot):
        """Revision SHA is forwarded to snapshot_download for shards."""
        mock_list.return_value = [
            FakeRepoFile(path="model-00001-of-00002.gguf", size=4_000_000_000),
            FakeRepoFile(path="model-00002-of-00002.gguf", size=3_500_000_000),
        ]

        resolve_hf_checkpoint(
            "org/sharded-repo",
            revision="deadbeef",
            artifact_format="gguf",
        )

        mock_snapshot.assert_called_once()
        assert mock_snapshot.call_args.kwargs["revision"] == "deadbeef"


# ---------------------------------------------------------------------------
# Tests: resolution context flow (alias -> _try_source -> resolve -> resolver)
# ---------------------------------------------------------------------------


class TestResolutionContextFlow:
    """Verify resolution context propagates from alias through to HuggingFaceSource."""

    def test_manifest_to_aliases_carries_context(self):
        """_manifest_to_aliases() produces dicts for HF sources."""
        from octomil.sources.resolver import _manifest_to_aliases

        manifest = {
            "models": [
                {
                    "id": "qwen2.5-7b",
                    "packages": [
                        {
                            "runtime_executor": "llamacpp",
                            "artifact_format": "gguf",
                            "quantization": "Q4_K_M",
                            "resources": [
                                {
                                    "kind": "weights",
                                    "uri": "hf://Qwen/Qwen2.5-7B-Instruct-GGUF",
                                    "metadata": {
                                        "uri_type": "repo",
                                        "quantization": "q4_k_m",
                                        "artifact_format": "gguf",
                                        "revision": "abc123",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        aliases = _manifest_to_aliases(manifest)
        hf_alias = aliases["qwen2.5-7b"]["hf"]

        assert isinstance(hf_alias, dict)
        assert hf_alias["repo_id"] == "Qwen/Qwen2.5-7B-Instruct-GGUF"
        assert hf_alias["filename"] is None
        assert hf_alias["revision"] == "abc123"
        assert hf_alias["quantization_hint"] == "q4_k_m"
        assert hf_alias["artifact_format"] == "gguf"
        assert hf_alias["uri_type"] == "repo"

    def test_manifest_to_aliases_single_file(self):
        """Single-file GGUF alias still carries context."""
        from octomil.sources.resolver import _manifest_to_aliases

        manifest = {
            "models": [
                {
                    "id": "qwen2.5-3b",
                    "packages": [
                        {
                            "runtime_executor": "llamacpp",
                            "artifact_format": "gguf",
                            "quantization": "Q4_K_M",
                            "resources": [
                                {
                                    "kind": "weights",
                                    "uri": "hf://Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf",
                                    "metadata": {"uri_type": "file"},
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        aliases = _manifest_to_aliases(manifest)
        hf_alias = aliases["qwen2.5-3b"]["hf"]

        assert isinstance(hf_alias, dict)
        assert hf_alias["repo_id"] == "Qwen/Qwen2.5-3B-Instruct-GGUF"
        assert hf_alias["filename"] == "qwen2.5-3b-instruct-q4_k_m.gguf"
        assert hf_alias["uri_type"] == "file"

    def test_manifest_to_aliases_ollama_stays_string(self):
        """Ollama aliases remain bare strings."""
        from octomil.sources.resolver import _manifest_to_aliases

        manifest = {
            "models": [
                {
                    "id": "test-model",
                    "packages": [
                        {
                            "runtime_executor": "ollama",
                            "resources": [
                                {"kind": "weights", "uri": "gemma3:1b"},
                            ],
                        }
                    ],
                }
            ],
        }

        aliases = _manifest_to_aliases(manifest)
        assert aliases["test-model"]["ollama"] == "gemma3:1b"
        assert isinstance(aliases["test-model"]["ollama"], str)


# ---------------------------------------------------------------------------
# Tests: GGUFSource metadata extraction
# ---------------------------------------------------------------------------


class TestGGUFSourceMetadata:
    def test_gguf_source_with_metadata(self):
        """GGUFSource carries resolution metadata from manifest."""
        from octomil.models.catalog import _package_to_variant_field

        pkg = {
            "runtime_executor": "llamacpp",
            "artifact_format": "gguf",
            "resources": [
                {
                    "kind": "weights",
                    "uri": "hf://Qwen/Qwen2.5-7B-Instruct-GGUF",
                    "metadata": {
                        "uri_type": "repo",
                        "quantization": "q4_k_m",
                        "revision": "abc123",
                    },
                }
            ],
        }

        mlx, gguf, ort, mlc, ollama, source = _package_to_variant_field(pkg)
        assert gguf is not None
        assert gguf.repo == "Qwen/Qwen2.5-7B-Instruct-GGUF"
        assert gguf.filename == ""
        assert gguf.revision == "abc123"
        assert gguf.quantization_hint == "q4_k_m"
        assert gguf.uri_type == "repo"

    def test_gguf_source_defaults(self):
        """GGUFSource without metadata uses defaults."""
        from octomil.models.catalog import _package_to_variant_field

        pkg = {
            "runtime_executor": "llamacpp",
            "artifact_format": "gguf",
            "resources": [
                {
                    "kind": "weights",
                    "uri": "hf://org/repo/model.gguf",
                }
            ],
        }

        mlx, gguf, ort, mlc, ollama, source = _package_to_variant_field(pkg)
        assert gguf is not None
        assert gguf.filename == "model.gguf"
        assert gguf.revision is None
        assert gguf.quantization_hint is None
        assert gguf.uri_type == "file"
