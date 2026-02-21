"""Tests for the model registry, tag syntax parser, and source backends."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the edgeml package is importable from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from edgeml.model_registry import (
    DEFAULT_TAG,
    MODEL_FAMILIES,
    ModelResolutionError,
    ModelSource,
    _sort_sources_by_trust,
    _suggest_families,
    get_family,
    list_families,
    parse_model_tag,
    resolve_model,
)
from edgeml.sources.huggingface import HuggingFaceSource
from edgeml.sources.kaggle import KaggleSource
from edgeml.sources.ollama import OllamaSource, _parse_ollama_ref


# =====================================================================
# Tag parsing
# =====================================================================


class TestParseModelTag:
    """Tests for the ``name:tag`` parser."""

    def test_plain_name_gets_default_tag(self) -> None:
        """gemma-4b -> family=gemma-4b, tag=q4_k_m (default)."""
        family, tag = parse_model_tag("gemma-4b")
        assert family == "gemma-4b"
        assert tag == DEFAULT_TAG

    def test_name_with_tag(self) -> None:
        """gemma-4b:q8_0 -> family=gemma-4b, tag=q8_0."""
        family, tag = parse_model_tag("gemma-4b:q8_0")
        assert family == "gemma-4b"
        assert tag == "q8_0"

    def test_name_with_fp16_tag(self) -> None:
        """llama-1b:fp16 -> family=llama-1b, tag=fp16."""
        family, tag = parse_model_tag("llama-1b:fp16")
        assert family == "llama-1b"
        assert tag == "fp16"

    def test_full_repo_passthrough(self) -> None:
        """mlx-community/foo -> (None, full path)."""
        family, tag = parse_model_tag("mlx-community/foo")
        assert family is None
        assert tag == "mlx-community/foo"

    def test_full_repo_with_colon_passthrough(self) -> None:
        """user/repo:branch -> (None, full path with colon)."""
        family, tag = parse_model_tag("user/repo:branch")
        assert family is None
        assert tag == "user/repo:branch"

    def test_bartowski_repo_passthrough(self) -> None:
        """bartowski/Model-GGUF -> passes through."""
        family, tag = parse_model_tag("bartowski/Model-GGUF")
        assert family is None
        assert tag == "bartowski/Model-GGUF"


# =====================================================================
# Model resolution
# =====================================================================


class TestResolveModel:
    """Tests for resolve_model()."""

    def test_resolve_default_tag(self) -> None:
        """gemma-4b resolves to the default variant (q4_k_m)."""
        result = resolve_model("gemma-4b")
        assert result.family == "gemma-4b"
        assert result.tag == "q4_k_m"
        assert result.mlx_repo == "mlx-community/gemma-3-4b-it-4bit"
        assert result.variant is not None
        assert result.source is not None

    def test_resolve_explicit_tag(self) -> None:
        """gemma-4b:q8_0 resolves to the q8_0 variant."""
        result = resolve_model("gemma-4b:q8_0")
        assert result.family == "gemma-4b"
        assert result.tag == "q8_0"
        assert result.mlx_repo == "mlx-community/gemma-3-4b-it-8bit"

    def test_resolve_gguf_backend(self) -> None:
        """gemma-4b with gguf backend picks a GGUF source."""
        result = resolve_model("gemma-4b", backend="gguf")
        assert result.source is not None
        assert result.source.file is not None
        assert result.source.file.endswith(".gguf")

    def test_resolve_mlx_backend(self) -> None:
        """gemma-4b with mlx backend returns mlx_repo."""
        result = resolve_model("gemma-4b", backend="mlx")
        assert result.mlx_repo is not None
        assert "mlx-community" in result.mlx_repo

    def test_resolve_full_repo_passthrough(self) -> None:
        """Full repo paths pass through without registry lookup."""
        result = resolve_model("mlx-community/custom-model")
        assert result.family is None
        assert result.tag == "mlx-community/custom-model"
        assert result.source is not None
        assert result.source.ref == "mlx-community/custom-model"
        assert result.mlx_repo is None

    def test_resolve_unknown_model_raises(self) -> None:
        """Unknown model names raise ModelResolutionError."""
        with pytest.raises(ModelResolutionError, match="Unknown model 'nonexistent'"):
            resolve_model("nonexistent")

    def test_resolve_unknown_model_suggests(self) -> None:
        """Unknown model close to a real name includes suggestions."""
        with pytest.raises(ModelResolutionError, match="Did you mean"):
            resolve_model("gemma-4")  # close to gemma-4b

    def test_resolve_unknown_tag_raises(self) -> None:
        """Unknown tags on known models raise ModelResolutionError."""
        with pytest.raises(ModelResolutionError, match="Unknown tag 'q2_k'"):
            resolve_model("gemma-4b:q2_k")

    def test_all_families_have_default_variant(self) -> None:
        """Every family's default_tag must exist in its variants."""
        for name, family in MODEL_FAMILIES.items():
            assert family.default_tag in family.variants, (
                f"{name} has default_tag '{family.default_tag}' but no matching variant"
            )


# =====================================================================
# Trust priority
# =====================================================================


class TestTrustPriority:
    """Tests for source trust ordering."""

    def test_official_before_community(self) -> None:
        """Official sources should sort before community ones."""
        sources = [
            ModelSource(type="huggingface", ref="community/repo", trust="community"),
            ModelSource(type="huggingface", ref="official/repo", trust="official"),
            ModelSource(type="ollama", ref="model", trust="curated"),
        ]
        sorted_sources = _sort_sources_by_trust(sources)
        assert sorted_sources[0].trust == "official"
        assert sorted_sources[1].trust == "curated"
        assert sorted_sources[2].trust == "community"

    def test_resolve_picks_best_trust(self) -> None:
        """The resolved source should be the highest-trust available."""
        result = resolve_model("gemma-4b")
        assert result.source is not None
        # Google official repo should be first for gemma
        assert result.source.trust == "official"


# =====================================================================
# Registry completeness
# =====================================================================


class TestRegistryCompleteness:
    """Verify the registry covers all the old hardcoded models."""

    OLD_MLX_MODELS = [
        "gemma-1b",
        "gemma-4b",
        "gemma-12b",
        "gemma-27b",
        "llama-1b",
        "llama-3b",
        "llama-8b",
        "phi-4",
        "phi-mini",
        "qwen-1.5b",
        "qwen-3b",
        "qwen-7b",
        "mistral-7b",
        "smollm-360m",
    ]

    OLD_GGUF_MODELS = [
        "gemma-1b",
        "gemma-4b",
        "llama-1b",
        "llama-3b",
        "llama-8b",
        "phi-mini",
        "qwen-1.5b",
        "qwen-3b",
        "mistral-7b",
        "smollm-360m",
    ]

    def test_all_old_mlx_models_in_registry(self) -> None:
        """All models from the old _MLX_MODELS dict are in the registry."""
        for model_name in self.OLD_MLX_MODELS:
            assert model_name in MODEL_FAMILIES, (
                f"'{model_name}' was in old _MLX_MODELS but missing from registry"
            )
            family = MODEL_FAMILIES[model_name]
            default_variant = family.variants.get(family.default_tag)
            assert default_variant is not None
            assert default_variant.mlx is not None, (
                f"'{model_name}' missing MLX repo in default variant"
            )

    def test_all_old_gguf_models_in_registry(self) -> None:
        """All models from the old _GGUF_MODELS dict are in the registry."""
        for model_name in self.OLD_GGUF_MODELS:
            assert model_name in MODEL_FAMILIES, (
                f"'{model_name}' was in old _GGUF_MODELS but missing from registry"
            )
            family = MODEL_FAMILIES[model_name]
            default_variant = family.variants.get(family.default_tag)
            assert default_variant is not None
            # Must have at least one GGUF source
            gguf_sources = [
                s
                for s in default_variant.sources
                if s.file and s.file.endswith(".gguf")
            ]
            assert len(gguf_sources) > 0, (
                f"'{model_name}' missing GGUF source in default variant"
            )


# =====================================================================
# Backwards-compatible dicts
# =====================================================================


class TestBackwardsCompatDicts:
    """Verify the _MLX_MODELS and _GGUF_MODELS dicts in serve.py are populated."""

    def test_mlx_models_populated(self) -> None:
        from edgeml.serve import _MLX_MODELS

        assert len(_MLX_MODELS) >= 13, (
            f"Expected at least 13 MLX models, got {len(_MLX_MODELS)}"
        )
        assert "gemma-4b" in _MLX_MODELS
        assert "mlx-community" in _MLX_MODELS["gemma-4b"]

    def test_gguf_models_populated(self) -> None:
        from edgeml.serve import _GGUF_MODELS

        assert len(_GGUF_MODELS) >= 9, (
            f"Expected at least 9 GGUF models, got {len(_GGUF_MODELS)}"
        )
        assert "gemma-4b" in _GGUF_MODELS
        repo, fname = _GGUF_MODELS["gemma-4b"]
        assert fname.endswith(".gguf")


# =====================================================================
# resolve_model_name (serve.py integration)
# =====================================================================


class TestResolveModelName:
    """Tests for the updated resolve_model_name in serve.py."""

    def test_mlx_resolution(self) -> None:
        from edgeml.serve import resolve_model_name

        result = resolve_model_name("gemma-4b", "mlx")
        assert "mlx-community" in result

    def test_mlx_with_tag(self) -> None:
        from edgeml.serve import resolve_model_name

        result = resolve_model_name("gemma-4b:q8_0", "mlx")
        assert "8bit" in result or "8b" in result.lower()

    def test_gguf_resolution(self) -> None:
        from edgeml.serve import resolve_model_name

        result = resolve_model_name("gemma-4b", "gguf")
        assert result == "gemma-4b"

    def test_full_repo_passthrough(self) -> None:
        from edgeml.serve import resolve_model_name

        result = resolve_model_name("user/custom-model", "mlx")
        assert result == "user/custom-model"

    def test_unknown_model_raises(self) -> None:
        from edgeml.serve import resolve_model_name

        with pytest.raises(ValueError, match="Unknown model"):
            resolve_model_name("fake-model", "mlx")


# =====================================================================
# Ollama source backend
# =====================================================================


class TestOllamaSource:
    """Tests for the OllamaSource backend."""

    def test_parse_ollama_ref_with_tag(self) -> None:
        model, tag = _parse_ollama_ref("gemma3:4b")
        assert model == "gemma3"
        assert tag == "4b"

    def test_parse_ollama_ref_without_tag(self) -> None:
        model, tag = _parse_ollama_ref("llama3.2")
        assert model == "llama3.2"
        assert tag == "latest"

    def test_check_cache_found(self, tmp_path) -> None:
        """Ollama cache detection with a mocked filesystem."""
        # Build fake Ollama directory structure
        models_dir = tmp_path / "models"
        manifest_dir = (
            models_dir
            / "manifests"
            / "registry.ollama.ai"
            / "library"
            / "gemma3"
            / "4b"
        )
        manifest_dir.parent.mkdir(parents=True)
        blob_dir = models_dir / "blobs"
        blob_dir.mkdir(parents=True)

        # Create a fake blob
        fake_hash = "abc123def456"
        blob_path = blob_dir / f"sha256-{fake_hash}"
        blob_path.write_bytes(b"fake model data")

        # Create manifest pointing to the blob
        manifest = {
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": f"sha256:{fake_hash}",
                    "size": 1000,
                }
            ]
        }
        manifest_dir.write_text(json.dumps(manifest))

        source = OllamaSource(models_dir=str(models_dir))
        result = source.check_cache("gemma3:4b")
        assert result is not None
        assert result == str(blob_path)

    def test_check_cache_not_found(self, tmp_path) -> None:
        """Returns None when model is not in Ollama cache."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        source = OllamaSource(models_dir=str(models_dir))
        result = source.check_cache("nonexistent:latest")
        assert result is None

    def test_check_cache_manifest_missing_blob(self, tmp_path) -> None:
        """Returns None when manifest exists but blob doesn't."""
        models_dir = tmp_path / "models"
        manifest_dir = (
            models_dir
            / "manifests"
            / "registry.ollama.ai"
            / "library"
            / "test"
            / "latest"
        )
        manifest_dir.parent.mkdir(parents=True)
        blob_dir = models_dir / "blobs"
        blob_dir.mkdir(parents=True)

        manifest = {
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256:missing",
                    "size": 1000,
                }
            ]
        }
        manifest_dir.write_text(json.dumps(manifest))

        source = OllamaSource(models_dir=str(models_dir))
        result = source.check_cache("test:latest")
        assert result is None

    @patch("shutil.which", return_value="/usr/local/bin/ollama")
    def test_is_available_with_cli(self, mock_which) -> None:
        source = OllamaSource()
        assert source.is_available() is True

    @patch("shutil.which", return_value=None)
    def test_is_available_without_cli(self, mock_which) -> None:
        source = OllamaSource()
        assert source.is_available() is False

    def test_resolve_from_cache(self, tmp_path) -> None:
        """resolve() returns cached result when available."""
        models_dir = tmp_path / "models"
        manifest_dir = (
            models_dir
            / "manifests"
            / "registry.ollama.ai"
            / "library"
            / "gemma3"
            / "4b"
        )
        manifest_dir.parent.mkdir(parents=True)
        blob_dir = models_dir / "blobs"
        blob_dir.mkdir(parents=True)

        fake_hash = "resolve_test_hash"
        blob_path = blob_dir / f"sha256-{fake_hash}"
        blob_path.write_bytes(b"model weights")

        manifest = {
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": f"sha256:{fake_hash}",
                    "size": 100,
                }
            ]
        }
        manifest_dir.write_text(json.dumps(manifest))

        source = OllamaSource(models_dir=str(models_dir))
        result = source.resolve("gemma3:4b")
        assert result.cached is True
        assert result.source_type == "ollama"
        assert result.path == str(blob_path)


# =====================================================================
# HuggingFace source backend
# =====================================================================


class TestHuggingFaceSource:
    """Tests for the HuggingFaceSource backend."""

    def test_is_available_when_installed(self) -> None:
        """Should return True when huggingface_hub can be imported."""
        source = HuggingFaceSource()
        # huggingface_hub is likely installed in the test environment
        # If not, the test would need to be mocked — but we test both paths
        with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}):
            assert source.is_available() is True

    def test_is_available_when_not_installed(self) -> None:
        """Should return False when huggingface_hub import fails."""
        source = HuggingFaceSource()
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert source.is_available() is False


# =====================================================================
# Kaggle source backend
# =====================================================================


class TestKaggleSource:
    """Tests for the KaggleSource backend."""

    @patch("shutil.which", return_value="/usr/local/bin/kaggle")
    def test_is_available_with_cli(self, mock_which) -> None:
        source = KaggleSource()
        assert source.is_available() is True

    @patch("shutil.which", return_value=None)
    def test_is_available_without_cli(self, mock_which) -> None:
        source = KaggleSource()
        assert source.is_available() is False

    @patch("shutil.which", return_value=None)
    def test_resolve_without_cli_raises(self, mock_which) -> None:
        source = KaggleSource()
        with pytest.raises(RuntimeError, match="Kaggle CLI is not installed"):
            source.resolve("google/gemma/pyTorch/gemma-2b")

    @patch("shutil.which", return_value="/usr/local/bin/kaggle")
    def test_resolve_with_cli_raises_not_implemented(self, mock_which) -> None:
        source = KaggleSource()
        with pytest.raises(RuntimeError, match="not yet implemented"):
            source.resolve("google/gemma/pyTorch/gemma-2b")


# =====================================================================
# Source fallback chain
# =====================================================================


class TestSourceFallback:
    """Tests for the source selection / fallback logic."""

    def test_gguf_backend_prefers_gguf_sources(self) -> None:
        """When backend=gguf, GGUF sources should be selected."""
        result = resolve_model("gemma-4b", backend="gguf")
        assert result.source is not None
        assert result.source.file is not None
        assert result.source.file.endswith(".gguf")

    def test_mlx_backend_has_mlx_repo(self) -> None:
        """When backend=mlx, mlx_repo should be populated."""
        result = resolve_model("gemma-4b", backend="mlx")
        assert result.mlx_repo is not None

    def test_auto_backend_includes_everything(self) -> None:
        """Default backend includes both MLX and source info."""
        result = resolve_model("gemma-4b", backend="auto")
        assert result.mlx_repo is not None
        assert result.source is not None


# =====================================================================
# edgeml list CLI command
# =====================================================================


class TestListCLI:
    """Tests for the ``edgeml list`` CLI command output."""

    def test_list_all_families(self) -> None:
        """edgeml list without args shows a table of all families."""
        from click.testing import CliRunner
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "gemma-4b" in result.output
        assert "llama-8b" in result.output
        assert "model families available" in result.output

    def test_list_specific_family(self) -> None:
        """edgeml list gemma-4b shows variants with sources."""
        from click.testing import CliRunner
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["list", "gemma-4b"])
        assert result.exit_code == 0
        assert "gemma-4b" in result.output
        assert "Google" in result.output
        assert "q4_k_m" in result.output
        assert "official" in result.output or "[official]" in result.output

    def test_list_unknown_family_error(self) -> None:
        """edgeml list nonexistent shows error with suggestions."""
        from click.testing import CliRunner
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["list", "nonexistent"])
        assert result.exit_code != 0
        assert "Unknown model family" in result.output

    def test_list_close_match_suggests(self) -> None:
        """edgeml list gemma-4 (typo) suggests gemma-4b."""
        from click.testing import CliRunner
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["list", "gemma-4"])
        assert result.exit_code != 0
        assert "Did you mean" in result.output


# =====================================================================
# Helper functions
# =====================================================================


class TestHelpers:
    """Tests for internal helper functions."""

    def test_suggest_families_close_match(self) -> None:
        suggestions = _suggest_families("gemma-4")
        assert "gemma-4b" in suggestions

    def test_suggest_families_no_match(self) -> None:
        suggestions = _suggest_families("zzzznotamodel")
        assert suggestions == []

    def test_list_families_returns_all(self) -> None:
        families = list_families()
        assert len(families) == len(MODEL_FAMILIES)
        assert "gemma-4b" in families

    def test_get_family_exists(self) -> None:
        family = get_family("gemma-4b")
        assert family is not None
        assert family.publisher == "Google"

    def test_get_family_not_exists(self) -> None:
        family = get_family("nonexistent")
        assert family is None
