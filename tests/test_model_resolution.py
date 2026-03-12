"""Tests for the unified model resolution layer (octomil.models).

Covers:
- Parser: model:variant syntax, bare names, passthrough
- Catalog: structure, engine coverage (with mock server data)
- Resolver: engine selection, quant alias mapping, error handling
- Backward compatibility: existing short names, _MLX_MODELS, _GGUF_MODELS
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the octomil package is importable from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from octomil.models.catalog import (
    CATALOG,
    MODEL_ALIASES,
    _resolve_alias,
    get_model,
    list_models,
    resolve_ollama_tag,
    supports_engine,
)
from octomil.models.parser import normalize_variant, parse
from octomil.models.resolver import ModelResolutionError, _resolve_via_server, resolve

# =====================================================================
# Parser tests
# =====================================================================


class TestParse:
    """Tests for the model name parser."""

    def test_bare_name(self) -> None:
        """Bare name like 'phi-mini' parses to family with no variant."""
        p = parse("phi-mini")
        assert p.family == "phi-mini"
        assert p.variant is None
        assert p.is_passthrough is False

    def test_name_with_variant(self) -> None:
        """'gemma-3b:4bit' splits into family and variant."""
        p = parse("gemma-3b:4bit")
        assert p.family == "gemma-3b"
        assert p.variant == "4bit"
        assert p.is_passthrough is False

    def test_name_with_engine_specific_variant(self) -> None:
        """'gemma-3b:q4_k_m' parses the engine-specific quant."""
        p = parse("gemma-3b:q4_k_m")
        assert p.family == "gemma-3b"
        assert p.variant == "q4_k_m"

    def test_name_with_fp16(self) -> None:
        """'llama-8b:fp16' parses full precision variant."""
        p = parse("llama-8b:fp16")
        assert p.family == "llama-8b"
        assert p.variant == "fp16"

    def test_name_with_8bit(self) -> None:
        """'gemma-1b:8bit' parses 8-bit variant."""
        p = parse("gemma-1b:8bit")
        assert p.family == "gemma-1b"
        assert p.variant == "8bit"

    def test_hf_repo_passthrough(self) -> None:
        """Full HuggingFace repo ID passes through without parsing."""
        p = parse("some-org/some-model")
        assert p.family is None
        assert p.variant is None
        assert p.is_passthrough is True
        assert p.raw == "some-org/some-model"

    def test_local_gguf_passthrough(self) -> None:
        """Local .gguf file path passes through."""
        p = parse("./my-model.gguf")
        assert p.is_passthrough is True
        assert p.is_local_file is True
        assert p.raw == "./my-model.gguf"

    def test_local_pte_passthrough(self) -> None:
        """Local .pte file path passes through."""
        p = parse("model.pte")
        assert p.is_passthrough is True
        assert p.is_local_file is True

    def test_case_normalization(self) -> None:
        """Family and variant are lowercased."""
        p = parse("Gemma-4B:Q4_K_M")
        assert p.family == "gemma-4b"
        assert p.variant == "q4_k_m"

    def test_raw_preserved(self) -> None:
        """The raw input string is preserved."""
        p = parse("gemma-3b:4bit")
        assert p.raw == "gemma-3b:4bit"


class TestNormalizeVariant:
    """Tests for quant alias normalization."""

    def test_4bit(self) -> None:
        assert normalize_variant("4bit") == "4bit"

    def test_8bit(self) -> None:
        assert normalize_variant("8bit") == "8bit"

    def test_fp16(self) -> None:
        assert normalize_variant("fp16") == "fp16"

    def test_f16_alias(self) -> None:
        assert normalize_variant("f16") == "fp16"

    def test_16bit_alias(self) -> None:
        assert normalize_variant("16bit") == "fp16"

    def test_q4_k_m_alias(self) -> None:
        assert normalize_variant("q4_k_m") == "4bit"

    def test_q8_0_alias(self) -> None:
        assert normalize_variant("q8_0") == "8bit"

    def test_default_alias(self) -> None:
        assert normalize_variant("default") == "4bit"

    def test_case_insensitive(self) -> None:
        assert normalize_variant("Q4_K_M") == "4bit"
        assert normalize_variant("FP16") == "fp16"

    def test_unknown_passes_through(self) -> None:
        """Unknown variant strings pass through lowercased."""
        assert normalize_variant("custom_quant") == "custom_quant"


# =====================================================================
# Catalog tests
# =====================================================================


class TestCatalog:
    """Tests for the unified model catalog (server-fetched, mocked)."""

    def test_catalog_not_empty(self) -> None:
        """Catalog should have model families from mock server data."""
        assert len(CATALOG) > 0

    def test_all_families_have_default_variant(self) -> None:
        """Every family must have its default_quant variant."""
        for name, entry in CATALOG.items():
            assert entry.default_quant in entry.variants, (
                f"{name} has default_quant '{entry.default_quant}' but no matching variant"
            )

    def test_all_families_have_publisher(self) -> None:
        """Every family must have a publisher."""
        for name, entry in CATALOG.items():
            assert entry.publisher, f"{name} missing publisher"

    def test_all_families_have_params(self) -> None:
        """Every family must have params (e.g. '4B')."""
        for name, entry in CATALOG.items():
            assert entry.params, f"{name} missing params"

    def test_all_families_have_engines(self) -> None:
        """Every family must declare at least one engine."""
        for name, entry in CATALOG.items():
            assert len(entry.engines) > 0, f"{name} has no engines"

    def test_list_models_sorted(self) -> None:
        """list_models() returns sorted model names."""
        names = list_models()
        assert names == sorted(names)
        assert len(names) == len(CATALOG)

    def test_get_model_exists(self) -> None:
        """get_model() returns an entry for known models."""
        entry = get_model("gemma-4b")
        assert entry is not None
        assert entry.publisher == "Google"

    def test_get_model_not_exists(self) -> None:
        """get_model() returns None for unknown models."""
        assert get_model("nonexistent") is None

    def test_supports_engine(self) -> None:
        """supports_engine() checks the engines set."""
        assert supports_engine("gemma-4b", "mlx-lm") is True
        assert supports_engine("gemma-4b", "llama.cpp") is True
        assert supports_engine("qwen-7b", "llama.cpp") is False
        assert supports_engine("nonexistent", "mlx-lm") is False


class TestCatalogMLXCoverage:
    """Verify models that should have MLX artifacts do."""

    EXPECTED_MLX_MODELS = [
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

    def test_mlx_artifacts_present(self) -> None:
        """Default variant of MLX-capable models must have mlx repo."""
        for name in self.EXPECTED_MLX_MODELS:
            entry = CATALOG[name]
            default = entry.variants[entry.default_quant]
            assert default.mlx is not None, f"{name} default variant missing MLX repo"


class TestCatalogGGUFCoverage:
    """Verify models that should have GGUF artifacts do."""

    EXPECTED_GGUF_MODELS = [
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

    def test_gguf_artifacts_present(self) -> None:
        """Default variant of GGUF-capable models must have GGUF source."""
        for name in self.EXPECTED_GGUF_MODELS:
            entry = CATALOG[name]
            default = entry.variants[entry.default_quant]
            assert default.gguf is not None, f"{name} default variant missing GGUF source"
            assert default.gguf.filename.endswith(".gguf"), f"{name} GGUF filename does not end with .gguf"


# =====================================================================
# Alias tests
# =====================================================================


class TestModelAliases:
    """Tests for user-friendly model name aliases."""

    def test_phi_4_mini_alias(self) -> None:
        """phi-4-mini resolves to phi-mini."""
        assert _resolve_alias("phi-4-mini") == "phi-mini"
        entry = get_model("phi-4-mini")
        assert entry is not None
        assert entry.publisher == "Microsoft"

    def test_llama_3_2_aliases(self) -> None:
        """llama-3.2-* aliases resolve to llama-* catalog keys."""
        assert _resolve_alias("llama-3.2-1b") == "llama-1b"
        assert _resolve_alias("llama-3.2-3b") == "llama-3b"
        assert get_model("llama-3.2-1b") is not None
        assert get_model("llama-3.2-3b") is not None

    def test_qwen_2_5_aliases(self) -> None:
        """qwen-2.5-* aliases resolve to qwen-* catalog keys."""
        assert _resolve_alias("qwen-2.5-1.5b") == "qwen-1.5b"
        assert _resolve_alias("qwen-2.5-3b") == "qwen-3b"
        assert get_model("qwen-2.5-1.5b") is not None

    def test_gemma_3_aliases(self) -> None:
        """gemma-3-* aliases resolve to gemma-* catalog keys."""
        assert _resolve_alias("gemma-3-1b") == "gemma-1b"
        assert _resolve_alias("gemma-3-4b") == "gemma-4b"
        assert get_model("gemma-3-1b") is not None

    def test_canonical_names_pass_through(self) -> None:
        """Canonical names (already in catalog) pass through unchanged."""
        assert _resolve_alias("phi-mini") == "phi-mini"
        assert _resolve_alias("llama-1b") == "llama-1b"
        assert _resolve_alias("gemma-4b") == "gemma-4b"

    def test_unknown_names_pass_through(self) -> None:
        """Unknown names pass through (let resolver handle the error)."""
        assert _resolve_alias("totally-fake-model") == "totally-fake-model"

    def test_all_aliases_map_to_valid_catalog_entries(self) -> None:
        """Every alias must point to a real catalog key."""
        for alias, canonical in MODEL_ALIASES.items():
            assert canonical in CATALOG, f"Alias '{alias}' maps to '{canonical}' which is not in CATALOG"

    def test_resolve_with_alias(self) -> None:
        """resolve() works with aliased model names."""
        r = resolve("phi-4-mini")
        assert r.family == "phi-mini"
        assert r.quant == "4bit"

    def test_resolve_alias_with_variant(self) -> None:
        """resolve() works with aliased name + variant."""
        r = resolve("llama-3.2-1b:8bit")
        assert r.family == "llama-1b"
        assert r.quant == "8bit"


# =====================================================================
# Resolver tests
# =====================================================================


class TestResolve:
    """Tests for the resolve() function."""

    def test_bare_name_default_variant(self) -> None:
        """Bare name resolves to default quant."""
        r = resolve("gemma-4b")
        assert r.family == "gemma-4b"
        assert r.quant == "4bit"
        assert r.mlx_repo is not None

    def test_with_4bit_variant(self) -> None:
        """Explicit 4bit variant."""
        r = resolve("gemma-4b:4bit")
        assert r.quant == "4bit"
        assert r.mlx_repo == "mlx-community/REDACTED"

    def test_with_8bit_variant(self) -> None:
        """8bit variant resolves to 8bit artifacts."""
        r = resolve("gemma-4b:8bit")
        assert r.quant == "8bit"
        assert r.mlx_repo == "mlx-community/REDACTED-8bit"

    def test_with_fp16_variant(self) -> None:
        """fp16 variant resolves to source repo."""
        r = resolve("gemma-4b:fp16")
        assert r.quant == "fp16"
        assert r.source_repo == "REDACTED"

    def test_engine_specific_q4_k_m(self) -> None:
        """q4_k_m (GGUF-specific) normalizes to 4bit."""
        r = resolve("llama-8b:q4_k_m")
        assert r.quant == "4bit"

    def test_engine_specific_q8_0(self) -> None:
        """q8_0 (GGUF-specific) normalizes to 8bit."""
        r = resolve("llama-8b:q8_0")
        assert r.quant == "8bit"

    def test_mlx_engine_gets_mlx_repo(self) -> None:
        """Forcing mlx-lm engine returns MLX repo."""
        r = resolve("gemma-4b:4bit", engine="mlx-lm")
        assert r.engine == "mlx-lm"
        assert r.hf_repo == "mlx-community/REDACTED"

    def test_llamacpp_engine_gets_gguf(self) -> None:
        """Forcing llama.cpp engine returns GGUF repo+file."""
        r = resolve("gemma-4b:4bit", engine="llama.cpp")
        assert r.engine == "llama.cpp"
        assert r.is_gguf is True
        assert r.filename is not None
        assert r.filename.endswith(".gguf")

    def test_gguf_alias_for_llamacpp(self) -> None:
        """Engine alias 'gguf' maps to llama.cpp."""
        r = resolve("phi-mini", engine="gguf")
        assert r.engine == "llama.cpp"
        assert r.is_gguf is True

    def test_mlx_alias(self) -> None:
        """Engine alias 'mlx' maps to mlx-lm."""
        r = resolve("phi-mini", engine="mlx")
        assert r.engine == "mlx-lm"

    def test_hf_repo_passthrough(self) -> None:
        """Full HuggingFace repo passes through unchanged."""
        r = resolve("mlx-community/custom-model")
        assert r.family is None
        assert r.hf_repo == "mlx-community/custom-model"

    def test_local_file_passthrough(self) -> None:
        """Local .gguf file passes through."""
        r = resolve("model.gguf")
        assert r.hf_repo == "model.gguf"
        assert r.filename == "model.gguf"

    def test_unknown_model_raises(self) -> None:
        """Unknown model family raises ModelResolutionError."""
        with pytest.raises(ModelResolutionError, match="Unknown model 'nonexistent'"):
            resolve("nonexistent")

    def test_unknown_model_suggests(self) -> None:
        """Close match gets a suggestion."""
        with pytest.raises(ModelResolutionError, match="Did you mean"):
            resolve("gemma-4")  # close to gemma-4b

    def test_unknown_variant_raises(self) -> None:
        """Unknown variant on known model raises ModelResolutionError."""
        with pytest.raises(ModelResolutionError, match="Unknown variant"):
            resolve("gemma-4b:q2_k")

    def test_available_engines_picks_best(self) -> None:
        """With available_engines, picks highest priority one that has artifacts."""
        r = resolve("gemma-4b:4bit", available_engines=["llama.cpp", "echo"])
        assert r.engine == "llama.cpp"

    def test_available_engines_skips_unavailable(self) -> None:
        """If preferred engine is not in available_engines, picks next best."""
        r = resolve("gemma-4b:4bit", available_engines=["llama.cpp"])
        assert r.engine == "llama.cpp"

    def test_case_insensitive_name(self) -> None:
        """Model names are case-insensitive."""
        r = resolve("Gemma-4B:4bit")
        assert r.family == "gemma-4b"

    def test_case_insensitive_variant(self) -> None:
        """Variant names are case-insensitive."""
        r = resolve("gemma-4b:FP16")
        assert r.quant == "fp16"


class TestResolveAllModels:
    """Verify every catalog model resolves without errors."""

    def test_all_models_resolve_default(self) -> None:
        """Every model in the catalog should resolve with default variant."""
        for name in CATALOG:
            r = resolve(name)
            assert r.family == name
            assert r.hf_repo, f"{name} resolved to empty hf_repo"

    def test_all_models_resolve_all_variants(self) -> None:
        """Every variant of every model should resolve."""
        for name, entry in CATALOG.items():
            for quant in entry.variants:
                r = resolve(f"{name}:{quant}")
                assert r.family == name, f"Family mismatch for {name}:{quant}"
                assert r.quant == quant, f"Quant mismatch for {name}:{quant}"


# =====================================================================
# Backward compatibility tests
# =====================================================================


class TestBackwardCompat:
    """Verify backward compatibility with existing code."""

    def test_resolve_model_name_mlx(self) -> None:
        """resolve_model_name() works for mlx backend."""
        from octomil.serve import resolve_model_name

        result = resolve_model_name("gemma-4b", "mlx")
        assert "mlx-community" in result

    def test_resolve_model_name_mlx_with_variant(self) -> None:
        """resolve_model_name() with :8bit returns 8bit MLX repo."""
        from octomil.serve import resolve_model_name

        result = resolve_model_name("gemma-4b:8bit", "mlx")
        assert "8bit" in result.lower() or "8b" in result.lower()

    def test_resolve_model_name_gguf(self) -> None:
        """resolve_model_name() works for gguf backend."""
        from octomil.serve import resolve_model_name

        result = resolve_model_name("phi-mini", "gguf")
        assert result == "phi-mini"

    def test_resolve_model_name_repo_passthrough(self) -> None:
        """Full repo paths pass through."""
        from octomil.serve import resolve_model_name

        result = resolve_model_name("user/custom-model", "mlx")
        assert result == "user/custom-model"

    def test_resolve_model_name_gguf_file(self) -> None:
        """Local .gguf file passes through."""
        from octomil.serve import resolve_model_name

        result = resolve_model_name("model.gguf", "gguf")
        assert result == "model.gguf"

    def test_resolve_model_name_unknown_raises(self) -> None:
        """Unknown model raises ValueError."""
        from octomil.serve import resolve_model_name

        with pytest.raises(ValueError, match="Unknown model"):
            resolve_model_name("fake-model", "mlx")

    def test_data_classes_importable(self) -> None:
        """Legacy data classes are importable from octomil.models."""
        from octomil.models import (
            DeploymentPlan,
        )

        # Quick sanity check
        plan = DeploymentPlan(model_name="test", model_version="1.0")
        assert plan.model_name == "test"


# =====================================================================
# Server-side resolve fallback tests
# =====================================================================


class TestServerResolveFallback:
    """Tests for _resolve_via_server() — HTTP fallback when local catalog has no variants."""

    def test_server_resolve_success(self) -> None:
        """Server returns a valid resolved model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "family": "gemma-1b",
            "quant": "4bit",
            "engine": "mlx-lm",
            "hf_repo": "mlx-community/gemma-2-2b-it-4bit",
            "mlx_repo": "mlx-community/gemma-2-2b-it-4bit",
            "source_repo": "google/gemma-2-2b-it",
            "architecture": "dense",
        }

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            result = _resolve_via_server("gemma-1b", engine="mlx-lm")

        assert result is not None
        assert result.family == "gemma-1b"
        assert result.engine == "mlx-lm"
        assert result.hf_repo == "mlx-community/gemma-2-2b-it-4bit"
        assert result.mlx_repo == "mlx-community/gemma-2-2b-it-4bit"

    def test_server_resolve_404(self) -> None:
        """Server returns 404 for unknown model — returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            result = _resolve_via_server("nonexistent-model")

        assert result is None

    def test_server_resolve_network_error(self) -> None:
        """Network error returns None (graceful fallback)."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = ConnectionError("unreachable")

        with patch("httpx.Client", return_value=mock_client):
            result = _resolve_via_server("gemma-1b")

        assert result is None

    def test_resolve_falls_back_to_server_on_empty_variants(self) -> None:
        """resolve() calls the server when local catalog entry has empty variants."""
        from octomil.models import catalog as cat_mod
        from octomil.models.catalog import ModelEntry

        # Add a model with empty variants to simulate the scrubbed catalog
        cat_mod.CATALOG["test-empty"] = ModelEntry(
            publisher="Test",
            params="1B",
            default_quant="4bit",
            variants={},
            engines=frozenset(["mlx-lm"]),
        )

        mock_server_result = MagicMock()
        mock_server_result.family = "test-empty"
        mock_server_result.quant = "4bit"
        mock_server_result.engine = "mlx-lm"
        mock_server_result.hf_repo = "mlx-community/test-model-4bit"

        with patch("octomil.models.resolver._resolve_via_server", return_value=mock_server_result) as mock_fn:
            result = resolve("test-empty")

        mock_fn.assert_called_once_with("test-empty", engine="")
        assert result.hf_repo == "mlx-community/test-model-4bit"

        # Clean up
        del cat_mod.CATALOG["test-empty"]

    def test_resolve_raises_when_server_also_fails(self) -> None:
        """resolve() raises ModelResolutionError when both local and server fail."""
        from octomil.models import catalog as cat_mod
        from octomil.models.catalog import ModelEntry

        cat_mod.CATALOG["test-empty2"] = ModelEntry(
            publisher="Test",
            params="1B",
            default_quant="4bit",
            variants={},
            engines=frozenset(["mlx-lm"]),
        )

        with patch("octomil.models.resolver._resolve_via_server", return_value=None):
            with pytest.raises(ModelResolutionError, match="no downloadable variants"):
                resolve("test-empty2")

        # Clean up
        del cat_mod.CATALOG["test-empty2"]


# =====================================================================
# Ollama tag resolution tests
# =====================================================================


class TestOllamaTagResolution:
    """Tests for resolving Ollama-native tags (e.g. qwen2.5:3b)."""

    def test_resolve_ollama_tag_known(self) -> None:
        """resolve_ollama_tag finds a matching catalog entry."""
        result = resolve_ollama_tag("qwen2.5:3b")
        assert result is not None
        family, quant = result
        assert family == "qwen-3b"
        assert quant == "4bit"

    def test_resolve_ollama_tag_unknown(self) -> None:
        """resolve_ollama_tag returns None for unknown tags."""
        assert resolve_ollama_tag("nonexistent:9b") is None

    def test_resolve_ollama_tag_case_insensitive(self) -> None:
        """Ollama tags should match case-insensitively."""
        result = resolve_ollama_tag("Qwen2.5:3b")
        assert result is not None
        assert result[0] == "qwen-3b"

    def test_resolve_ollama_tag_gemma(self) -> None:
        """gemma3:4b Ollama tag resolves to gemma-4b."""
        result = resolve_ollama_tag("gemma3:4b")
        assert result is not None
        assert result[0] == "gemma-4b"

    def test_resolve_ollama_tag_llama(self) -> None:
        """llama3.2:3b Ollama tag resolves to llama-3b."""
        result = resolve_ollama_tag("llama3.2:3b")
        assert result is not None
        assert result[0] == "llama-3b"

    def test_resolve_via_resolver_ollama_tag(self) -> None:
        """resolve() should work with Ollama-native tags like qwen2.5:3b."""
        r = resolve("qwen2.5:3b")
        assert r.family == "qwen-3b"
        assert r.quant == "4bit"

    def test_resolve_via_resolver_ollama_tag_gemma(self) -> None:
        """resolve() with gemma3:1b Ollama tag."""
        r = resolve("gemma3:1b")
        assert r.family == "gemma-1b"
        assert r.quant == "4bit"

    def test_resolve_via_resolver_ollama_tag_llama(self) -> None:
        """resolve() with llama3.1:8b Ollama tag."""
        r = resolve("llama3.1:8b")
        assert r.family == "llama-8b"
        assert r.quant == "4bit"

    def test_resolve_via_resolver_ollama_tag_mistral(self) -> None:
        """resolve() with mistral:7b Ollama tag."""
        r = resolve("mistral:7b")
        assert r.family == "mistral-7b"
        assert r.quant == "4bit"
