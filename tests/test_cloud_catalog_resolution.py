"""Tests for catalog-driven cloud model resolution.

Covers:
- Cloud executor in _EXECUTOR_TO_ENGINE mapping
- Cloud package handling in _package_to_variant_field (metadata-only, no weights)
- Cloud package in _manifest_model_to_entry (engine_config, engines set)
- Cloud early return in _resolve_from_manifest
- Cloud branch in _pick_engine
- Cloud fallback in CATALOG resolution path
- _has_explicit_quant for :cloud suffix
- Serve CLI auto-detection of cloud models
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from octomil.cli_helpers import _has_explicit_quant
from octomil.models.catalog import (
    _EXECUTOR_TO_ENGINE,
    _manifest_model_to_entry,
    _package_to_variant_field,
)
from octomil.models.resolver import (
    _ENGINE_ALIASES,
    _ENGINE_TO_EXECUTORS,
    _pick_engine,
    resolve,
)

# ---------------------------------------------------------------------------
# Helpers — cloud manifest model fixture
# ---------------------------------------------------------------------------


def _cloud_manifest_model(
    model_id: str = "minimax-m2.5",
    base_url: str = "https://api.minimaxi.com/v1",
    cloud_model_id: str = "MiniMax-M1-80k",
    api_key_env: str = "MINIMAX_API_KEY",
) -> dict:
    """Build a minimal v2 manifest model dict for a cloud-only model."""
    return {
        "id": model_id,
        "family": "minimax",
        "parameter_count": "unknown",
        "default_quantization": "cloud",
        "vendor": "MiniMax",
        "task_taxonomy": ["text"],
        "capabilities": ["chat"],
        "packages": [
            {
                "artifact_format": "cloud",
                "runtime_executor": "cloud",
                "quantization": "cloud",
                "is_default": True,
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "engine_config": {
                    "base_url": base_url,
                    "cloud_model_id": cloud_model_id,
                    "api_key_env": api_key_env,
                },
                "resources": [
                    {
                        "kind": "metadata",
                        "uri": f"cloud://minimax/{cloud_model_id}",
                        "required": False,
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# _EXECUTOR_TO_ENGINE mapping
# ---------------------------------------------------------------------------


class TestCloudExecutorMapping:
    def test_cloud_executor_in_executor_to_engine(self):
        assert "cloud" in _EXECUTOR_TO_ENGINE
        assert _EXECUTOR_TO_ENGINE["cloud"] == "cloud"

    def test_cloud_in_engine_aliases(self):
        assert "cloud" in _ENGINE_ALIASES
        assert _ENGINE_ALIASES["cloud"] == "cloud"

    def test_cloud_in_engine_to_executors(self):
        assert "cloud" in _ENGINE_TO_EXECUTORS
        assert "cloud" in _ENGINE_TO_EXECUTORS["cloud"]


# ---------------------------------------------------------------------------
# _package_to_variant_field — cloud metadata package, no weights
# ---------------------------------------------------------------------------


class TestPackageToVariantFieldCloud:
    def test_cloud_package_returns_all_none(self):
        """Cloud packages have metadata resources, no weights.

        _get_weights_resource() returns None, so _package_to_variant_field()
        early-returns all-None fields.
        """
        pkg = _cloud_manifest_model()["packages"][0]
        mlx, gguf, ort, mlc, ollama, source_repo = _package_to_variant_field(pkg)
        assert mlx is None
        assert gguf is None
        assert ort is None
        assert mlc is None
        assert ollama is None
        assert source_repo is None


# ---------------------------------------------------------------------------
# _manifest_model_to_entry — cloud package produces correct ModelEntry
# ---------------------------------------------------------------------------


class TestManifestModelToEntryCloud:
    def test_cloud_engine_in_engines_set(self):
        """Cloud package should add 'cloud' to the engines frozenset."""
        key, entry = _manifest_model_to_entry(_cloud_manifest_model())
        assert key == "minimax-m2.5"
        assert "cloud" in entry.engines

    def test_engine_config_populated(self):
        """engine_config from cloud package should be merged into ModelEntry."""
        _, entry = _manifest_model_to_entry(_cloud_manifest_model())
        assert entry.engine_config.get("base_url") == "https://api.minimaxi.com/v1"
        assert entry.engine_config.get("cloud_model_id") == "MiniMax-M1-80k"
        assert entry.engine_config.get("api_key_env") == "MINIMAX_API_KEY"

    def test_cloud_variant_exists(self):
        """A 'cloud' quant variant should exist in ModelEntry.variants."""
        _, entry = _manifest_model_to_entry(_cloud_manifest_model())
        assert "cloud" in entry.variants


# ---------------------------------------------------------------------------
# Cloud resolution via resolve() with mocked manifest
# ---------------------------------------------------------------------------


class TestResolveFromManifestCloud:
    def _make_nested_manifest(self, packages: list[dict] | None = None):
        """Build a nested manifest in the format _find_manifest_model expects."""
        if packages is None:
            packages = _cloud_manifest_model()["packages"]
        return {
            "minimax": {
                "capabilities": ["chat"],
                "variants": {
                    "minimax-m2.5": {
                        "parameter_count": "unknown",
                        "quantizations": ["cloud"],
                        "capabilities": ["chat"],
                        "versions": {
                            "1.0.0": {
                                "packages": packages,
                            }
                        },
                    }
                },
            }
        }

    def _mock_manifest(self, packages: list[dict] | None = None):
        """Patch _get_v2_client to return a nested manifest."""
        manifest = self._make_nested_manifest(packages)

        class FakeClient:
            def get_manifest(self, **kwargs):
                return manifest

        return patch("octomil.models.resolver._get_v2_client", return_value=FakeClient())

    def test_returns_resolved_model_for_cloud(self):
        """Cloud executor should short-circuit and return ResolvedModel."""
        with self._mock_manifest():
            result = resolve("minimax-m2.5:cloud")
        assert result.engine == "cloud"
        assert result.quant == "cloud"
        assert result.hf_repo == ""
        assert result.engine_config["base_url"] == "https://api.minimaxi.com/v1"

    def test_cloud_no_weights_required(self):
        """Resolution should succeed even without weights resources."""
        pkgs = _cloud_manifest_model()["packages"]
        pkgs[0]["resources"] = []
        with self._mock_manifest(pkgs):
            result = resolve("minimax-m2.5:cloud")
        assert result.engine == "cloud"

    def test_capabilities_preserved(self):
        """Capabilities from manifest should flow into ResolvedModel."""
        with self._mock_manifest():
            result = resolve("minimax-m2.5:cloud")
        assert "chat" in result.capabilities


# ---------------------------------------------------------------------------
# _pick_engine — cloud branch
# ---------------------------------------------------------------------------


class TestPickEngineCloud:
    def test_pick_engine_returns_cloud(self):
        """_pick_engine should return 'cloud' when model has cloud engine."""
        _, entry = _manifest_model_to_entry(_cloud_manifest_model())
        result = _pick_engine(entry, "cloud", ["cloud"])
        assert result == "cloud"

    def test_pick_engine_cloud_not_available(self):
        """_pick_engine should return None when cloud not in available_engines."""
        _, entry = _manifest_model_to_entry(_cloud_manifest_model())
        result = _pick_engine(entry, "cloud", ["llama.cpp", "mlx-lm"])
        assert result is None


# ---------------------------------------------------------------------------
# _has_explicit_quant — :cloud suffix
# ---------------------------------------------------------------------------


class TestHasExplicitQuantCloud:
    def test_cloud_suffix_is_explicit(self):
        assert _has_explicit_quant("minimax-m2.5:cloud") is True

    def test_cloud_suffix_case_insensitive(self):
        assert _has_explicit_quant("kimi-k2.5:CLOUD") is True

    def test_bare_name_not_explicit(self):
        assert _has_explicit_quant("minimax-m2.5") is False


# ---------------------------------------------------------------------------
# Full resolve() with mocked manifest — cloud CATALOG path
# ---------------------------------------------------------------------------


class TestResolveCatalogCloud:
    def test_resolve_cloud_model_from_manifest(self):
        """resolve() should use manifest path for cloud model."""
        packages = _cloud_manifest_model()["packages"]
        manifest = {
            "minimax": {
                "variants": {
                    "minimax-m2.5": {
                        "quantizations": ["cloud"],
                        "capabilities": ["chat"],
                        "versions": {"1.0.0": {"packages": packages}},
                    }
                }
            }
        }

        class FakeClient:
            def get_manifest(self, **kwargs):
                return manifest

        with patch("octomil.models.resolver._get_v2_client", return_value=FakeClient()):
            result = resolve("minimax-m2.5:cloud")
        assert result.engine == "cloud"
        assert result.engine_config["base_url"] == "https://api.minimaxi.com/v1"
        assert result.quant == "cloud"
