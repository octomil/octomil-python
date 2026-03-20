"""Tests for multimodal model support.

Covers:
- Modality enum values and membership
- ModelEntry with input/output modalities (required fields)
- ResourceBindingSpec construction and kind mapping
- AppModelEntry with resource bindings and engine_config
- ResolvedModel.is_multimodal property
- Embedded catalog: VL models have correct input_modalities
- Embedded catalog: audio models have correct input_modalities
- Package parsing extracts input_modalities/output_modalities/engine_config
- task_taxonomy replaces legacy modalities field
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from octomil._generated.artifact_resource_kind import ArtifactResourceKind
from octomil._generated.delivery_mode import DeliveryMode
from octomil._generated.modality import Modality
from octomil._generated.model_capability import ModelCapability
from octomil.manifest.types import AppModelEntry, ResourceBinding
from octomil.models.catalog import (
    CATALOG,
    ModelEntry,
    ResourceBindingSpec,
    _build_resource_bindings,
    _hydrate_manifest,
    _manifest_model_to_entry,
    _parse_modalities,
)
from octomil.models.resolver import ResolvedModel, resolve

_TEXT = [Modality.TEXT]
_IMAGE = [Modality.IMAGE]
_AUDIO = [Modality.AUDIO]
_TEXT_IMAGE = [Modality.IMAGE, Modality.TEXT]


# =====================================================================
# Modality enum tests
# =====================================================================


class TestModalityEnum:
    def test_text_value(self) -> None:
        assert Modality.TEXT.value == "text"

    def test_image_value(self) -> None:
        assert Modality.IMAGE.value == "image"

    def test_audio_value(self) -> None:
        assert Modality.AUDIO.value == "audio"

    def test_video_value(self) -> None:
        assert Modality.VIDEO.value == "video"

    def test_modality_is_str_enum(self) -> None:
        assert isinstance(Modality.TEXT, str)
        assert Modality.TEXT == "text"

    def test_all_values(self) -> None:
        values = {m.value for m in Modality}
        assert values == {"text", "image", "audio", "video"}


# =====================================================================
# _parse_modalities tests
# =====================================================================


class TestParseModalities:
    def test_none_defaults_to_text(self) -> None:
        result = _parse_modalities(None)
        assert result == [Modality.TEXT]

    def test_empty_list_defaults_to_text(self) -> None:
        result = _parse_modalities([])
        assert result == [Modality.TEXT]

    def test_single_text(self) -> None:
        result = _parse_modalities(["text"])
        assert result == [Modality.TEXT]

    def test_text_and_image(self) -> None:
        result = _parse_modalities(["text", "image"])
        assert Modality.TEXT in result
        assert Modality.IMAGE in result

    def test_audio_only(self) -> None:
        result = _parse_modalities(["audio"])
        assert result == [Modality.AUDIO]

    def test_case_insensitive(self) -> None:
        result = _parse_modalities(["TEXT", "Image"])
        assert Modality.TEXT in result
        assert Modality.IMAGE in result

    def test_unknown_values_ignored(self) -> None:
        result = _parse_modalities(["text", "unknown_modality"])
        assert result == [Modality.TEXT]

    def test_all_unknown_defaults_to_text(self) -> None:
        result = _parse_modalities(["garbage", "nonsense"])
        assert result == [Modality.TEXT]


# =====================================================================
# ResourceBindingSpec tests
# =====================================================================


class TestResourceBindingSpec:
    def test_create_weights_binding(self) -> None:
        binding = ResourceBindingSpec(
            kind=ArtifactResourceKind.WEIGHTS,
            uri="hf://org/repo/model.gguf",
            path="model.gguf",
            size_bytes=1024,
            required=True,
        )
        assert binding.kind == ArtifactResourceKind.WEIGHTS
        assert binding.uri == "hf://org/repo/model.gguf"
        assert binding.path == "model.gguf"
        assert binding.size_bytes == 1024

    def test_create_projector_binding(self) -> None:
        binding = ResourceBindingSpec(
            kind=ArtifactResourceKind.PROJECTOR,
            uri="hf://org/repo/mmproj.gguf",
            path="mmproj.gguf",
        )
        assert binding.kind == ArtifactResourceKind.PROJECTOR

    def test_create_tokenizer_binding(self) -> None:
        binding = ResourceBindingSpec(
            kind=ArtifactResourceKind.TOKENIZER,
            uri="hf://org/repo/tokenizer.json",
            path="tokenizer.json",
        )
        assert binding.kind == ArtifactResourceKind.TOKENIZER


class TestBuildResourceBindings:
    def test_empty_package(self) -> None:
        pkg: dict = {"resources": []}
        bindings = _build_resource_bindings(pkg)
        assert bindings == []

    def test_weights_only(self) -> None:
        pkg: dict = {
            "resources": [
                {
                    "kind": "weights",
                    "uri": "hf://org/repo/model.gguf",
                    "path": "model.gguf",
                    "size_bytes": 2048,
                    "required": True,
                }
            ]
        }
        bindings = _build_resource_bindings(pkg)
        assert len(bindings) == 1
        assert bindings[0].kind == ArtifactResourceKind.WEIGHTS
        assert bindings[0].size_bytes == 2048

    def test_multiple_resources(self) -> None:
        pkg: dict = {
            "resources": [
                {"kind": "weights", "uri": "hf://org/repo/model.gguf", "path": "model.gguf"},
                {"kind": "projector", "uri": "hf://org/repo/mmproj.gguf", "path": "mmproj.gguf"},
                {"kind": "tokenizer", "uri": "hf://org/repo/tokenizer.json", "path": "tokenizer.json"},
            ]
        }
        bindings = _build_resource_bindings(pkg)
        assert len(bindings) == 3
        kinds = {b.kind for b in bindings}
        assert ArtifactResourceKind.WEIGHTS in kinds
        assert ArtifactResourceKind.PROJECTOR in kinds
        assert ArtifactResourceKind.TOKENIZER in kinds

    def test_unknown_kind_skipped(self) -> None:
        pkg: dict = {
            "resources": [
                {"kind": "weights", "uri": "hf://org/repo/model.gguf"},
                {"kind": "totally_unknown_kind", "uri": "hf://org/repo/unknown"},
            ]
        }
        bindings = _build_resource_bindings(pkg)
        assert len(bindings) == 1
        assert bindings[0].kind == ArtifactResourceKind.WEIGHTS


# =====================================================================
# AppModelEntry with modality fields
# =====================================================================


class TestAppModelEntryMultimodal:
    def test_text_model_entry(self) -> None:
        entry = AppModelEntry(
            id="phi-4-mini",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.MANAGED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        assert entry.input_modalities == [Modality.TEXT]
        assert entry.output_modalities == [Modality.TEXT]

    def test_vl_model_entry(self) -> None:
        entry = AppModelEntry(
            id="gemma3-4b",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.MANAGED,
            input_modalities=[Modality.TEXT, Modality.IMAGE],
            output_modalities=[Modality.TEXT],
        )
        assert Modality.TEXT in entry.input_modalities
        assert Modality.IMAGE in entry.input_modalities
        assert entry.output_modalities == [Modality.TEXT]

    def test_audio_model_entry(self) -> None:
        entry = AppModelEntry(
            id="whisper-base",
            capability=ModelCapability.TRANSCRIPTION,
            delivery=DeliveryMode.MANAGED,
            input_modalities=[Modality.AUDIO],
            output_modalities=[Modality.TEXT],
        )
        assert entry.input_modalities == [Modality.AUDIO]
        assert entry.output_modalities == [Modality.TEXT]

    def test_resource_bindings(self) -> None:
        entry = AppModelEntry(
            id="test-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.MANAGED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            resource_bindings=[
                ResourceBinding(
                    kind=ArtifactResourceKind.WEIGHTS,
                    uri="hf://org/repo/model.gguf",
                    path="model.gguf",
                ),
                ResourceBinding(
                    kind=ArtifactResourceKind.TOKENIZER,
                    uri="hf://org/repo/tokenizer.json",
                    path="tokenizer.json",
                ),
            ],
        )
        assert len(entry.resource_bindings) == 2
        weights = entry.binding_for(ArtifactResourceKind.WEIGHTS)
        assert weights is not None
        assert weights.uri == "hf://org/repo/model.gguf"

    def test_binding_for_missing(self) -> None:
        entry = AppModelEntry(
            id="test-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.MANAGED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
        )
        assert entry.binding_for(ArtifactResourceKind.WEIGHTS) is None

    def test_bindings_for_multiple(self) -> None:
        entry = AppModelEntry(
            id="test-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.MANAGED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            resource_bindings=[
                ResourceBinding(kind=ArtifactResourceKind.WEIGHTS, uri="a"),
                ResourceBinding(kind=ArtifactResourceKind.WEIGHTS, uri="b"),
                ResourceBinding(kind=ArtifactResourceKind.TOKENIZER, uri="c"),
            ],
        )
        weights_bindings = entry.bindings_for(ArtifactResourceKind.WEIGHTS)
        assert len(weights_bindings) == 2

    def test_engine_config(self) -> None:
        entry = AppModelEntry(
            id="test-model",
            capability=ModelCapability.CHAT,
            delivery=DeliveryMode.MANAGED,
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            engine_config={"n_gpu_layers": "32", "ctx_size": "4096"},
        )
        assert entry.engine_config["n_gpu_layers"] == "32"
        assert entry.engine_config["ctx_size"] == "4096"


# =====================================================================
# ModelEntry with modality fields
# =====================================================================


class TestModelEntryMultimodal:
    def test_text_only_model(self) -> None:
        entry = ModelEntry(
            publisher="Test",
            params="1B",
            input_modalities=[Modality.TEXT],
            output_modalities=[Modality.TEXT],
        )
        assert entry.input_modalities == [Modality.TEXT]
        assert entry.output_modalities == [Modality.TEXT]

    def test_vl_model(self) -> None:
        entry = ModelEntry(
            publisher="Google",
            params="4B",
            input_modalities=[Modality.IMAGE, Modality.TEXT],
            output_modalities=[Modality.TEXT],
            task_taxonomy=["text", "image"],
        )
        assert Modality.IMAGE in entry.input_modalities
        assert Modality.TEXT in entry.input_modalities
        assert entry.task_taxonomy == ["text", "image"]

    def test_model_entry_requires_modalities(self) -> None:
        """ModelEntry requires input_modalities and output_modalities."""
        with pytest.raises(TypeError, match="input_modalities"):
            ModelEntry(publisher="Test", params="1B")  # type: ignore[call-arg]


# =====================================================================
# ResolvedModel multimodal tests
# =====================================================================


class TestResolvedModelMultimodal:
    def test_text_model_not_multimodal(self) -> None:
        r = ResolvedModel(
            family="phi-mini",
            quant="4bit",
            engine="mlx-lm",
            hf_repo="mlx-community/phi-mini",
            input_modalities=[Modality.TEXT],
            output_modalities=[Modality.TEXT],
        )
        assert r.is_multimodal is False

    def test_vl_model_is_multimodal(self) -> None:
        r = ResolvedModel(
            family="gemma3-4b",
            quant="4bit",
            engine="llama.cpp",
            hf_repo="org/gemma3-4b-gguf",
            input_modalities=[Modality.TEXT, Modality.IMAGE],
            output_modalities=[Modality.TEXT],
        )
        assert r.is_multimodal is True

    def test_audio_model_is_multimodal(self) -> None:
        r = ResolvedModel(
            family="whisper-base",
            quant="fp16",
            engine="whisper.cpp",
            hf_repo="ggerganov/whisper.cpp",
            input_modalities=[Modality.AUDIO],
            output_modalities=[Modality.TEXT],
        )
        assert r.is_multimodal is True

    def test_resolved_has_engine_config(self) -> None:
        r = ResolvedModel(
            family="test",
            quant="4bit",
            engine="llama.cpp",
            hf_repo="test/repo",
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            engine_config={"n_gpu_layers": "32"},
        )
        assert r.engine_config["n_gpu_layers"] == "32"

    def test_resolved_has_resource_bindings(self) -> None:
        bindings = [
            ResourceBindingSpec(
                kind=ArtifactResourceKind.WEIGHTS,
                uri="hf://org/repo/model.gguf",
                path="model.gguf",
            )
        ]
        r = ResolvedModel(
            family="test",
            quant="4bit",
            engine="llama.cpp",
            hf_repo="test/repo",
            input_modalities=_TEXT,
            output_modalities=_TEXT,
            resource_bindings=bindings,
        )
        assert len(r.resource_bindings) == 1
        assert r.resource_bindings[0].kind == ArtifactResourceKind.WEIGHTS


# =====================================================================
# Manifest parsing multimodal tests
# =====================================================================


class TestManifestParsingMultimodal:
    """Test that _manifest_model_to_entry extracts modality fields from packages."""

    def test_text_model_parsing(self) -> None:
        model_dict: dict = {
            "id": "test-text",
            "family": "test",
            "parameter_count": "1B",
            "default_quantization": "q4_k_m",
            "task_taxonomy": ["text-generation"],
            "packages": [
                {
                    "runtime_executor": "llamacpp",
                    "artifact_format": "gguf",
                    "quantization": "q4_k_m",
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                    "resources": [
                        {"kind": "weights", "uri": "hf://org/model.gguf", "path": "model.gguf"},
                    ],
                },
            ],
        }
        key, entry = _manifest_model_to_entry(model_dict)
        assert key == "test-text"
        assert Modality.TEXT in entry.input_modalities
        assert Modality.TEXT in entry.output_modalities
        assert entry.task_taxonomy == ["text-generation"]

    def test_vl_model_parsing(self) -> None:
        model_dict: dict = {
            "id": "test-vl",
            "family": "test",
            "parameter_count": "4B",
            "default_quantization": "q4_k_m",
            "task_taxonomy": ["multimodal"],
            "packages": [
                {
                    "runtime_executor": "llamacpp",
                    "artifact_format": "gguf",
                    "quantization": "q4_k_m",
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "engine_config": {"n_gpu_layers": "32"},
                    "resources": [
                        {"kind": "weights", "uri": "hf://org/model.gguf", "path": "model.gguf"},
                        {"kind": "projector", "uri": "hf://org/mmproj.gguf", "path": "mmproj.gguf"},
                    ],
                },
            ],
        }
        key, entry = _manifest_model_to_entry(model_dict)
        assert key == "test-vl"
        assert Modality.TEXT in entry.input_modalities
        assert Modality.IMAGE in entry.input_modalities
        assert entry.output_modalities == [Modality.TEXT]
        assert entry.engine_config["n_gpu_layers"] == "32"
        assert entry.task_taxonomy == ["multimodal"]

        # Should have resource bindings including projector
        kinds = {b.kind for b in entry.resource_bindings}
        assert ArtifactResourceKind.WEIGHTS in kinds
        assert ArtifactResourceKind.PROJECTOR in kinds

    def test_engine_config_merged_across_packages(self) -> None:
        model_dict: dict = {
            "id": "test-merge",
            "family": "test",
            "parameter_count": "1B",
            "default_quantization": "q4_k_m",
            "packages": [
                {
                    "runtime_executor": "llamacpp",
                    "artifact_format": "gguf",
                    "quantization": "q4_k_m",
                    "engine_config": {"n_gpu_layers": "32"},
                    "resources": [{"kind": "weights", "uri": "hf://org/a.gguf"}],
                },
                {
                    "runtime_executor": "mlx",
                    "artifact_format": "mlx",
                    "quantization": "q4_k_m",
                    "engine_config": {"use_mmap": "true", "n_gpu_layers": "64"},
                    "resources": [{"kind": "weights", "uri": "hf://org/b"}],
                },
            ],
        }
        _, entry = _manifest_model_to_entry(model_dict)
        # First value wins per key
        assert entry.engine_config["n_gpu_layers"] == "32"
        assert entry.engine_config["use_mmap"] == "true"


# =====================================================================
# Hydrate manifest with task_taxonomy
# =====================================================================


class TestHydrateManifestTaskTaxonomy:
    """Test that _hydrate_manifest correctly reads task_taxonomy and modalities from packages."""

    def test_task_taxonomy_from_family(self) -> None:
        manifest: dict = {
            "test-family": {
                "id": "test-family",
                "vendor": "Test",
                "task_taxonomy": ["text-generation"],
                "variants": {
                    "test-1b": {
                        "id": "test-1b",
                        "parameter_count": "1B",
                        "quantizations": ["Q4_K_M"],
                        "versions": {
                            "1.0.0": {
                                "packages": [
                                    {
                                        "runtime_executor": "llamacpp",
                                        "artifact_format": "gguf",
                                        "quantization": "Q4_K_M",
                                        "input_modalities": ["text"],
                                        "output_modalities": ["text"],
                                        "resources": [
                                            {"kind": "weights", "uri": "hf://org/model.gguf", "path": "model.gguf"}
                                        ],
                                    }
                                ]
                            }
                        },
                    }
                },
            }
        }
        catalog = _hydrate_manifest(manifest)
        assert "test-1b" in catalog
        entry = catalog["test-1b"]
        assert entry.task_taxonomy == ["text-generation"]
        assert entry.input_modalities == [Modality.TEXT]

    def test_legacy_modalities_fallback(self) -> None:
        """Old manifests with 'modalities' instead of 'task_taxonomy' still work."""
        manifest: dict = {
            "old-family": {
                "id": "old-family",
                "vendor": "Old",
                "modalities": ["text"],
                "variants": {
                    "old-1b": {
                        "id": "old-1b",
                        "parameter_count": "1B",
                        "modalities": ["text"],
                        "quantizations": ["Q4_K_M"],
                        "versions": {
                            "1.0.0": {
                                "packages": [
                                    {
                                        "runtime_executor": "llamacpp",
                                        "artifact_format": "gguf",
                                        "quantization": "Q4_K_M",
                                        "input_modalities": ["text"],
                                        "output_modalities": ["text"],
                                        "resources": [
                                            {"kind": "weights", "uri": "hf://org/model.gguf", "path": "model.gguf"}
                                        ],
                                    }
                                ]
                            }
                        },
                    }
                },
            }
        }
        catalog = _hydrate_manifest(manifest)
        assert "old-1b" in catalog
        entry = catalog["old-1b"]
        # Legacy "modalities" should be read as task_taxonomy
        assert entry.task_taxonomy == ["text"]


# =====================================================================
# Resolve multimodal models
# =====================================================================


class TestResolveMultimodal:
    """Test that resolve() propagates modality info through resolution."""

    def test_passthrough_defaults_to_text(self) -> None:
        r = resolve("user/custom-model")
        assert r.input_modalities == [Modality.TEXT]
        assert r.output_modalities == [Modality.TEXT]
        assert r.is_multimodal is False

    def test_local_file_defaults_to_text(self) -> None:
        r = resolve("model.gguf")
        assert r.input_modalities == [Modality.TEXT]
        assert r.is_multimodal is False

    def test_all_resolved_models_have_modalities(self) -> None:
        """Every model in CATALOG should resolve with valid modalities."""
        for name in CATALOG:
            r = resolve(name)
            assert len(r.input_modalities) > 0, f"{name} resolved with no input modalities"
            assert len(r.output_modalities) > 0, f"{name} resolved with no output modalities"


# =====================================================================
# ArtifactResourceKind enum
# =====================================================================


class TestArtifactResourceKind:
    """Verify ArtifactResourceKind matches contract enum exactly."""

    EXPECTED_KINDS = {
        "weights",
        "tokenizer",
        "tokenizer_config",
        "model_config",
        "generation_config",
        "processor",
        "vocab",
        "merges",
        "adapter",
        "projector",
        "manifest",
        "signature",
        "metadata",
    }

    def test_all_expected_kinds_present(self) -> None:
        actual = {k.value for k in ArtifactResourceKind}
        assert actual == self.EXPECTED_KINDS

    def test_projector_is_multimodal(self) -> None:
        """Projector kind must exist for VL model mmproj weights."""
        assert ArtifactResourceKind.PROJECTOR.value == "projector"

    def test_processor_exists(self) -> None:
        """Processor kind for multi-modal model processor configs."""
        assert ArtifactResourceKind.PROCESSOR.value == "processor"
