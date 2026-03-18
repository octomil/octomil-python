"""Unified model catalog — single source of truth for all engines.

Model entries map quant variants to engine-specific artifacts:

- ``mlx``: HuggingFace MLX repo ID
- ``gguf``: Tuple of (HF repo, filename)
- ``source``: Original model HF repo (for engines that download and convert)

Powered by the v2 manifest endpoint via
:class:`~octomil.models.catalog_client.CatalogClientV2`. Manifest
models and packages are converted to ``ModelEntry`` / ``VariantSpec``
dataclasses consumed by the engine layer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from octomil._generated.artifact_resource_kind import ArtifactResourceKind
from octomil._generated.modality import Modality

from .catalog_client import CatalogClientV2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes — unchanged public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GGUFSource:
    """GGUF model source — HuggingFace repo + optional filename."""

    repo: str
    filename: str  # empty for repo-level URIs
    revision: Optional[str] = None
    quantization_hint: Optional[str] = None
    uri_type: str = "file"  # "file" or "repo"
    projector_repo: Optional[str] = None  # HF repo for mmproj (same repo if None)
    projector_filename: Optional[str] = None  # mmproj GGUF filename


@dataclass(frozen=True)
class VariantSpec:
    """Artifact locations for a single quantization variant across engines."""

    mlx: Optional[str] = None
    gguf: Optional[GGUFSource] = None
    ort: Optional[str] = None  # ONNX Runtime model repo ID
    mlc: Optional[str] = None  # MLC-LLM pre-compiled model repo ID
    ollama: Optional[str] = None  # Ollama model tag (e.g. "gemma3:1b")
    source_repo: Optional[str] = None  # original (fp16/bf16) repo


@dataclass(frozen=True)
class ResourceBindingSpec:
    """A resolved resource within a package, mapping kind to URI/path."""

    kind: ArtifactResourceKind
    uri: str
    path: str = ""
    size_bytes: int = 0
    checksum_sha256: str = ""
    required: bool = True
    load_order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)  # type: ignore[assignment]


@dataclass(frozen=True)
class MoEMetadata:
    """Mixture of Experts model metadata.

    Captures the sparse activation pattern that defines MoE models:
    only ``active_experts`` out of ``num_experts`` are activated per token,
    meaning RAM usage is closer to ``active_params`` than ``total_params``.
    """

    num_experts: int
    active_experts: int
    expert_size: str  # human-readable, e.g. "7B"
    total_params: str  # total parameter count, e.g. "46.7B"
    active_params: str  # params active per token, e.g. "12.9B"


@dataclass
class ModelEntry:
    """A model family with its per-engine, per-quant variants."""

    publisher: str
    params: str
    input_modalities: list[Modality]
    output_modalities: list[Modality]
    default_quant: str = "4bit"
    variants: dict[str, VariantSpec] = field(default_factory=dict)
    engines: frozenset[str] = frozenset()  # engines this model is known to work on
    architecture: str = "dense"  # "dense" or "moe"
    moe: Optional[MoEMetadata] = None  # populated when architecture == "moe"
    download_size: Optional[str] = None  # human-readable, e.g. "1.7 GB"
    task_taxonomy: list[str] = field(default_factory=list)
    engine_config: dict[str, Any] = field(default_factory=dict)
    resource_bindings: list[ResourceBindingSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manifest-to-legacy conversion helpers
# ---------------------------------------------------------------------------

# Maps v2 manifest quantization names to the canonical quant labels
# used in the v1 catalog (4bit, 8bit, fp16).
_QUANT_TO_CANONICAL: dict[str, str] = {
    "q4_k_m": "4bit",
    "q4_k_s": "4bit",
    "q4_0": "4bit",
    "4bit": "4bit",
    "q8_0": "8bit",
    "q8_1": "8bit",
    "8bit": "8bit",
    "fp16": "fp16",
    "f16": "fp16",
    "float16": "fp16",
}

# Maps v2 runtime_executor names to the canonical engine names
# used by the resolver and engine registry.
_EXECUTOR_TO_ENGINE: dict[str, str] = {
    "llamacpp": "llama.cpp",
    "llama.cpp": "llama.cpp",
    "mlx": "mlx-lm",
    "mlx-lm": "mlx-lm",
    "whisper": "whisper.cpp",
    "whisper.cpp": "whisper.cpp",
    "onnxruntime": "onnxruntime",
    "ort": "onnxruntime",
    "mlc": "mlc-llm",
    "mlc-llm": "mlc-llm",
    "ollama": "ollama",
    "echo": "echo",
    "mnn": "mnn",
    "executorch": "executorch",
    "cactus": "cactus",
}

# Maps v2 manifest family names to publisher names for ModelEntry.
# Families not listed here will get "Unknown" as publisher.
_FAMILY_TO_PUBLISHER: dict[str, str] = {
    "gemma": "Google",
    "gemma-2": "Google",
    "gemma-3": "Google",
    "qwen": "Alibaba",
    "qwen2": "Alibaba",
    "qwen2.5": "Alibaba",
    "llama": "Meta",
    "llama-2": "Meta",
    "llama-3": "Meta",
    "llama-3.1": "Meta",
    "llama-3.2": "Meta",
    "phi": "Microsoft",
    "phi-2": "Microsoft",
    "phi-3": "Microsoft",
    "phi-4": "Microsoft",
    "whisper": "OpenAI",
    "mistral": "Mistral AI",
    "mixtral": "Mistral AI",
    "deepseek": "DeepSeek",
    "deepseek-r1": "DeepSeek",
    "starcoder": "BigCode",
    "starcoder2": "BigCode",
    "smolvlm2": "HuggingFace",
    "ultravox": "Fixie AI",
}


def _parse_hf_uri(uri: str) -> tuple[str, str]:
    """Parse a ``hf://org/repo/filename`` URI into (repo, filename).

    Returns (full_path_without_prefix, "") if no filename part is found.

    Examples::

        >>> _parse_hf_uri("hf://bartowski/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf")
        ("bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf")

        >>> _parse_hf_uri("hf://mlx-community/gemma-2-2b-it-4bit")
        ("mlx-community/gemma-2-2b-it-4bit", "")
    """
    path = uri
    if path.startswith("hf://"):
        path = path[len("hf://") :]

    # Split into org/repo and optional filename
    # Pattern: org/repo/file.ext or org/repo
    parts = path.split("/", 2)
    if len(parts) == 3:
        repo = f"{parts[0]}/{parts[1]}"
        filename = parts[2]
        return repo, filename
    if len(parts) == 2:
        return path, ""
    return path, ""


def _get_resource_by_kind(pkg: dict, kind: str) -> Optional[dict]:
    """Extract the first resource of a given kind from a package."""
    for res in pkg.get("resources", []):
        if res.get("kind") == kind:
            return res
    return None


def _get_weights_resource(pkg: dict) -> Optional[dict]:
    """Extract the first weights resource from a package."""
    return _get_resource_by_kind(pkg, "weights")


def _get_projector_resource(pkg: dict) -> Optional[dict]:
    """Extract the first projector resource from a package."""
    return _get_resource_by_kind(pkg, "projector")


def _parse_modalities(raw: list[str] | None) -> list[Modality]:
    """Parse a list of raw modality strings into Modality enum values."""
    if not raw:
        return [Modality.TEXT]
    result: list[Modality] = []
    for m in raw:
        try:
            result.append(Modality(m.lower()))
        except ValueError:
            pass
    return result or [Modality.TEXT]


def _build_resource_bindings(pkg: dict) -> list[ResourceBindingSpec]:
    """Build ResourceBindingSpec list from package resources."""
    bindings: list[ResourceBindingSpec] = []
    for res in pkg.get("resources", []):
        kind_str = res.get("kind", "")
        try:
            kind = ArtifactResourceKind(kind_str)
        except ValueError:
            continue
        bindings.append(
            ResourceBindingSpec(
                kind=kind,
                uri=res.get("uri", ""),
                path=res.get("path", ""),
                size_bytes=res.get("size_bytes", 0),
                checksum_sha256=res.get("checksum_sha256", ""),
                required=res.get("required", True),
                load_order=res.get("load_order", 0),
                metadata=res.get("metadata") or {},
            )
        )
    return bindings


def _package_to_variant_field(
    pkg: dict,
) -> tuple[Optional[str], Optional[GGUFSource], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Convert a v2 package dict to VariantSpec field values.

    Returns (mlx, gguf, ort, mlc, ollama, source_repo).
    """
    executor = pkg.get("runtime_executor", "")
    fmt = pkg.get("artifact_format", "")
    weights = _get_weights_resource(pkg)

    mlx: Optional[str] = None
    gguf: Optional[GGUFSource] = None
    ort: Optional[str] = None
    mlc: Optional[str] = None
    ollama: Optional[str] = None
    source_repo: Optional[str] = None

    if weights is None:
        return mlx, gguf, ort, mlc, ollama, source_repo

    uri = weights.get("uri", "")
    path = weights.get("path", "")

    if executor == "mlx" or fmt == "mlx":
        # MLX repos: strip hf:// prefix, use full path as repo ID
        repo, _ = _parse_hf_uri(uri)
        mlx = repo
    elif executor in ("llamacpp", "llama.cpp") and fmt == "gguf":
        repo, filename = _parse_hf_uri(uri)
        meta = weights.get("metadata") or {}

        # Check for projector resource (mmproj for multimodal)
        projector = _get_projector_resource(pkg)
        proj_repo: Optional[str] = None
        proj_filename: Optional[str] = None
        if projector:
            proj_uri = projector.get("uri", "")
            pr, pf = _parse_hf_uri(proj_uri)
            proj_repo = pr if pr != repo else None  # omit if same repo
            proj_filename = pf or None

        gguf = GGUFSource(
            repo=repo,
            filename=path or filename,
            revision=meta.get("revision"),
            quantization_hint=(meta.get("quantization", "") or "").lower() or None,
            uri_type=meta.get("uri_type", "file"),
            projector_repo=proj_repo,
            projector_filename=proj_filename,
        )
    elif executor == "whisper" or executor == "whisper.cpp":
        # Whisper models use GGUF-like download pattern
        repo, filename = _parse_hf_uri(uri)
        meta = weights.get("metadata") or {}
        gguf = GGUFSource(
            repo=repo,
            filename=path or filename,
            revision=meta.get("revision"),
            quantization_hint=(meta.get("quantization", "") or "").lower() or None,
            uri_type=meta.get("uri_type", "file"),
        )
    elif executor in ("onnxruntime", "ort"):
        repo, _ = _parse_hf_uri(uri)
        ort = repo
    elif executor in ("mlc", "mlc-llm"):
        repo, _ = _parse_hf_uri(uri)
        mlc = repo
    elif executor == "ollama":
        # Ollama tags are stored directly in the URI (not hf://)
        ollama = uri
    else:
        # Unknown executor — store as source_repo for fallback
        repo, _ = _parse_hf_uri(uri)
        source_repo = repo

    return mlx, gguf, ort, mlc, ollama, source_repo


def _manifest_model_to_entry(model: dict) -> tuple[str, ModelEntry]:
    """Convert a single v2 manifest model dict to (catalog_key, ModelEntry).

    The catalog key is the model ``id`` field (e.g. ``"gemma-2-2b"``).
    """
    model_id: str = model.get("id", "")
    family: str = model.get("family", "")
    param_count: str = model.get("parameter_count", "")
    default_quant_raw: str = model.get("default_quantization", "4bit")

    # Map the default quant to canonical form
    default_quant = _QUANT_TO_CANONICAL.get(default_quant_raw, default_quant_raw)

    # Determine publisher from family
    publisher = _FAMILY_TO_PUBLISHER.get(family, "Unknown")

    # Extract task_taxonomy from model (renamed from legacy "modalities" field)
    task_taxonomy: list[str] = model.get("task_taxonomy", [])

    # Build variants and engines from packages
    packages = model.get("packages", [])
    engine_names: set[str] = set()

    # Collect input/output modalities and engine_config across all packages
    all_input_modalities: set[Modality] = set()
    all_output_modalities: set[Modality] = set()
    merged_engine_config: dict[str, Any] = {}
    all_resource_bindings: list[ResourceBindingSpec] = []

    # Group packages by canonical quant to build VariantSpec per quant
    quant_packages: dict[str, list[dict]] = {}
    for pkg in packages:
        pkg_quant_raw = pkg.get("quantization", default_quant_raw)
        pkg_quant = _QUANT_TO_CANONICAL.get(pkg_quant_raw, pkg_quant_raw)
        quant_packages.setdefault(pkg_quant, []).append(pkg)

        # Collect engine names
        executor = pkg.get("runtime_executor", "")
        engine = _EXECUTOR_TO_ENGINE.get(executor, executor)
        if engine:
            engine_names.add(engine)

        # Collect modalities from packages
        pkg_input = _parse_modalities(pkg.get("input_modalities"))
        pkg_output = _parse_modalities(pkg.get("output_modalities"))
        all_input_modalities.update(pkg_input)
        all_output_modalities.update(pkg_output)

        # Merge engine_config (first wins per key)
        for k, v in (pkg.get("engine_config") or {}).items():
            if k not in merged_engine_config:
                merged_engine_config[k] = v

        # Build resource bindings
        all_resource_bindings.extend(_build_resource_bindings(pkg))

    # Sort modalities for deterministic output
    input_modalities = sorted(all_input_modalities, key=lambda m: m.value)
    output_modalities = sorted(all_output_modalities, key=lambda m: m.value)

    # Build VariantSpec for each quant level by merging all packages
    variants: dict[str, VariantSpec] = {}
    for quant, pkgs in quant_packages.items():
        merged_mlx: Optional[str] = None
        merged_gguf: Optional[GGUFSource] = None
        merged_ort: Optional[str] = None
        merged_mlc: Optional[str] = None
        merged_ollama: Optional[str] = None
        merged_source: Optional[str] = None

        for pkg in pkgs:
            mlx, gguf, ort_val, mlc_val, ollama_val, source = _package_to_variant_field(pkg)
            if mlx and not merged_mlx:
                merged_mlx = mlx
            if gguf and not merged_gguf:
                merged_gguf = gguf
            if ort_val and not merged_ort:
                merged_ort = ort_val
            if mlc_val and not merged_mlc:
                merged_mlc = mlc_val
            if ollama_val and not merged_ollama:
                merged_ollama = ollama_val
            if source and not merged_source:
                merged_source = source

        variants[quant] = VariantSpec(
            mlx=merged_mlx,
            gguf=merged_gguf,
            ort=merged_ort,
            mlc=merged_mlc,
            ollama=merged_ollama,
            source_repo=merged_source,
        )

    return model_id, ModelEntry(
        publisher=publisher,
        params=param_count,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        default_quant=default_quant,
        variants=variants,
        engines=frozenset(engine_names),
        task_taxonomy=task_taxonomy,
        engine_config=merged_engine_config,
        resource_bindings=all_resource_bindings,
    )


def _hydrate_manifest(manifest: dict) -> dict[str, ModelEntry]:
    """Convert a v2 nested manifest to the catalog dict format.

    Walks the canonical nested manifest (family → variants → versions →
    packages) directly.  Each variant becomes a ``ModelEntry`` keyed by
    variant name.
    """
    result: dict[str, ModelEntry] = {}
    for family_name, family_data in manifest.items():
        if not isinstance(family_data, dict) or "variants" not in family_data:
            continue

        # Extract task_taxonomy from family level (renamed from legacy "modalities")
        family_task_taxonomy: list[str] = family_data.get("task_taxonomy", family_data.get("modalities", []))

        for variant_name, variant_data in family_data["variants"].items():
            packages: list[dict] = []
            for ver_data in variant_data.get("versions", {}).values():
                packages.extend(ver_data.get("packages", []))

            quants = variant_data.get("quantizations", [])
            default_quant = quants[0].lower() if quants else "q4_k_m"

            # Variant-level task_taxonomy overrides family-level
            variant_task_taxonomy: list[str] = variant_data.get(
                "task_taxonomy", variant_data.get("modalities", family_task_taxonomy)
            )

            model: dict = {
                "id": variant_name,
                "family": family_name,
                "name": variant_name,
                "parameter_count": variant_data.get("parameter_count", ""),
                "default_quantization": default_quant,
                "packages": packages,
                "task_taxonomy": variant_task_taxonomy,
            }
            try:
                key, entry = _manifest_model_to_entry(model)
                if key:
                    result[key] = entry
            except Exception:
                logger.debug(
                    "Failed to hydrate manifest model %s",
                    variant_name,
                    exc_info=True,
                )
    return result


def _build_aliases(manifest: dict) -> dict[str, str]:
    """Build alias map from v2 manifest model names and families.

    Walks the canonical nested manifest (family → variants) directly.
    Maps common name variations to canonical model IDs:
    - family name -> model ID (e.g. "gemma-2" -> "gemma-2-2b")
    - lowercase model name -> model ID
    - name with spaces/hyphens normalized -> model ID
    """
    aliases: dict[str, str] = {}
    family_models: dict[str, list[str]] = {}

    for family_name, family_data in manifest.items():
        if not isinstance(family_data, dict) or "variants" not in family_data:
            continue
        for variant_name, variant_data in family_data["variants"].items():
            if not variant_name:
                continue

            # Name-based aliases (lowercase, normalized)
            name_lower = variant_name.lower()
            name_hyphen = re.sub(r"\s+", "-", name_lower)
            if name_hyphen != variant_name:
                aliases[name_hyphen] = variant_name
            aliases[name_lower] = variant_name

            family_models.setdefault(family_name, []).append(variant_name)

    # For families with only one variant, alias the family name to that variant
    for family, variant_ids in family_models.items():
        if len(variant_ids) == 1:
            aliases[family] = variant_ids[0]

    return aliases


# ---------------------------------------------------------------------------
# Singleton client — lazily initialized
# ---------------------------------------------------------------------------

_client: Optional[CatalogClientV2] = None


def _get_client() -> CatalogClientV2:
    """Return the module-level CatalogClientV2 singleton."""
    global _client
    if _client is None:
        _client = CatalogClientV2()
    return _client


def _get_catalog() -> dict[str, ModelEntry]:
    """Fetch and hydrate the catalog from the v2 manifest (cached)."""
    manifest = _get_client().get_manifest()
    return _hydrate_manifest(manifest)


def _get_aliases() -> dict[str, str]:
    """Build aliases from the v2 manifest (cached)."""
    manifest = _get_client().get_manifest()
    return _build_aliases(manifest)


# ---------------------------------------------------------------------------
# Public module-level names — backward-compatible
# ---------------------------------------------------------------------------

# These are properties accessed by many consumers. We use a lazy wrapper
# that fetches from the server on first access but looks like a plain dict.


class _LazyDict(dict):  # type: ignore[type-arg]
    """Dict that populates itself on first access from a callable."""

    def __init__(self, loader: Any) -> None:
        super().__init__()
        self._loader = loader
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._loaded = True
            data = self._loader()
            super().update(data)

    def __getitem__(self, key: Any) -> Any:
        self._ensure_loaded()
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        self._ensure_loaded()
        return super().__contains__(key)

    def __iter__(self) -> Any:
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self) -> int:
        self._ensure_loaded()
        return super().__len__()

    def get(self, key: Any, default: Any = None) -> Any:
        self._ensure_loaded()
        return super().get(key, default)

    def keys(self) -> Any:
        self._ensure_loaded()
        return super().keys()

    def values(self) -> Any:
        self._ensure_loaded()
        return super().values()

    def items(self) -> Any:
        self._ensure_loaded()
        return super().items()

    def __repr__(self) -> str:
        self._ensure_loaded()
        return super().__repr__()

    def __bool__(self) -> bool:
        self._ensure_loaded()
        return len(self) > 0


CATALOG: dict[str, ModelEntry] = _LazyDict(_get_catalog)  # type: ignore[assignment]

MODEL_ALIASES: dict[str, str] = _LazyDict(_get_aliases)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public helper functions — unchanged signatures
# ---------------------------------------------------------------------------


def _resolve_alias(name: str) -> str:
    """Resolve a model name alias to its canonical catalog key."""
    return MODEL_ALIASES.get(name.lower(), name.lower())


def list_models() -> list[str]:
    """Return sorted list of all known model family names."""
    return sorted(CATALOG.keys())


def get_model(name: str) -> Optional[ModelEntry]:
    """Look up a model entry by family name, checking aliases."""
    key = _resolve_alias(name)
    return CATALOG.get(key)


def resolve_ollama_tag(tag: str) -> Optional[tuple[str, str]]:
    """Reverse-lookup an Ollama tag to (catalog_name, quant).

    Searches all catalog entries for a variant whose ``ollama`` field
    matches *tag* (case-insensitive).  Returns ``None`` if no match.

    Examples::

        resolve_ollama_tag("qwen2.5:3b")  -> ("qwen-3b", "4bit")
        resolve_ollama_tag("gemma3:1b")   -> ("gemma-1b", "4bit")
    """
    tag_lower = tag.lower()
    for family, entry in CATALOG.items():
        for quant, variant in entry.variants.items():
            if variant.ollama and variant.ollama.lower() == tag_lower:
                return (family, quant)
    return None


def supports_engine(name: str, engine: str) -> bool:
    """Check if a model family is known to work on a given engine."""
    entry = CATALOG.get(name.lower())
    if entry is None:
        return False
    return engine in entry.engines


def is_moe_model(name: str) -> bool:
    """Check if a model uses Mixture of Experts architecture."""
    entry = get_model(name)
    if entry is None:
        return False
    return entry.architecture == "moe"


def list_moe_models() -> list[str]:
    """Return sorted list of all MoE model family names."""
    return sorted(name for name, entry in CATALOG.items() if entry.architecture == "moe")


def get_moe_metadata(name: str) -> Optional[MoEMetadata]:
    """Get MoE metadata for a model, or None if not an MoE model."""
    entry = get_model(name)
    if entry is None:
        return None
    return entry.moe
