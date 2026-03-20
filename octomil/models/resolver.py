"""Model resolution — maps ``model:variant`` to engine-specific artifacts.

Takes a parsed model specifier and resolves it to a concrete HuggingFace
repo ID (and optional filename) for a given engine.

Powered by the v2 manifest: resolution selects the best package for the
requested engine and platform from the catalog manifest.

Usage::

    from octomil.models import resolve

    r = resolve("gemma-3b:4bit", available_engines=["mlx-lm", "llama.cpp"])
    print(r.engine, r.hf_repo, r.filename)
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from octomil._generated.modality import Modality

from .catalog import (
    _EXECUTOR_TO_ENGINE,
    _QUANT_TO_CANONICAL,
    CATALOG,
    ModelEntry,
    MoEMetadata,
    ResourceBindingSpec,
    _build_resource_bindings,
    _parse_hf_uri,
    _parse_modalities,
    _resolve_alias,
    resolve_ollama_tag,
)
from .catalog_client import CatalogClientV2
from .parser import normalize_variant, parse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedModel:
    """Result of resolving a model specifier to engine-specific artifacts."""

    family: Optional[str]
    quant: str
    engine: Optional[str]
    hf_repo: str
    input_modalities: list[Modality]
    output_modalities: list[Modality]
    filename: Optional[str] = None
    mlx_repo: Optional[str] = None
    source_repo: Optional[str] = None
    raw: str = ""
    architecture: str = "dense"
    moe: Optional[MoEMetadata] = None
    capabilities: list[str] = field(default_factory=list)  # type: ignore[assignment]
    engine_config: dict[str, Any] = field(default_factory=dict)  # type: ignore[assignment]
    resource_bindings: list[ResourceBindingSpec] = field(default_factory=list)  # type: ignore[assignment]

    @property
    def is_reasoning(self) -> bool:
        """True if this model emits separable thinking tokens."""
        return "reasoning" in self.capabilities

    @property
    def is_gguf(self) -> bool:
        return self.filename is not None and self.filename.endswith(".gguf")

    @property
    def is_moe(self) -> bool:
        """True if this model uses Mixture of Experts architecture."""
        return self.architecture == "moe" and self.moe is not None

    @property
    def is_multimodal(self) -> bool:
        """True if this model accepts non-text input modalities."""
        return any(m != Modality.TEXT for m in self.input_modalities)


class ModelResolutionError(ValueError):
    """Raised when a model specifier cannot be resolved."""


# Engine name normalization for matching — these are just string aliases,
# not proprietary intelligence.  Kept client-side for fast normalization.
_ENGINE_ALIASES: dict[str, str] = {
    "mlx": "mlx-lm",
    "mlx-lm": "mlx-lm",
    "gguf": "llama.cpp",
    "llama.cpp": "llama.cpp",
    "llamacpp": "llama.cpp",
    "llama-cpp": "llama.cpp",
    "mnn": "mnn",
    "mnn-llm": "mnn",
    "executorch": "executorch",
    "onnxruntime": "onnxruntime",
    "onnx": "onnxruntime",
    "ort": "onnxruntime",
    "mlc-llm": "mlc-llm",
    "mlc": "mlc-llm",
    "mlcllm": "mlc-llm",
    "whisper.cpp": "whisper.cpp",
    "whisper": "whisper.cpp",
    "whispercpp": "whisper.cpp",
    "ollama": "ollama",
    "echo": "echo",
}

# Reverse map: canonical engine name -> v2 manifest runtime_executor names
_ENGINE_TO_EXECUTORS: dict[str, list[str]] = {
    "llama.cpp": ["llamacpp", "llama.cpp"],
    "mlx-lm": ["mlx", "mlx-lm"],
    "whisper.cpp": ["whisper", "whisper.cpp"],
    "onnxruntime": ["onnxruntime", "ort"],
    "mlc-llm": ["mlc", "mlc-llm"],
    "ollama": ["ollama"],
    "echo": ["echo"],
    "mnn": ["mnn"],
    "executorch": ["executorch"],
    "cactus": ["cactus"],
}

# Backward-compatible module-level name for direct imports.
# Tests that do ``from octomil.models.resolver import _ENGINE_PRIORITY``
# will get the fallback value.  Runtime code should call _get_engine_priority().
_ENGINE_PRIORITY: list[str] = ["auto"]


def _normalize_engine(engine: str) -> str:
    """Normalize an engine name to its canonical form."""
    return _ENGINE_ALIASES.get(engine.lower(), engine.lower())


# ---------------------------------------------------------------------------
# V2 manifest-based resolution
# ---------------------------------------------------------------------------

_v2_client: Optional[CatalogClientV2] = None


def _get_v2_client() -> CatalogClientV2:
    """Return the module-level CatalogClientV2 singleton for resolution."""
    global _v2_client
    if _v2_client is None:
        _v2_client = CatalogClientV2()
    return _v2_client


def _find_manifest_model(manifest: dict, family: str, model_id: str | None = None) -> Optional[dict]:
    """Find a model in the nested manifest by variant ID or family match.

    Walks the canonical nested manifest (family → variants → versions →
    packages) and returns a flat dict compatible with ``_select_package``.

    Tries exact variant ID match first, then family match (first variant),
    then variant-ID-starts-with-family prefix match.
    """
    family_lower = family.lower()
    id_lower = model_id.lower() if model_id else family_lower

    def _variant_to_flat(fname: str, vname: str, vdata: dict) -> dict:
        packages: list[dict] = []
        for ver_data in vdata.get("versions", {}).values():
            packages.extend(ver_data.get("packages", []))
        quants = vdata.get("quantizations", [])
        default_quant = quants[0].lower() if quants else "q4_k_m"
        # Look up family-level capabilities as fallback
        family_caps = manifest.get(fname, {}).get("capabilities", []) if fname in manifest else []
        return {
            "id": vname,
            "family": fname,
            "parameter_count": vdata.get("parameter_count", ""),
            "default_quantization": default_quant,
            "packages": packages,
            "capabilities": vdata.get("capabilities", family_caps),
        }

    # Collect all (family_name, variant_name, variant_data) tuples
    all_variants: list[tuple[str, str, dict]] = []
    for fname, fdata in manifest.items():
        if not isinstance(fdata, dict) or "variants" not in fdata:
            continue
        for vname, vdata in fdata["variants"].items():
            all_variants.append((fname, vname, vdata))

    # 1. Exact variant ID match
    for fname, vname, vdata in all_variants:
        if vname.lower() == id_lower:
            return _variant_to_flat(fname, vname, vdata)

    # 2. Exact family match (returns first variant in that family)
    for fname, vname, vdata in all_variants:
        if fname.lower() == family_lower:
            return _variant_to_flat(fname, vname, vdata)

    # 3. Variant ID starts with family (e.g. family="gemma-2", id="gemma-2-2b")
    for fname, vname, vdata in all_variants:
        vid = vname.lower()
        if vid.startswith(family_lower + "-") or vid.startswith(family_lower):
            return _variant_to_flat(fname, vname, vdata)

    return None


def _select_package(
    packages: list[dict],
    quant: str,
    engine: str | None = None,
    available_engines: list[str] | None = None,
) -> Optional[dict]:
    """Select the best package from a model's package list.

    Selection priority:
    1. If engine is specified, find a package matching that engine + quant
    2. If available_engines is provided, find the best match in priority order
    3. Pick the default package (is_default=True) matching quant
    4. Pick any package matching quant

    The ``quant`` parameter should already be in raw manifest form
    (e.g. ``q4_k_m``) or canonical form (e.g. ``4bit``).
    """
    quant_lower = quant.lower()
    # Also get the canonical form for matching
    quant_canonical = _QUANT_TO_CANONICAL.get(quant_lower, quant_lower)

    def _quant_matches(pkg: dict) -> bool:
        pkg_quant = pkg.get("quantization", "").lower()
        pkg_canonical = _QUANT_TO_CANONICAL.get(pkg_quant, pkg_quant)
        return pkg_quant == quant_lower or pkg_canonical == quant_canonical

    def _engine_matches(pkg: dict, target_engine: str) -> bool:
        executor = pkg.get("runtime_executor", "")
        pkg_engine = _EXECUTOR_TO_ENGINE.get(executor, executor)
        target_executors = _ENGINE_TO_EXECUTORS.get(target_engine, [target_engine])
        return executor in target_executors or pkg_engine == target_engine

    # 1. Specific engine requested
    if engine:
        norm_engine = _normalize_engine(engine)
        for pkg in packages:
            if _quant_matches(pkg) and _engine_matches(pkg, norm_engine):
                return pkg
        # Relax quant constraint — just match engine
        for pkg in packages:
            if _engine_matches(pkg, norm_engine):
                return pkg
        return None

    # 2. Available engines — pick first match in order
    if available_engines:
        norm_available = [_normalize_engine(e) for e in available_engines]
        for eng in norm_available:
            for pkg in packages:
                if _quant_matches(pkg) and _engine_matches(pkg, eng):
                    return pkg

    # 3. Default package matching quant
    for pkg in packages:
        if _quant_matches(pkg) and pkg.get("is_default", False):
            return pkg

    # 4. Any package matching quant
    for pkg in packages:
        if _quant_matches(pkg):
            return pkg

    # 5. Absolute fallback: default package regardless of quant
    for pkg in packages:
        if pkg.get("is_default", False):
            return pkg

    # 6. First package
    return packages[0] if packages else None


def _resolve_from_manifest(
    name: str,
    parsed_family: str,
    parsed_variant: str | None,
    *,
    engine: str | None = None,
    available_engines: list[str] | None = None,
) -> Optional[ResolvedModel]:
    """Attempt to resolve a model specifier from the v2 manifest.

    Returns None if the model is not found in the manifest,
    allowing fallback to the legacy catalog-based resolution.
    """
    client = _get_v2_client()
    try:
        manifest = client.get_manifest()
    except Exception:
        logger.debug("Failed to get v2 manifest for resolution", exc_info=True)
        return None

    if not manifest:
        return None

    # Resolve alias first
    canonical = _resolve_alias(parsed_family)

    # Find model in manifest
    manifest_model = _find_manifest_model(manifest, canonical, model_id=canonical)
    if manifest_model is None:
        # Try with the original (un-aliased) family name
        manifest_model = _find_manifest_model(manifest, parsed_family, model_id=parsed_family)

    if manifest_model is None:
        return None

    # Determine quant
    default_quant_raw = manifest_model.get("default_quantization", "4bit")
    if parsed_variant is not None:
        quant = normalize_variant(parsed_variant)
    else:
        quant = _QUANT_TO_CANONICAL.get(default_quant_raw, default_quant_raw)

    # Select best package
    packages = manifest_model.get("packages", [])
    if not packages:
        return None

    selected = _select_package(packages, quant, engine=engine, available_engines=available_engines)
    if selected is None:
        return None

    # Build ResolvedModel from selected package
    executor = selected.get("runtime_executor", "")
    resolved_engine = _EXECUTOR_TO_ENGINE.get(executor, executor)
    if engine:
        resolved_engine = _normalize_engine(engine)

    fmt = selected.get("artifact_format", "")
    weights = None
    for res in selected.get("resources", []):
        if res.get("kind") == "weights":
            weights = res
            break

    if weights is None:
        return None

    uri = weights.get("uri", "")
    path = weights.get("path", "")
    repo, filename = _parse_hf_uri(uri)

    hf_repo: str
    resolved_filename: str | None = None
    mlx_repo: str | None = None

    if executor in ("mlx", "mlx-lm") or fmt == "mlx":
        hf_repo = repo
        mlx_repo = repo
    elif executor in ("llamacpp", "llama.cpp") and fmt == "gguf":
        hf_repo = repo
        resolved_filename = path or filename
    elif executor in ("whisper", "whisper.cpp"):
        hf_repo = repo
        resolved_filename = path or filename
    elif executor in ("onnxruntime", "ort"):
        hf_repo = repo
    elif executor in ("mlc", "mlc-llm"):
        hf_repo = repo
    elif executor == "ollama":
        hf_repo = uri  # Ollama tags stored directly
    else:
        hf_repo = repo
        if path and path != ".":
            resolved_filename = path

    model_id = manifest_model.get("id", canonical)
    pkg_quant_raw = selected.get("quantization", default_quant_raw)
    resolved_quant = _QUANT_TO_CANONICAL.get(pkg_quant_raw, pkg_quant_raw)

    # Extract modalities and engine_config from selected package
    pkg_input_modalities = _parse_modalities(selected.get("input_modalities"))
    pkg_output_modalities = _parse_modalities(selected.get("output_modalities"))
    pkg_engine_config: dict[str, Any] = selected.get("engine_config") or {}
    pkg_resource_bindings = _build_resource_bindings(selected)

    manifest_capabilities: list[str] = manifest_model.get("capabilities", [])

    return ResolvedModel(
        family=model_id,
        quant=resolved_quant,
        engine=resolved_engine,
        hf_repo=hf_repo,
        input_modalities=pkg_input_modalities,
        output_modalities=pkg_output_modalities,
        filename=resolved_filename,
        mlx_repo=mlx_repo,
        source_repo=None,
        raw=name,
        capabilities=manifest_capabilities,
        engine_config=pkg_engine_config,
        resource_bindings=pkg_resource_bindings,
    )


def _suggest_models(name: str, n: int = 3) -> list[str]:
    """Return close matches for typo suggestions."""
    return list(difflib.get_close_matches(name, CATALOG.keys(), n=n, cutoff=0.4))


def _pick_engine(
    entry: ModelEntry,
    quant: str,
    available_engines: list[str],
) -> Optional[str]:
    """Pick the best available engine for a model + quant combination."""
    variant = entry.variants.get(quant)
    if variant is None:
        return None

    normalized_available = [_normalize_engine(e) for e in available_engines]
    engine_priority = list(dict.fromkeys(normalized_available))

    for engine_name in engine_priority:
        if engine_name not in normalized_available:
            continue
        if engine_name == "ollama" and variant.ollama:
            return engine_name
        if engine_name not in entry.engines:
            continue
        if engine_name == "mlx-lm" and variant.mlx:
            return engine_name
        if engine_name == "mlc-llm" and (variant.mlc or variant.source_repo):
            return engine_name
        if engine_name == "llama.cpp" and variant.gguf:
            return engine_name
        if engine_name in ("mnn", "executorch") and (variant.gguf or variant.source_repo):
            return engine_name
        if engine_name == "onnxruntime" and (variant.ort or variant.source_repo):
            return engine_name
        if engine_name == "echo":
            return engine_name

    return None


def resolve(
    name: str,
    *,
    available_engines: Optional[list[str]] = None,
    engine: Optional[str] = None,
) -> ResolvedModel:
    """Resolve a model specifier to an engine-specific artifact.

    Resolution strategy:
    1. Parse the model specifier
    2. If passthrough (local file or full repo ID), return directly
    3. Try direct v2 manifest lookup (raw manifest packages)
    4. Fall back to CATALOG lookup (hydrated from same v2 manifest)

    Parameters
    ----------
    name:
        Model specifier (e.g. ``"gemma-3b:4bit"``, ``"phi-mini"``,
        ``"mlx-community/model-4bit"``).
    available_engines:
        List of engine names available on this system. Used to pick
        the best engine when ``engine`` is not specified.
    engine:
        Force a specific engine. Overrides ``available_engines``.

    Returns
    -------
    ResolvedModel
        Resolved artifact with ``hf_repo``, ``filename`` (for GGUF),
        ``engine``, and ``quant``.

    Raises
    ------
    ModelResolutionError
        If the model family or variant is not found, or no engine
        can serve the requested combination.
    """
    parsed = parse(name)

    # --- Passthrough: local files and full repo IDs ---
    if parsed.is_passthrough:
        resolved_engine = _normalize_engine(engine) if engine else None
        return ResolvedModel(
            family=None,
            quant="unknown",
            engine=resolved_engine,
            hf_repo=parsed.raw,
            input_modalities=[Modality.TEXT],
            output_modalities=[Modality.TEXT],
            filename=parsed.raw if parsed.is_local_file else None,
            raw=name,
        )

    # --- V2 manifest resolution (preferred path) ---
    assert parsed.family is not None
    manifest_result = _resolve_from_manifest(
        name,
        parsed.family,
        parsed.variant,
        engine=engine,
        available_engines=available_engines,
    )
    if manifest_result is not None:
        return manifest_result

    # --- CATALOG lookup (hydrated from the same v2 manifest) ---
    canonical = _resolve_alias(parsed.family)
    entry = CATALOG.get(canonical)

    # Try reverse Ollama tag lookup.
    # e.g. "qwen2.5:3b" -> catalog entry "qwen-3b" at quant "4bit"
    ollama_resolved_quant: str | None = None
    if entry is None:
        ollama_match = resolve_ollama_tag(name)
        if ollama_match is not None:
            canonical, ollama_resolved_quant = ollama_match
            entry = CATALOG.get(canonical)
            logger.info("Resolved Ollama tag '%s' to catalog '%s:%s'", name, canonical, ollama_resolved_quant)

    if entry is None:
        suggestions = _suggest_models(parsed.family)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ModelResolutionError(
            f"Unknown model '{parsed.family}'. Available: {', '.join(sorted(CATALOG.keys()))}.{hint}"
        )

    # Determine quant level
    if ollama_resolved_quant is not None:
        quant = ollama_resolved_quant
    elif parsed.variant is not None:
        quant = normalize_variant(parsed.variant)
    else:
        quant = entry.default_quant

    # Check variant exists
    variant = entry.variants.get(quant)
    if variant is None and ollama_resolved_quant is None:
        ollama_match = resolve_ollama_tag(name)
        if ollama_match is not None:
            canonical, quant = ollama_match
            ollama_entry = CATALOG.get(canonical)
            if ollama_entry is not None:
                entry = ollama_entry
                variant = entry.variants.get(quant)

    if variant is None:
        available_quants = ", ".join(sorted(entry.variants.keys()))
        raise ModelResolutionError(
            f"Unknown variant '{parsed.variant or quant}' for model '{parsed.family}'. Available: {available_quants}"
            if available_quants
            else f"Model '{parsed.family}' has no downloadable variants. "
            f"Try a full HuggingFace repo ID (e.g. 'mlx-community/gemma-2-2b-it-4bit')."
        )

    # Determine engine
    if engine:
        resolved_engine = _normalize_engine(engine)
    elif available_engines:
        resolved_engine = _pick_engine(entry, quant, available_engines)
    else:
        resolved_engine = _pick_engine(entry, quant, list(_ENGINE_ALIASES.values()))

    # Build result based on engine
    hf_repo: str
    filename: Optional[str] = None
    mlx_repo: Optional[str] = variant.mlx

    if resolved_engine == "mlx-lm" and variant.mlx:
        hf_repo = variant.mlx
    elif resolved_engine == "mlc-llm" and variant.mlc:
        hf_repo = variant.mlc
    elif resolved_engine == "llama.cpp" and variant.gguf:
        hf_repo = variant.gguf.repo
        filename = variant.gguf.filename
    elif resolved_engine == "onnxruntime" and variant.ort:
        hf_repo = variant.ort
    elif resolved_engine == "ollama" and variant.ollama:
        hf_repo = variant.ollama
    elif variant.gguf:
        hf_repo = variant.gguf.repo
        filename = variant.gguf.filename
    elif variant.mlx:
        hf_repo = variant.mlx
    elif variant.source_repo:
        hf_repo = variant.source_repo
    else:
        raise ModelResolutionError(
            f"No artifact found for '{parsed.family}:{quant}' "
            f"on engine '{resolved_engine}'. "
            f"Try a different variant or pass a full HuggingFace repo ID."
        )

    return ResolvedModel(
        family=canonical,
        quant=quant,
        engine=resolved_engine,
        hf_repo=hf_repo,
        input_modalities=entry.input_modalities,
        output_modalities=entry.output_modalities,
        filename=filename,
        mlx_repo=mlx_repo,
        source_repo=variant.source_repo,
        raw=name,
        capabilities=entry.capabilities,
        architecture=entry.architecture,
        moe=entry.moe,
        engine_config=entry.engine_config,
        resource_bindings=entry.resource_bindings,
    )
