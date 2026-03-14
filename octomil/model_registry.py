"""Model registry with tag syntax and multi-source backends.

Provides structured model metadata, tag-based resolution (``name:tag``),
and multi-source download paths for HuggingFace, Ollama, and Kaggle.

Model data is derived from the v2 catalog manifest via
:class:`~octomil.models.catalog_client.CatalogClientV2`.

Usage::

    from octomil.model_registry import parse_model_tag, resolve_model, MODEL_FAMILIES

    family, tag = parse_model_tag("gemma-4b:q8_0")
    result = resolve_model("gemma-4b:q8_0", backend="mlx")
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Optional

from .models.catalog import _FAMILY_TO_PUBLISHER, _QUANT_TO_CANONICAL, _parse_hf_uri
from .models.catalog_client import CatalogClientV2

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelSource:
    """A single downloadable source for a model variant."""

    type: str  # "huggingface", "ollama", "kaggle"
    ref: str  # repo ID, ollama ref, or kaggle model path
    file: Optional[str] = None  # specific file in repo (for GGUF)
    trust: str = "community"  # "official", "curated", "community"


@dataclass
class ModelVariant:
    """A specific quantization/format variant of a model family."""

    quantization_family: str  # "4bit", "8bit", "16bit", "32bit"
    sources: list[ModelSource] = field(default_factory=list)
    mlx: Optional[str] = None  # MLX repo for Apple Silicon


@dataclass
class ModelFamily:
    """A named model family with multiple quantization variants."""

    default_tag: str
    publisher: str
    params: str  # e.g. "4B", "7B"
    variants: dict[str, ModelVariant] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tag syntax parser
# ---------------------------------------------------------------------------

DEFAULT_TAG = "q4_k_m"


def parse_model_tag(name: str) -> tuple[Optional[str], str]:
    """Parse ``name:tag`` into (family_key, tag).

    Returns ``(None, name)`` for full repo paths (containing ``/``).

    Examples::

        parse_model_tag("gemma-4b")         -> ("gemma-4b", "q4_k_m")
        parse_model_tag("gemma-4b:q8_0")    -> ("gemma-4b", "q8_0")
        parse_model_tag("mlx-community/foo") -> (None, "mlx-community/foo")
        parse_model_tag("user/repo:branch")  -> (None, "user/repo:branch")
    """
    # Full repo paths pass through unchanged
    if "/" in name:
        return None, name

    if ":" in name:
        family, tag = name.split(":", 1)
        return family, tag

    return name, DEFAULT_TAG


@dataclass
class ResolvedModel:
    """Result of resolving a model name to a downloadable reference."""

    family: Optional[str]
    tag: str
    source: Optional[ModelSource]
    mlx_repo: Optional[str]
    variant: Optional[ModelVariant]
    raw_name: str  # original input


class ModelResolutionError(ValueError):
    """Raised when a model name cannot be resolved."""


# ---------------------------------------------------------------------------
# Trust priority ordering
# ---------------------------------------------------------------------------

TRUST_PRIORITY: dict[str, int] = {"official": 0, "curated": 1, "community": 2}


def _sort_sources_by_trust(sources: list[ModelSource]) -> list[ModelSource]:
    """Sort sources by trust level (official first)."""
    return sorted(sources, key=lambda s: TRUST_PRIORITY.get(s.trust, 99))


# ---------------------------------------------------------------------------
# V2 manifest → ModelFamily conversion
# ---------------------------------------------------------------------------

_v2_client: Optional[CatalogClientV2] = None


def _get_v2_client() -> CatalogClientV2:
    """Return the module-level CatalogClientV2 singleton."""
    global _v2_client
    if _v2_client is None:
        _v2_client = CatalogClientV2()
    return _v2_client


def _manifest_to_families(manifest: dict) -> dict[str, ModelFamily]:
    """Convert a v2 catalog manifest to a dict of ModelFamily objects.

    Parses the canonical nested manifest format returned by the server::

        {
            "family_name": {
                "vendor": "...",
                "variants": {
                    "variant_name": {
                        "parameter_count": "2B",
                        "quantizations": ["Q4_K_M"],
                        "versions": {
                            "1.0.0": {
                                "packages": [...]
                            }
                        }
                    }
                }
            }
        }

    Each variant becomes a ModelFamily keyed by variant name (e.g. ``"gemma-2-2b"``).
    Packages across all versions are collected and grouped by quantization tag
    into ModelVariant objects, with each package contributing a ModelSource.
    """
    result: dict[str, ModelFamily] = {}

    for family_name, family_data in manifest.items():
        if not isinstance(family_data, dict) or "variants" not in family_data:
            continue

        vendor = family_data.get("vendor", "")
        publisher = _FAMILY_TO_PUBLISHER.get(family_name, vendor or "Unknown")

        for variant_name, variant_data in family_data["variants"].items():
            params: str = variant_data.get("parameter_count", "")
            quants = variant_data.get("quantizations", [])
            default_quant_raw: str = quants[0].lower() if quants else "q4_k_m"

            # Collect all packages across all versions
            tag_variants: dict[str, ModelVariant] = {}

            for _ver_key, ver_data in variant_data.get("versions", {}).items():
                for pkg in ver_data.get("packages", []):
                    quant_raw: str = pkg.get("quantization", default_quant_raw).lower()
                    executor: str = pkg.get("runtime_executor", "")

                    # Find weights resource
                    weights = None
                    for res in pkg.get("resources", []):
                        if res.get("kind") == "weights":
                            weights = res
                            break
                    if weights is None:
                        continue

                    uri: str = weights.get("uri", "")
                    path: str = weights.get("path", "")

                    # Build ModelSource from package
                    if executor == "ollama":
                        source = ModelSource(type="ollama", ref=uri, trust="curated")
                    elif uri.startswith("hf://"):
                        repo, filename = _parse_hf_uri(uri)
                        source = ModelSource(
                            type="huggingface",
                            ref=repo,
                            file=path or filename or None,
                            trust="official",
                        )
                    else:
                        source = ModelSource(type="huggingface", ref=uri, trust="community")

                    # Determine MLX repo
                    mlx_repo: Optional[str] = None
                    if executor in ("mlx", "mlx-lm"):
                        repo, _ = _parse_hf_uri(uri)
                        mlx_repo = repo

                    # Map quant to canonical form for quantization_family
                    quant_canonical = _QUANT_TO_CANONICAL.get(quant_raw, quant_raw)

                    if quant_raw not in tag_variants:
                        tag_variants[quant_raw] = ModelVariant(
                            quantization_family=quant_canonical,
                            sources=[source],
                            mlx=mlx_repo,
                        )
                    else:
                        existing = tag_variants[quant_raw]
                        existing.sources.append(source)
                        if mlx_repo and not existing.mlx:
                            existing.mlx = mlx_repo

            if not tag_variants:
                continue

            result[variant_name] = ModelFamily(
                default_tag=default_quant_raw,
                publisher=publisher,
                params=params,
                variants=tag_variants,
            )

    return result


def _get_model_families() -> dict[str, ModelFamily]:
    """Fetch and convert model families from the v2 manifest (cached)."""
    manifest = _get_v2_client().get_manifest()
    return _manifest_to_families(manifest)


# ---------------------------------------------------------------------------
# Lazy-loading MODEL_FAMILIES
# ---------------------------------------------------------------------------


class _LazyFamiliesDict(dict):  # type: ignore[type-arg]
    """Dict that populates itself on first access via a loader callable."""

    def __init__(self, loader: Any = None) -> None:
        super().__init__()
        self._loader = loader or _get_model_families
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


MODEL_FAMILIES: dict[str, ModelFamily] = _LazyFamiliesDict()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Resolution logic
# ---------------------------------------------------------------------------


def _suggest_families(name: str, n: int = 3) -> list[str]:
    """Find the closest matching family names for typo suggestions."""
    matches = difflib.get_close_matches(name, MODEL_FAMILIES.keys(), n=n, cutoff=0.4)
    return list(matches)


def resolve_model(
    name: str,
    *,
    backend: str = "auto",
    prefer_local: bool = True,
) -> ResolvedModel:
    """Resolve a model name (with optional tag) to a downloadable reference.

    Parameters
    ----------
    name:
        Model specifier. Can be ``"gemma-4b"``, ``"gemma-4b:q8_0"``,
        or a full repo path like ``"org/model"``.
    backend:
        Target backend: ``"mlx"``, ``"gguf"``, or ``"auto"``.
    prefer_local:
        When True, check for local Ollama cache before remote sources.

    Returns
    -------
    ResolvedModel
        Contains the resolved source, MLX repo (if applicable), and variant info.

    Raises
    ------
    ModelResolutionError
        If the model family or tag is not found in the registry.
    """
    family_key, tag = parse_model_tag(name)

    # Full repo path — pass through unchanged
    if family_key is None:
        return ResolvedModel(
            family=None,
            tag=tag,
            source=ModelSource(type="huggingface", ref=tag, trust="community"),
            mlx_repo=None,
            variant=None,
            raw_name=name,
        )

    # Look up family
    family = MODEL_FAMILIES.get(family_key)
    if family is None:
        suggestions = _suggest_families(family_key)
        suggestion_str = ""
        if suggestions:
            suggestion_str = f" Did you mean: {', '.join(suggestions)}?"
        raise ModelResolutionError(
            f"Unknown model '{family_key}'. Available: {', '.join(sorted(MODEL_FAMILIES))}.{suggestion_str}"
        )

    # Use family default tag if the parsed tag matches the default sentinel
    effective_tag = tag if tag != DEFAULT_TAG else family.default_tag

    # Look up variant
    variant = family.variants.get(effective_tag)
    if variant is None:
        available_tags = ", ".join(sorted(family.variants.keys()))
        raise ModelResolutionError(
            f"Unknown tag '{effective_tag}' for model '{family_key}'. Available tags: {available_tags}"
        )

    # Select best source by trust priority
    sources = _sort_sources_by_trust(variant.sources)

    # For MLX backend, prefer the mlx field
    mlx_repo = variant.mlx if backend in ("mlx", "auto") else None

    # Filter sources by backend preference
    if backend == "gguf":
        # Prefer sources with GGUF files
        gguf_sources = [s for s in sources if s.file and s.file.endswith(".gguf")]
        if gguf_sources:
            sources = gguf_sources
    elif backend == "mlx":
        # For MLX, the mlx repo is the primary; sources are fallback
        pass

    best_source = sources[0] if sources else None

    return ResolvedModel(
        family=family_key,
        tag=effective_tag,
        source=best_source,
        mlx_repo=mlx_repo,
        variant=variant,
        raw_name=name,
    )


def list_families() -> dict[str, ModelFamily]:
    """Return all registered model families."""
    return dict(MODEL_FAMILIES)


def get_family(name: str) -> Optional[ModelFamily]:
    """Return a specific model family by name, or None."""
    return MODEL_FAMILIES.get(name)
