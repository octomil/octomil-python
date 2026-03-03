"""Model registry with tag syntax and multi-source backends.

Provides structured model metadata, tag-based resolution (``name:tag``),
and multi-source download paths for HuggingFace, Ollama, and Kaggle.

Model family data is fetched from the Octomil server at runtime and
cached locally. An empty fallback is used when the server is
unreachable — full repo paths still work as passthrough.

Usage::

    from octomil.model_registry import parse_model_tag, resolve_model, MODEL_FAMILIES

    family, tag = parse_model_tag("gemma-4b:q8_0")
    result = resolve_model("gemma-4b:q8_0", backend="mlx")
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Optional

from .models.catalog_client import ModelFamiliesClient

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

REDACTED


def _sort_sources_by_trust(sources: list[ModelSource]) -> list[ModelSource]:
    """Sort sources by trust level (official first)."""
    return sorted(sources, key=lambda s: TRUST_PRIORITY.get(s.trust, 99))


# ---------------------------------------------------------------------------
# Server-fetched model families (singleton + hydration)
# ---------------------------------------------------------------------------

_families_client: Optional[ModelFamiliesClient] = None


def _get_families_client() -> ModelFamiliesClient:
    """Return the module-level ModelFamiliesClient singleton."""
    global _families_client
    if _families_client is None:
        _families_client = ModelFamiliesClient()
    return _families_client


def _source_from_dict(d: dict[str, Any]) -> ModelSource:
    """Hydrate a ModelSource from a server dict."""
    return ModelSource(
        type=d.get("type", "huggingface"),
        ref=d.get("ref", ""),
        file=d.get("file"),
        trust=d.get("trust", "community"),
    )


def _variant_from_dict(d: dict[str, Any]) -> ModelVariant:
    """Hydrate a ModelVariant from a server dict."""
    sources_raw = d.get("sources", [])
    sources = [_source_from_dict(s) for s in sources_raw if isinstance(s, dict)]
    return ModelVariant(
        quantization_family=d.get("quantization_family", "4bit"),
        sources=sources,
        mlx=d.get("mlx"),
    )


def _family_from_dict(d: dict[str, Any]) -> ModelFamily:
    """Hydrate a ModelFamily from a server dict."""
    variants_raw = d.get("variants", {})
    variants = {k: _variant_from_dict(v) for k, v in variants_raw.items() if isinstance(v, dict)}
    return ModelFamily(
        default_tag=d.get("default_tag", DEFAULT_TAG),
        publisher=d.get("publisher", ""),
        params=d.get("params", ""),
        variants=variants,
    )


def _hydrate_families(raw: dict[str, Any]) -> dict[str, ModelFamily]:
    """Convert a server-returned families dict to typed ModelFamily dict."""
    result: dict[str, ModelFamily] = {}
    for name, family_data in raw.items():
        if isinstance(family_data, ModelFamily):
            result[name] = family_data
        elif isinstance(family_data, dict):
            try:
                result[name] = _family_from_dict(family_data)
            except Exception:
                pass
    return result


def _get_model_families() -> dict[str, ModelFamily]:
    """Fetch and hydrate model families from the server (cached)."""
    raw = _get_families_client().get_families()
    return _hydrate_families(raw)


# ---------------------------------------------------------------------------
# Lazy-loading MODEL_FAMILIES for backward compatibility
# ---------------------------------------------------------------------------


class _LazyFamiliesDict(dict):  # type: ignore[type-arg]
    """Dict that populates itself on first access from the server."""

    def __init__(self) -> None:
        super().__init__()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._loaded = True
            data = _get_model_families()
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
        return super().__bool__()


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
        or a full repo path like ``"REDACTED"``.
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
            source=ModelSource(type="huggingface", ref=tag, REDACTED),
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
