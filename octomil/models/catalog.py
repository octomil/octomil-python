"""Unified model catalog — single source of truth for all engines.

Model entries map quant variants to engine-specific artifacts:

- ``mlx``: HuggingFace MLX repo ID
- ``gguf``: Tuple of (HF repo, filename)
- ``source``: Original model HF repo (for engines that download and convert)

The catalog and alias data are fetched from the Octomil server at
runtime and cached locally. A minimal fallback catalog (with no
engine-specific variants) is embedded for offline bootstrap only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .catalog_client import CatalogClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes — unchanged public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GGUFSource:
    """GGUF model source — HuggingFace repo + filename."""

    repo: str
    filename: str


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
    default_quant: str = "4bit"
    variants: dict[str, VariantSpec] = field(default_factory=dict)
    engines: frozenset[str] = frozenset()  # engines this model is known to work on
    architecture: str = "dense"  # "dense" or "moe"
    moe: Optional[MoEMetadata] = None  # populated when architecture == "moe"


# ---------------------------------------------------------------------------
# Server JSON -> dataclass hydration
# ---------------------------------------------------------------------------


def _gguf_from_dict(d: Any) -> Optional[GGUFSource]:
    """Hydrate a GGUFSource from a server dict or None."""
    if d is None:
        return None
    if isinstance(d, dict):
        return GGUFSource(repo=d.get("repo", ""), filename=d.get("filename", ""))
    return None


def _variant_from_dict(d: dict[str, Any]) -> VariantSpec:
    """Hydrate a VariantSpec from a server dict."""
    return VariantSpec(
        mlx=d.get("mlx"),
        gguf=_gguf_from_dict(d.get("gguf")),
        ort=d.get("ort"),
        mlc=d.get("mlc"),
        ollama=d.get("ollama"),
        source_repo=d.get("source_repo"),
    )


def _moe_from_dict(d: Any) -> Optional[MoEMetadata]:
    """Hydrate MoEMetadata from a server dict or None."""
    if d is None or not isinstance(d, dict):
        return None
    return MoEMetadata(
        num_experts=d.get("num_experts", 0),
        active_experts=d.get("active_experts", 0),
        expert_size=d.get("expert_size", ""),
        total_params=d.get("total_params", ""),
        active_params=d.get("active_params", ""),
    )


def _entry_from_dict(d: dict[str, Any]) -> ModelEntry:
    """Hydrate a ModelEntry from a server dict."""
    variants_raw = d.get("variants", {})
    variants = {k: _variant_from_dict(v) for k, v in variants_raw.items()}
    engines_raw = d.get("engines", [])
    engines = frozenset(engines_raw) if isinstance(engines_raw, list) else frozenset()
    return ModelEntry(
        publisher=d.get("publisher", ""),
        params=d.get("params", ""),
        default_quant=d.get("default_quant", "4bit"),
        variants=variants,
        engines=engines,
        architecture=d.get("architecture", "dense"),
        moe=_moe_from_dict(d.get("moe")),
    )


def _hydrate_catalog(raw: dict[str, Any]) -> dict[str, ModelEntry]:
    """Convert a server-returned catalog dict to typed ModelEntry dict."""
    result: dict[str, ModelEntry] = {}
    for name, entry_data in raw.items():
        if isinstance(entry_data, ModelEntry):
            # Already hydrated (from fallback)
            result[name] = entry_data
        elif isinstance(entry_data, dict):
            try:
                result[name] = _entry_from_dict(entry_data)
            except Exception:
                logger.debug("Failed to hydrate catalog entry %s", name, exc_info=True)
        else:
            logger.debug("Skipping invalid catalog entry %s (type=%s)", name, type(entry_data))
    return result


# ---------------------------------------------------------------------------
# Singleton client — lazily initialized
# ---------------------------------------------------------------------------

_client: Optional[CatalogClient] = None


def _get_client() -> CatalogClient:
    """Return the module-level CatalogClient singleton."""
    global _client
    if _client is None:
        _client = CatalogClient()
    return _client


def _get_catalog() -> dict[str, ModelEntry]:
    """Fetch and hydrate the catalog from the server (cached)."""
    raw = _get_client().get_catalog()
    return _hydrate_catalog(raw)


def _get_aliases() -> dict[str, str]:
    """Fetch catalog aliases from the server (cached)."""
    return _get_client().get_aliases()


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
        return super().__bool__()


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
