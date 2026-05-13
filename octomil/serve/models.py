"""Pydantic models and model catalog helpers for the serve package."""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from ..errors import OctomilError, OctomilErrorCode
from ..models.catalog import CATALOG as _UNIFIED_CATALOG
from ..models.resolver import ModelResolutionError as _NewResolutionError
from ..models.resolver import resolve as _resolve_new

# ---------------------------------------------------------------------------
# Pydantic models for OpenAI-compatible API
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    model_config = {"extra": "allow"}

    role: str
    content: Optional[Union[str, list[Any]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[Any]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionBody(BaseModel):
    model_config = {"extra": "allow"}

    model: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: Optional[int] = 512
    max_completion_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    response_format: Optional[dict[str, Any]] = None
    grammar: Optional[str] = None
    stop: Optional[Union[str, list[str]]] = None
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


# ---------------------------------------------------------------------------
# Model catalog -- backwards-compatible dicts derived from unified catalog
# ---------------------------------------------------------------------------

_MLX_MODELS: dict[str, str] = {}
_GGUF_MODELS: dict[str, tuple[str, str]] = {}

for _name, _entry in _UNIFIED_CATALOG.items():
    _default_variant = _entry.variants.get(_entry.default_quant)
    if _default_variant is None:
        continue
    if _default_variant.mlx:
        _MLX_MODELS[_name] = _default_variant.mlx
    if _default_variant.gguf:
        _GGUF_MODELS[_name] = (
            _default_variant.gguf.repo,
            _default_variant.gguf.filename,
        )


def resolve_model_name(name: str, backend: str) -> str:
    """Resolve a short model name (with optional :variant) to a HuggingFace repo ID.

    Uses the unified model catalog for structured resolution. Supports
    Ollama-style ``model:variant`` syntax (e.g. ``gemma-3b:4bit``).

    Full repo paths (containing ``/``) and local file paths pass through
    unchanged.
    """
    # Local file path
    if name.endswith((".gguf", ".pte", ".mnn")):
        return name

    # Full repo path -- pass through
    if "/" in name:
        return name

    # Map backend names to engine names for the resolver
    engine_map = {"mlx": "mlx-lm", "gguf": "llama.cpp"}
    engine = engine_map.get(backend, backend)

    try:
        resolved = _resolve_new(name, engine=engine)
    except _NewResolutionError as exc:
        raise OctomilError(code=OctomilErrorCode.MODEL_NOT_FOUND, message=str(exc)) from exc

    if backend == "mlx":
        if resolved.mlx_repo:
            return resolved.mlx_repo
        if resolved.hf_repo:
            return resolved.hf_repo
        raise OctomilError(
            code=OctomilErrorCode.MODEL_NOT_FOUND,
            message=f"No MLX source found for '{name}'. Pass a full HuggingFace repo ID (e.g. 'mlx-community/model-4bit').",
        )

    if backend == "gguf":
        # For GGUF, return the short name -- LlamaCppBackend resolves via _GGUF_MODELS
        family = resolved.family
        if family and family in _GGUF_MODELS:
            return family
        # If the resolver found a GGUF artifact, return the family name
        if resolved.is_gguf and family:
            return family
        raise OctomilError(
            code=OctomilErrorCode.MODEL_NOT_FOUND,
            message=f"No GGUF source found for '{name}'. Pass a path to a local .gguf file or a HuggingFace repo ID.",
        )

    return name
