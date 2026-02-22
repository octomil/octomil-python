"""Model name parser — splits ``model:variant`` into structured components.

Supports Ollama-style specifiers::

    gemma-3b:4bit   -> family=gemma-3b, variant=4bit
    phi-mini        -> family=phi-mini, variant=None (uses default)
    llama-8b:fp16   -> family=llama-8b, variant=fp16
    gemma-3b:q4_k_m -> family=gemma-3b, variant=q4_k_m (engine-specific)
    user/repo       -> passthrough (not parsed)
    ./model.gguf    -> passthrough (local file)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ParsedModel:
    """Result of parsing a ``model:variant`` string."""

    family: Optional[str]
    variant: Optional[str]
    raw: str
    is_passthrough: bool = False

    @property
    def is_local_file(self) -> bool:
        return self.raw.endswith((".gguf", ".pte", ".mnn"))


# User-friendly quant aliases -> canonical internal tag.
# Each engine maps these canonical tags to its own format in the catalog.
QUANT_ALIASES: dict[str, str] = {
    # User-friendly names
    "4bit": "4bit",
    "8bit": "8bit",
    "fp16": "fp16",
    "f16": "fp16",
    "16bit": "fp16",
    "default": "4bit",
    # Engine-specific GGUF names -> canonical
    "q4_k_m": "4bit",
    "q4_k_s": "4bit",
    "q8_0": "8bit",
    "q8_1": "8bit",
    "f16": "fp16",
    # Engine-specific MLX names -> canonical
    "float16": "fp16",
}


def normalize_variant(variant: str) -> str:
    """Normalize a variant string to a canonical quant level.

    Returns the canonical form (``4bit``, ``8bit``, ``fp16``),
    or the original string if no alias is found (allows engine-specific
    pass-through like ``q4_k_s``).
    """
    return QUANT_ALIASES.get(variant.lower(), variant.lower())


def parse(name: str) -> ParsedModel:
    """Parse a model specifier into structured components.

    Parameters
    ----------
    name:
        Model specifier string. Can be:
        - Short name: ``"gemma-3b"``
        - Short name with variant: ``"gemma-3b:4bit"``
        - Full HuggingFace repo: ``"mlx-community/gemma-3-1b-it-4bit"``
        - Local file path: ``"./model.gguf"``

    Returns
    -------
    ParsedModel
        Parsed components. ``is_passthrough`` is True for repo IDs
        and local files that bypass catalog lookup.
    """
    # Local file paths
    if name.endswith((".gguf", ".pte", ".mnn")):
        return ParsedModel(family=None, variant=None, raw=name, is_passthrough=True)

    # Full HuggingFace repo ID (contains /)
    if "/" in name:
        return ParsedModel(family=None, variant=None, raw=name, is_passthrough=True)

    # model:variant syntax
    if ":" in name:
        family, variant = name.split(":", 1)
        return ParsedModel(
            family=family.lower().strip(),
            variant=variant.lower().strip(),
            raw=name,
        )

    # Bare model name — no variant specified
    return ParsedModel(
        family=name.lower().strip(),
        variant=None,
        raw=name,
    )
