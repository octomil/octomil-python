"""Unified model resolution across Ollama, HuggingFace, and Kaggle.

Resolves human-friendly model names to local file paths, downloading
from the best available source if needed.

Supports:
- Explicit sources: ``hf:org/model``, ``ollama:name:tag``, ``kaggle:org/model``
- Known aliases: ``phi-4-mini``, ``gemma-1b``, ``llama-3.2-3b``
- Direct HuggingFace repo IDs: ``microsoft/Phi-4-mini-instruct``
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import click

from .base import SourceResult
from .huggingface import HuggingFaceSource
from .kaggle import KaggleSource
from .ollama import OllamaSource

logger = logging.getLogger(__name__)

# ── Known model aliases ──────────────────────────────────────────────────────
# Maps common short names to source-specific identifiers.

_MODEL_ALIASES: Dict[str, Dict[str, str]] = {
    # Phi family
    "phi-4-mini": {"hf": "microsoft/Phi-4-mini-instruct", "ollama": "phi4-mini"},
    "phi-mini": {"hf": "microsoft/Phi-3.5-mini-instruct", "ollama": "phi3.5"},
    # Gemma family
    "gemma-1b": {"hf": "google/gemma-2-2b-it", "ollama": "gemma2:2b"},
    "gemma-3b": {"hf": "google/gemma-3-4b-it", "ollama": "gemma3:4b"},
    "gemma-4b": {"hf": "google/gemma-3-4b-it", "ollama": "gemma3:4b"},
    # Llama family
    "llama-3.2-1b": {"hf": "meta-llama/Llama-3.2-1B-Instruct", "ollama": "llama3.2:1b"},
    "llama-3.2-3b": {"hf": "meta-llama/Llama-3.2-3B-Instruct", "ollama": "llama3.2:3b"},
    # Qwen family
    "qwen3": {"hf": "Qwen/Qwen3-4B", "ollama": "qwen3:4b"},
    "qwen3-1b": {"hf": "Qwen/Qwen3-1.7B", "ollama": "qwen3:1.7b"},
    # SmolLM
    "smollm-360m": {"hf": "HuggingFaceTB/SmolLM2-360M-Instruct", "ollama": "smollm2:360m"},
    "smollm-1.7b": {"hf": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "ollama": "smollm2:1.7b"},
    # DeepSeek
    "deepseek-coder-v2": {"hf": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "ollama": "deepseek-coder-v2"},
    # Whisper (HF only)
    "whisper-tiny": {"hf": "openai/whisper-tiny"},
    "whisper-base": {"hf": "openai/whisper-base"},
    "whisper-small": {"hf": "openai/whisper-small"},
    "whisper-medium": {"hf": "openai/whisper-medium"},
    "whisper-large-v3": {"hf": "openai/whisper-large-v3"},
    # Codestral
    "codestral": {"hf": "mistralai/Codestral-22B-v0.1", "ollama": "codestral"},
}

_ollama = OllamaSource()
_hf = HuggingFaceSource()
_kaggle = KaggleSource()


def _parse_explicit_source(name: str) -> Optional[tuple[str, str]]:
    """Parse ``hf:org/model`` → ``("hf", "org/model")``. Returns None if no prefix."""
    for prefix in ("hf:", "huggingface:", "ollama:", "kaggle:"):
        if name.startswith(prefix):
            source = prefix.rstrip(":")
            if source == "huggingface":
                source = "hf"
            return source, name[len(prefix):]
    return None


def _try_source(source: str, ref: str) -> Optional[SourceResult]:
    """Attempt a single source. Returns None on failure."""
    try:
        if source == "ollama":
            if not _ollama.is_available():
                return None
            return _ollama.resolve(ref)
        elif source == "hf":
            if not _hf.is_available():
                return None
            return _hf.resolve(ref)
        elif source == "kaggle":
            if not _kaggle.is_available():
                return None
            return _kaggle.resolve(ref)
    except Exception as exc:
        logger.debug("Source %s failed for %s: %s", source, ref, exc)
    return None


def resolve_and_download(name: str) -> str:
    """Resolve a model name and return a local file path.

    Parameters
    ----------
    name:
        Model name, alias, or explicit source reference.

    Returns
    -------
    str
        Local path to the downloaded model.

    Raises
    ------
    RuntimeError
        If the model cannot be resolved from any source.
    """
    # ── Explicit source ───────────────────────────────────────────────────
    explicit = _parse_explicit_source(name)
    if explicit:
        source, ref = explicit
        click.echo(f"  Downloading from {source}: {ref}")
        result = _try_source(source, ref)
        if result:
            if result.cached:
                click.echo(f"  Using cache: {result.path}")
            return result.path
        raise RuntimeError(f"Could not download '{ref}' from {source}")

    # ── Alias lookup ──────────────────────────────────────────────────────
    aliases = _MODEL_ALIASES.get(name)
    if aliases:
        # Try Ollama first (fastest — local cache), then HuggingFace
        if "ollama" in aliases:
            click.echo(f"  Checking Ollama for {aliases['ollama']}...")
            result = _try_source("ollama", aliases["ollama"])
            if result:
                if result.cached:
                    click.echo(f"  Using Ollama cache")
                else:
                    click.echo(f"  Downloaded from Ollama")
                return result.path

        if "hf" in aliases:
            click.echo(f"  Downloading from HuggingFace: {aliases['hf']}")
            result = _try_source("hf", aliases["hf"])
            if result:
                return result.path

        if "kaggle" in aliases:
            result = _try_source("kaggle", aliases["kaggle"])
            if result:
                return result.path

        raise RuntimeError(
            f"Could not download '{name}' from any source. "
            f"Tried: {', '.join(aliases.keys())}"
        )

    # ── Direct HuggingFace repo ID (org/model format) ─────────────────────
    if "/" in name:
        click.echo(f"  Downloading from HuggingFace: {name}")
        result = _try_source("hf", name)
        if result:
            return result.path
        raise RuntimeError(f"Could not download '{name}' from HuggingFace")

    # ── Unknown ───────────────────────────────────────────────────────────
    known = ", ".join(sorted(_MODEL_ALIASES.keys()))
    raise RuntimeError(
        f"Unknown model: '{name}'\n"
        f"  Known models: {known}\n"
        f"  Or use: hf:<org>/<model>, ollama:<name>, kaggle:<path>"
    )
