"""Source backends for downloading models from various providers.

Phase 1 provides:
- HuggingFaceSource — download from HuggingFace Hub
- OllamaSource — resolve from local Ollama cache
- KaggleSource — download via Kaggle CLI (stub)
"""

from __future__ import annotations

from .base import SourceBackend, SourceResult
from .huggingface import HuggingFaceSource
from .ollama import OllamaSource
from .kaggle import KaggleSource

__all__ = [
    "SourceBackend",
    "SourceResult",
    "HuggingFaceSource",
    "OllamaSource",
    "KaggleSource",
]
