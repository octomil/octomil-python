"""Base class for model source backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SourceResult:
    """Result from a source backend resolve/download operation."""

    path: str  # local file path or repo ID
    source_type: str  # "huggingface", "ollama", "kaggle"
    cached: bool = False  # whether the model was served from local cache
    repo_id: Optional[str] = None  # HuggingFace repo ID if applicable
    filename: Optional[str] = None  # specific file within repo


class SourceBackend:
    """Base class for source backends.

    Subclasses implement ``is_available()``, ``resolve()``, and optionally
    ``check_cache()``.
    """

    name: str = "base"

    def is_available(self) -> bool:
        """Check whether this source backend is usable (deps installed, etc)."""
        raise NotImplementedError

    def check_cache(self, ref: str, filename: Optional[str] = None) -> Optional[str]:
        """Check if a model is available in local cache.

        Returns the local path if cached, None otherwise.
        """
        return None

    def resolve(
        self,
        ref: str,
        filename: Optional[str] = None,
    ) -> SourceResult:
        """Resolve and optionally download a model.

        Parameters
        ----------
        ref:
            The model reference (repo ID, ollama name, etc).
        filename:
            Specific file within the reference (for GGUF files in HF repos).

        Returns
        -------
        SourceResult
            Contains the local path and metadata about the resolution.
        """
        raise NotImplementedError
