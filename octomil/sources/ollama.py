"""Ollama source backend.

Resolves models from the local Ollama cache. The Ollama manifest format:
``~/.ollama/models/manifests/registry.ollama.ai/library/{model}/{tag}``
contains JSON with a digest pointing to a blob at
``~/.ollama/models/blobs/sha256-{hash}``.

If the model is cached locally, returns the blob path directly.
If not cached, falls back to ``ollama pull`` CLI if available.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
from typing import Optional

from .base import SourceBackend, SourceResult

logger = logging.getLogger(__name__)


def _ollama_models_dir() -> str:
    """Return the default Ollama models directory for the current platform."""
    system = platform.system()
    if system == "Windows":
        return os.path.join(os.environ.get("USERPROFILE", ""), ".ollama", "models")
    return os.path.expanduser("~/.ollama/models")


def _parse_ollama_ref(ref: str) -> tuple[str, str]:
    """Parse an Ollama ref like ``gemma3:4b`` into (model, tag).

    If no tag is specified, defaults to ``"latest"``.
    """
    if ":" in ref:
        model, tag = ref.split(":", 1)
    else:
        model, tag = ref, "latest"
    return model, tag


class OllamaSource(SourceBackend):
    """Resolve models from local Ollama cache or via ``ollama pull``."""

    name = "ollama"

    def __init__(self, models_dir: Optional[str] = None) -> None:
        self._models_dir = models_dir or _ollama_models_dir()

    def is_available(self) -> bool:
        """Check if Ollama CLI is installed."""
        return shutil.which("ollama") is not None

    def _manifest_path(self, model: str, tag: str) -> str:
        """Return the path to the Ollama manifest file."""
        return os.path.join(
            self._models_dir,
            "manifests",
            "registry.ollama.ai",
            "library",
            model,
            tag,
        )

    def _resolve_blob_from_manifest(self, manifest_path: str) -> Optional[str]:
        """Read an Ollama manifest and return the path to the model blob.

        The manifest is JSON with a ``layers`` array. The model weights
        layer has ``mediaType`` containing ``"model"`` and a ``digest``
        field like ``"sha256:<hex>"``.
        """
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Failed to read Ollama manifest: %s", exc)
            return None

        layers = manifest.get("layers", [])
        for layer in layers:
            media_type = layer.get("mediaType", "")
            if "model" in media_type:
                digest = layer.get("digest", "")
                if digest:
                    # Ollama stores blobs as "sha256-<hex>" (dash, not colon)
                    blob_name = digest.replace(":", "-")
                    blob_path = os.path.join(self._models_dir, "blobs", blob_name)
                    if os.path.exists(blob_path):
                        return blob_path

        return None

    def check_cache(self, ref: str, filename: Optional[str] = None) -> Optional[str]:
        """Check if a model is available in the local Ollama cache.

        Parameters
        ----------
        ref:
            Ollama model reference (e.g. ``"gemma3:4b"``).

        Returns the local blob path if cached, None otherwise.
        """
        model, tag = _parse_ollama_ref(ref)
        manifest_path = self._manifest_path(model, tag)

        if not os.path.exists(manifest_path):
            return None

        return self._resolve_blob_from_manifest(manifest_path)

    def resolve(
        self,
        ref: str,
        filename: Optional[str] = None,
    ) -> SourceResult:
        """Resolve an Ollama model to a local blob path.

        First checks the local cache. If not found and the ``ollama`` CLI
        is available, runs ``ollama pull`` to download the model.

        Parameters
        ----------
        ref:
            Ollama model reference (e.g. ``"gemma3:4b"``).
        """
        # Check local cache first
        cached = self.check_cache(ref)
        if cached:
            logger.info("Ollama cache hit: %s -> %s", ref, cached)
            return SourceResult(
                path=cached,
                source_type="ollama",
                cached=True,
            )

        # Try to pull via CLI
        if not self.is_available():
            raise RuntimeError(
                f"Ollama model '{ref}' not found in local cache and "
                f"ollama CLI is not installed. Install ollama: https://ollama.com"
            )

        logger.info("Pulling %s via ollama CLI...", ref)
        try:
            subprocess.run(
                ["ollama", "pull", ref],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to pull '{ref}' via ollama: {exc.stderr.strip()}"
            ) from exc

        # Check cache again after pull
        cached = self.check_cache(ref)
        if cached:
            return SourceResult(
                path=cached,
                source_type="ollama",
                cached=False,  # freshly downloaded
            )

        raise RuntimeError(
            f"ollama pull '{ref}' succeeded but blob not found in cache. "
            f"This may indicate a non-standard Ollama installation."
        )
