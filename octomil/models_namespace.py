"""Models namespace -- model lifecycle operations (SDK Facade Contract).

**Tier: Core Contract (MUST)**

Provides the ``client.models`` sub-API with ``status()``, ``load()``,
``unload()``, ``list()``, and ``clear_cache()`` matching the facade
contract across all Octomil SDKs.

Cache status detection aggregates multiple backends:
- ``~/.octomil/models/`` (cactus, executorch, samsung, ort)
- HuggingFace Hub cache (mlx-lm, llama.cpp)
- Ollama local cache (``~/.ollama/models/``)
- In-memory loaded models on the parent ``OctomilClient``
"""

from __future__ import annotations

import logging
import os
import platform
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .client import OctomilClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default cache directory for models pulled via the Octomil registry
# ---------------------------------------------------------------------------
_OCTOMIL_MODELS_DIR = Path.home() / ".octomil" / "models"

# Model file extensions recognised as valid cached model files
_MODEL_EXTENSIONS = {
    ".safetensors",
    ".gguf",
    ".pt",
    ".pth",
    ".bin",
    ".onnx",
    ".pte",
    ".mnn",
    ".tflite",
    ".mlmodel",
    ".mlpackage",
    ".nnpackage",
}


class ModelStatus(str, Enum):
    """Cache status for a model."""

    NOT_CACHED = "not_cached"
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"


class OctomilModels:
    """Model lifecycle operations.

    Matches the ``models`` namespace in SDK_FACADE_CONTRACT.md:

    - ``load(model_id, version?) -> Model``
    - ``status(model_id) -> ModelStatus``
    - ``unload(model_id) -> None``
    - ``list() -> [CachedModel]``
    - ``clear_cache() -> None``
    """

    def __init__(self, client: OctomilClient) -> None:
        """Create the models namespace, backed by a parent OctomilClient."""
        self._client = client
        self._downloading: set[str] = set()
        self._errors: dict[str, str] = {}

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def status(self, model_id: str) -> ModelStatus:
        """Get cache status for a model.

        Checks multiple layers in order:

        1. In-flight downloads tracked by this namespace -> DOWNLOADING
        2. Error state from a previously failed download -> ERROR
        3. Already loaded in ``OctomilClient._models`` -> READY
        4. Present in ``~/.octomil/models/`` -> READY
        5. Present in HuggingFace Hub cache -> READY
        6. Present in Ollama local cache -> READY
        7. Otherwise -> NOT_CACHED
        """
        if model_id in self._downloading:
            return ModelStatus.DOWNLOADING

        if model_id in self._errors:
            return ModelStatus.ERROR

        # Already loaded in memory on the client
        if model_id in self._client._models:
            return ModelStatus.READY

        # Check ~/.octomil/models/<model_id> (cactus, executorch, ort, samsung)
        if self._check_octomil_cache(model_id):
            return ModelStatus.READY

        # Check HuggingFace Hub cache
        if self._check_hf_cache(model_id):
            return ModelStatus.READY

        # Check Ollama local cache
        if self._check_ollama_cache(model_id):
            return ModelStatus.READY

        return ModelStatus.NOT_CACHED

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    def load(self, model_id: str, *, version: Optional[str] = None) -> Any:
        """Download (if needed) and return a loaded Model.

        Delegates to ``OctomilClient.load_model()``.  Tracks download
        state so ``status()`` returns DOWNLOADING while in progress and
        ERROR if the load fails.
        """
        self._downloading.add(model_id)
        self._errors.pop(model_id, None)
        try:
            model = self._client.load_model(model_id, version=version)
        except Exception as exc:
            self._downloading.discard(model_id)
            self._errors[model_id] = str(exc)
            raise
        self._downloading.discard(model_id)
        return model

    # ------------------------------------------------------------------
    # unload
    # ------------------------------------------------------------------

    def unload(self, model_id: str) -> None:
        """Release runtime memory for a model.

        Removes the model from the parent client's in-memory cache.
        Does *not* delete files from disk -- use ``clear_cache()`` for that.
        """
        self._client._models.pop(model_id, None)

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        """List locally cached models.

        Aggregates models from:
        - The parent client's in-memory loaded models
        - Models in ``~/.octomil/models/``
        """
        result: list[dict[str, Any]] = []
        seen: set[str] = set()

        # In-memory loaded models
        for name, model in self._client._models.items():
            seen.add(name)
            result.append(
                {
                    "model_id": name,
                    "status": "loaded",
                    "version": getattr(model.metadata, "version", None),
                }
            )

        # On-disk models in ~/.octomil/models/
        if _OCTOMIL_MODELS_DIR.is_dir():
            for entry in sorted(_OCTOMIL_MODELS_DIR.iterdir()):
                if entry.name in seen:
                    continue
                if entry.is_dir() or entry.suffix in _MODEL_EXTENSIONS:
                    seen.add(entry.name)
                    result.append(
                        {
                            "model_id": entry.name,
                            "status": "cached",
                            "path": str(entry),
                        }
                    )

        return result

    # ------------------------------------------------------------------
    # clear_cache
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Remove all cached model files from ``~/.octomil/models/``.

        Also clears the in-memory model cache and resets error state.
        """
        import shutil

        self._client._models.clear()
        self._errors.clear()

        if _OCTOMIL_MODELS_DIR.is_dir():
            for entry in _OCTOMIL_MODELS_DIR.iterdir():
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                except OSError:
                    logger.warning("Failed to remove cached model: %s", entry)

    # ------------------------------------------------------------------
    # Internal cache probes
    # ------------------------------------------------------------------

    @staticmethod
    def _check_octomil_cache(model_id: str) -> bool:
        """Check if a model exists in ``~/.octomil/models/``."""
        base = _OCTOMIL_MODELS_DIR / model_id
        if base.is_dir():
            return True
        # Check with common extensions
        for ext in _MODEL_EXTENSIONS:
            if base.with_suffix(ext).is_file():
                return True
        # Also check a sanitized name (HF repo "org/name" -> "name")
        if "/" in model_id:
            local_name = model_id.split("/")[-1].lower()
            local_base = _OCTOMIL_MODELS_DIR / local_name
            if local_base.is_dir():
                return True
            for ext in _MODEL_EXTENSIONS:
                if local_base.with_suffix(ext).is_file():
                    return True
        return False

    @staticmethod
    def _check_hf_cache(model_id: str) -> bool:
        """Check if a model is in the HuggingFace Hub cache."""
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()
            for repo_info in cache_info.repos:
                if repo_info.repo_id == model_id:
                    return True
        except Exception:
            pass

        # Also try the model resolver's catalog to map short names
        # to HF repos, then check those
        try:
            from .models.resolver import resolve as _resolve

            resolved = _resolve(model_id)
            if resolved.hf_repo and resolved.hf_repo != model_id:
                from huggingface_hub import scan_cache_dir

                cache_info = scan_cache_dir()
                for repo_info in cache_info.repos:
                    if repo_info.repo_id == resolved.hf_repo:
                        return True
        except Exception:
            pass

        return False

    @staticmethod
    def _check_ollama_cache(model_id: str) -> bool:
        """Check if a model is available in the Ollama local cache."""
        system = platform.system()
        if system == "Windows":
            models_dir = os.path.join(os.environ.get("USERPROFILE", ""), ".ollama", "models")
        else:
            models_dir = os.path.expanduser("~/.ollama/models")

        # Parse ollama-style ref: "gemma3:4b" -> model="gemma3", tag="4b"
        if ":" in model_id:
            model, tag = model_id.split(":", 1)
        else:
            model, tag = model_id, "latest"

        manifest_path = os.path.join(
            models_dir,
            "manifests",
            "registry.ollama.ai",
            "library",
            model,
            tag,
        )
        return os.path.isfile(manifest_path)
