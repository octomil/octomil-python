"""Ollama bridge — detect and resolve local ollama models for EdgeML deployment."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"

# Map ollama quantization names → EdgeML equivalents
QUANT_MAP: dict[str, str] = {
    "Q2_K": "INT2",
    "Q3_K_S": "INT3",
    "Q3_K_M": "INT3",
    "Q3_K_L": "INT3",
    "Q4_0": "INT4",
    "Q4_1": "INT4",
    "Q4_K_S": "INT4",
    "Q4_K_M": "INT4",
    "Q5_0": "INT5",
    "Q5_1": "INT5",
    "Q5_K_S": "INT5",
    "Q5_K_M": "INT5",
    "Q6_K": "INT6",
    "Q8_0": "INT8",
    "F16": "FP16",
    "F32": "FP32",
}


def _ollama_models_dir() -> str:
    """Return the default ollama models directory for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return os.path.expanduser("~/.ollama/models")
    if system == "Linux":
        return os.path.expanduser("~/.ollama/models")
    if system == "Windows":
        return os.path.join(os.environ.get("USERPROFILE", ""), ".ollama", "models")
    return os.path.expanduser("~/.ollama/models")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class OllamaModel:
    """Represents a locally available ollama model."""

    name: str
    size: int  # bytes
    family: str  # e.g. "gemma", "llama", "phi"
    quantization: str  # e.g. "Q4_K_M"
    parameter_size: str  # e.g. "2B", "7B"
    modified_at: str
    digest: str
    gguf_path: Optional[str] = field(default=None)

    @property
    def edgeml_quantization(self) -> str:
        """Map ollama quantization to EdgeML equivalent."""
        return QUANT_MAP.get(self.quantization, self.quantization)

    @property
    def size_display(self) -> str:
        """Human-readable size string."""
        gb = self.size / (1024**3)
        if gb >= 1.0:
            return f"{gb:.1f} GB"
        mb = self.size / (1024**2)
        return f"{mb:.0f} MB"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_ollama_running(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Check whether the ollama server is reachable."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return False


def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> list[OllamaModel]:
    """Fetch all models from the local ollama instance."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, OSError):
        return []

    models: list[OllamaModel] = []
    for entry in resp.json().get("models", []):
        details = entry.get("details", {})
        model = OllamaModel(
            name=entry.get("name", ""),
            size=entry.get("size", 0),
            family=details.get("family", "unknown"),
            quantization=details.get("quantization_level", "unknown"),
            parameter_size=details.get("parameter_size", "unknown"),
            modified_at=entry.get("modified_at", ""),
            digest=entry.get("digest", ""),
        )
        model.gguf_path = resolve_gguf_path(model)
        models.append(model)

    return models


def get_ollama_model(
    name: str, base_url: str = OLLAMA_BASE_URL
) -> Optional[OllamaModel]:
    """Get a specific ollama model by name. Returns None if not found."""
    models = list_ollama_models(base_url=base_url)
    for m in models:
        if m.name == name or m.name.split(":")[0] == name.split(":")[0]:
            # Match full name (gemma:2b) or base name (gemma)
            if ":" not in name or m.name == name:
                return m
    return None


def resolve_gguf_path(model: OllamaModel) -> Optional[str]:
    """Resolve the local GGUF blob path for an ollama model.

    Ollama stores model blobs under ~/.ollama/models/blobs/ using the
    digest as the filename (sha256-<hex>).
    """
    if not model.digest:
        return None

    models_dir = _ollama_models_dir()
    blob_dir = os.path.join(models_dir, "blobs")

    # The digest from the API is usually "sha256:<hex>". Ollama stores
    # blobs as "sha256-<hex>" (dash instead of colon).
    blob_name = model.digest.replace(":", "-")
    blob_path = os.path.join(blob_dir, blob_name)

    if os.path.exists(blob_path):
        return blob_path

    return None


def map_quantization(ollama_quant: str) -> str:
    """Map an ollama quantization name to the EdgeML equivalent."""
    return QUANT_MAP.get(ollama_quant, ollama_quant)
