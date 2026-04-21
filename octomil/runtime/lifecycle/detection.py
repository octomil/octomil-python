"""Runtime detection — discover locally installed inference engines.

Checks for known engines (mlx-lm, llama.cpp, etc.) by probing imports
and PATH, returning structured results for the lifecycle planner.
"""

from __future__ import annotations

import importlib.util
import logging
import platform
import shutil
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstalledRuntime:
    """A locally installed inference engine."""

    engine_id: str
    version: str | None = None
    path: str | None = None
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def display(self) -> str:
        """Human-readable summary."""
        parts = [self.engine_id]
        if self.version:
            parts.append(f"v{self.version}")
        if self.path:
            parts.append(f"@ {self.path}")
        return " ".join(parts)


def detect_installed_runtimes() -> list[InstalledRuntime]:
    """Detect locally installed inference engines.

    Currently checks for:
    - mlx-lm (Python package, Apple Silicon only)
    - llama.cpp via llama-cpp-python (Python bindings)
    - llama.cpp CLI binary on PATH
    - onnxruntime (Python package)
    - ollama CLI on PATH

    Returns a list of InstalledRuntime entries. Empty list means no engines found.
    """
    results: list[InstalledRuntime] = []

    # --- mlx-lm (Apple Silicon only) ---
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        mlx_result = _detect_mlx()
        if mlx_result:
            results.append(mlx_result)

    # --- llama-cpp-python (pip package) ---
    llamacpp_result = _detect_llamacpp_python()
    if llamacpp_result:
        results.append(llamacpp_result)

    # --- llama.cpp CLI binary ---
    llamacpp_cli = _detect_llamacpp_cli()
    if llamacpp_cli:
        results.append(llamacpp_cli)

    # --- onnxruntime ---
    ort_result = _detect_onnxruntime()
    if ort_result:
        results.append(ort_result)

    # --- ollama CLI ---
    ollama_result = _detect_ollama()
    if ollama_result:
        results.append(ollama_result)

    return results


def _detect_mlx() -> InstalledRuntime | None:
    """Check for mlx-lm Python package."""
    spec = importlib.util.find_spec("mlx_lm")
    if spec is None:
        return None
    version = _get_package_version("mlx_lm")
    return InstalledRuntime(engine_id="mlx-lm", version=version)


def _detect_llamacpp_python() -> InstalledRuntime | None:
    """Check for llama-cpp-python bindings."""
    spec = importlib.util.find_spec("llama_cpp")
    if spec is None:
        return None
    version = _get_package_version("llama_cpp")
    return InstalledRuntime(engine_id="llama.cpp", version=version, extras={"binding": "python"})


def _detect_llamacpp_cli() -> InstalledRuntime | None:
    """Check for llama.cpp CLI binaries on PATH."""
    # Common binary names across versions
    candidates = ["llama-cli", "llama-server", "main"]
    for name in candidates:
        path = shutil.which(name)
        if path:
            return InstalledRuntime(
                engine_id="llama.cpp",
                path=path,
                extras={"binding": "cli", "binary": name},
            )
    return None


def _detect_onnxruntime() -> InstalledRuntime | None:
    """Check for ONNX Runtime."""
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None:
        return None
    version = _get_package_version("onnxruntime")
    return InstalledRuntime(engine_id="onnxruntime", version=version)


def _detect_ollama() -> InstalledRuntime | None:
    """Check for ollama CLI on PATH."""
    path = shutil.which("ollama")
    if path is None:
        return None
    return InstalledRuntime(engine_id="ollama", path=path)


def _get_package_version(module_name: str) -> str | None:
    """Best-effort version extraction for a Python package."""
    try:
        from importlib.metadata import version

        return version(module_name.replace("_", "-"))
    except Exception:
        pass
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None
