"""Cactus engine plugin -- llama.cpp mobile wrapper for on-device inference.

Cactus wraps llama.cpp with a simplified C FFI layer, providing native bindings
for Swift, Kotlin, Flutter, and Python without manual JNI/C++ bridging. Models
are converted from HuggingFace to Cactus's own weight format (quantized tensors
with a config.txt manifest) via ``cactus download`` or ``cactus convert``.

Key capabilities beyond raw llama.cpp:
- Cloud handoff detection (confidence-based routing to cloud APIs)
- Built-in tool calling / function calling
- Auto-RAG with vector index
- Audio transcription (Whisper models)
- Text, image, and audio embeddings
- Voice Activity Detection (VAD)
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass
from typing import Any, Optional

from .base import BenchmarkResult, EnginePlugin

logger = logging.getLogger(__name__)

# Models known to work with llama.cpp (and therefore Cactus) from the catalog.
from ..models.catalog import CATALOG as _UNIFIED_CATALOG

_CACTUS_CATALOG = {
    name
    for name, entry in _UNIFIED_CATALOG.items()
    if "llama.cpp" in entry.engines
}


@dataclass
class CactusMetrics:
    """Inference metrics returned alongside generated text."""

    total_tokens: int = 0
    tokens_per_second: float = 0.0
    ttfc_ms: float = 0.0


def _try_import_cactus() -> Optional[Any]:
    """Attempt to import the cactus Python FFI module.

    Returns the cactus module if available, None otherwise.
    """
    try:
        # The cactus package ships as ``cactus`` with FFI bindings that load
        # libcactus.dylib / libcactus.so at import time.
        from cactus import cactus as _cactus_ffi  # type: ignore[import-untyped]

        return _cactus_ffi
    except ImportError:
        pass

    # Fallback: cactus may be installed as a top-level module directly
    # exposing cactus_init / cactus_complete etc.
    try:
        import cactus  # type: ignore[import-untyped]

        if hasattr(cactus, "cactus_init"):
            return cactus
    except (ImportError, RuntimeError):
        # RuntimeError can happen when libcactus shared library is missing
        pass

    return None


class CactusEngine(EnginePlugin):
    """On-device inference engine using Cactus (llama.cpp mobile wrapper)."""

    @property
    def name(self) -> str:
        return "cactus"

    @property
    def display_name(self) -> str:
        sys = platform.system()
        machine = platform.machine()
        if sys == "Darwin":
            accel = "Metal"
        else:
            accel = "CPU"
        return f"Cactus ({accel}, {machine})"

    @property
    def priority(self) -> int:
        # Between llama.cpp (20) and ExecuTorch (25).
        # Cactus adds mobile-specific features on top of llama.cpp but
        # raw throughput on desktop is comparable.
        return 22

    def detect(self) -> bool:
        return _try_import_cactus() is not None

    def detect_info(self) -> str:
        sys = platform.system()
        if sys == "Darwin":
            return "Cactus FFI (CPU + Metal)"
        return "Cactus FFI (CPU)"

    def supports_model(self, model_name: str) -> bool:
        """Check if this engine can serve the given model.

        Cactus supports the same model families as llama.cpp (converted to
        its own weight format), plus any local directory that contains a
        ``config.txt`` manifest, and HuggingFace repo IDs.
        """
        import os

        # Catalog models supported by llama.cpp are also Cactus-compatible
        if model_name in _CACTUS_CATALOG:
            return True

        # Local directory with Cactus weight format
        if os.path.isdir(model_name) and os.path.isfile(
            os.path.join(model_name, "config.txt")
        ):
            return True

        # GGUF files (can be used with underlying llama.cpp)
        if model_name.endswith(".gguf"):
            return True

        # HuggingFace repo IDs
        if "/" in model_name:
            return True

        return False

    def benchmark(self, model_name: str, n_tokens: int = 32) -> BenchmarkResult:
        """Run a quick inference benchmark via Cactus FFI.

        Loads the model, generates ``n_tokens`` via chat completion, and
        measures throughput from the Cactus response metrics.
        """
        cactus = _try_import_cactus()
        if cactus is None:
            return BenchmarkResult(
                engine_name=self.name,
                error="cactus package not available",
            )

        try:
            model_path = self._resolve_model_path(model_name)
            handle = cactus.cactus_init(model_path)
            if handle is None:
                err = cactus.cactus_get_last_error()
                return BenchmarkResult(
                    engine_name=self.name,
                    error=f"cactus_init failed: {err}",
                )

            messages = json.dumps(
                [{"role": "user", "content": "Hello, how are you?"}]
            )

            try:
                start = time.monotonic()
                response_json = cactus.cactus_complete(
                    handle, messages, max_tokens=n_tokens
                )
                elapsed = time.monotonic() - start
            finally:
                cactus.cactus_destroy(handle)

            # Parse Cactus response for metrics
            try:
                resp = json.loads(response_json)
            except (json.JSONDecodeError, TypeError):
                resp = {}

            if not resp.get("success", False):
                return BenchmarkResult(
                    engine_name=self.name,
                    error=resp.get("error", "generation failed"),
                )

            decode_tps = resp.get("decode_tps", 0.0)
            ttft = resp.get("time_to_first_token_ms", 0.0)
            ram = resp.get("ram_usage_mb", 0.0)
            total_tokens = resp.get("total_tokens", 0)

            # Fallback throughput calculation
            if decode_tps == 0 and total_tokens > 0 and elapsed > 0:
                decode_tps = total_tokens / elapsed

            return BenchmarkResult(
                engine_name=self.name,
                tokens_per_second=decode_tps,
                ttft_ms=ttft,
                memory_mb=ram,
                metadata={
                    "prefill_tps": resp.get("prefill_tps", 0.0),
                    "confidence": resp.get("confidence", 0.0),
                    "cloud_handoff": resp.get("cloud_handoff", False),
                },
            )
        except Exception as exc:
            return BenchmarkResult(engine_name=self.name, error=str(exc))

    def create_backend(self, model_name: str, **kwargs: Any) -> Any:
        """Create a CactusBackend that wraps Cactus FFI inference."""
        return _CactusBackend(model_name, **kwargs)

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve a model name to a Cactus weights directory path.

        Cactus expects a directory containing config.txt and weight files,
        typically created by ``cactus download <model_id>``.
        """
        import os

        # Already a valid Cactus weights directory
        if os.path.isdir(model_name) and os.path.isfile(
            os.path.join(model_name, "config.txt")
        ):
            return model_name

        # Check ~/.edgeml/models/<name>
        home_path = os.path.expanduser(f"~/.edgeml/models/{model_name}")
        if os.path.isdir(home_path) and os.path.isfile(
            os.path.join(home_path, "config.txt")
        ):
            return home_path

        # For HF repo IDs, derive the local name
        if "/" in model_name:
            local_name = model_name.split("/")[-1].lower()
            home_path = os.path.expanduser(f"~/.edgeml/models/{local_name}")
            if os.path.isdir(home_path) and os.path.isfile(
                os.path.join(home_path, "config.txt")
            ):
                return home_path

        # Return as-is and let cactus_init report the error
        return model_name


class _CactusBackend:
    """Thin wrapper around Cactus FFI for the InferenceBackend interface."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self._handle: Any = None
        self._cactus: Any = None
        self._kwargs = kwargs

    def load_model(self, model_name: str) -> None:
        cactus = _try_import_cactus()
        if cactus is None:
            raise ImportError(
                "cactus package is not installed. "
                "Build Cactus with: cactus build --python"
            )
        self._cactus = cactus

        engine = CactusEngine()
        model_path = engine._resolve_model_path(model_name)

        handle = cactus.cactus_init(model_path)
        if handle is None:
            err = cactus.cactus_get_last_error()
            raise RuntimeError(f"cactus_init failed: {err}")

        self._handle = handle
        self.model_name = model_name

    def generate(self, request: Any) -> tuple[str, Any]:
        if self._handle is None:
            self.load_model(self.model_name)

        messages = request.messages if hasattr(request, "messages") else []
        max_tokens = getattr(request, "max_tokens", 512)

        messages_json = json.dumps(messages)

        start = time.monotonic()
        response_json = self._cactus.cactus_complete(
            self._handle, messages_json, max_tokens=max_tokens
        )
        elapsed = time.monotonic() - start

        try:
            resp = json.loads(response_json)
        except (json.JSONDecodeError, TypeError):
            resp = {"success": False, "error": "invalid response"}

        text = resp.get("response", "") or ""
        total_tokens = resp.get("total_tokens", 0)
        tps = resp.get("decode_tps", 0.0)
        ttft = resp.get("time_to_first_token_ms", 0.0)

        if tps == 0 and total_tokens > 0 and elapsed > 0:
            tps = total_tokens / elapsed

        return text, CactusMetrics(
            total_tokens=total_tokens,
            tokens_per_second=tps,
            ttfc_ms=ttft,
        )

    def list_models(self) -> list[str]:
        return [self.model_name] if self.model_name else []

    def __del__(self) -> None:
        if self._handle is not None and self._cactus is not None:
            try:
                self._cactus.cactus_destroy(self._handle)
            except Exception:
                pass
