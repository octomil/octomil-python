"""Model optimization based on detected hardware.

Computes optimal quantization, GPU offload layers, and runtime
configuration for local LLM inference across engines (edgeml serve,
Ollama, llama.cpp, etc.). Pure computation — no subprocess calls,
no file I/O.
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass
from enum import Enum

from edgeml.hardware import HardwareProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & result dataclasses
# ---------------------------------------------------------------------------


class MemoryStrategy(str, Enum):
    FULL_GPU = "full_gpu"
    PARTIAL_OFFLOAD = "partial_offload"
    CPU_ONLY = "cpu_only"
    AGGRESSIVE_QUANT = "aggressive_quant"


@dataclass(frozen=True)
class QuantOffloadResult:
    quantization: str
    gpu_layers: int  # -1 = all layers on GPU, 0 = none, N = partial
    strategy: MemoryStrategy
    vram_gb: float
    ram_gb: float
    total_gb: float
    warning: str | None = None


@dataclass(frozen=True)
class SpeedEstimate:
    tokens_per_second: float
    backend: str
    strategy: MemoryStrategy
    confidence: str  # "high", "medium", "low"


@dataclass(frozen=True)
class ModelRecommendation:
    model_size: str  # "7B", "13B", etc.
    quantization: str
    reason: str
    config: QuantOffloadResult
    speed: SpeedEstimate
    serve_command: str  # Ready to copy-paste


# ---------------------------------------------------------------------------
# Reference tables — single source of truth
# ---------------------------------------------------------------------------

# Bytes per parameter for each quantization level
_BYTES_PER_PARAM: dict[str, float] = {
    "Q2_K": 0.3125,
    "Q3_K_S": 0.375,
    "Q3_K_M": 0.4375,
    "Q4_0": 0.5,
    "Q4_K_S": 0.5625,
    "Q4_K_M": 0.625,
    "Q5_0": 0.625,
    "Q5_K_S": 0.6875,
    "Q5_K_M": 0.75,
    "Q6_K": 0.8125,
    "Q8_0": 1.0,
    "F16": 2.0,
    "F32": 4.0,
}

# Speed multiplier relative to Q4_K_M baseline (1.0)
_QUANT_SPEED_FACTORS: dict[str, float] = {
    "Q2_K": 1.4,
    "Q3_K_S": 1.25,
    "Q3_K_M": 1.15,
    "Q4_0": 1.1,
    "Q4_K_S": 1.05,
    "Q4_K_M": 1.0,
    "Q5_0": 0.95,
    "Q5_K_S": 0.9,
    "Q5_K_M": 0.85,
    "Q6_K": 0.8,
    "Q8_0": 0.7,
    "F16": 0.5,
    "F32": 0.25,
}

# KV cache memory per 1K context tokens (GB) keyed by model-parameter threshold.
# Look up the largest key <= model_size_b.
_KV_CACHE_PER_1K_TOKENS: dict[int, float] = {
    1: 0.05,
    3: 0.08,
    7: 0.15,
    13: 0.25,
    30: 0.5,
    70: 1.0,
    180: 2.5,
}

# Known model sizes in billions of parameters
_MODEL_SIZES: dict[str, float] = {
    "0.5B": 0.5,
    "1B": 1.0,
    "1.5B": 1.5,
    "3B": 3.0,
    "7B": 7.0,
    "8B": 8.0,
    "13B": 13.0,
    "14B": 14.0,
    "32B": 32.0,
    "34B": 34.0,
    "70B": 70.0,
    "72B": 72.0,
    "180B": 180.0,
    "405B": 405.0,
}

# Quantizations ordered from highest quality to lowest — the optimizer tries
# each in order and picks the best quality that fits in memory.
_QUANT_PREFERENCE_ORDER: list[str] = [
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q5_0",
    "Q4_K_M",
    "Q4_K_S",
    "Q4_0",
    "Q3_K_M",
    "Q3_K_S",
    "Q2_K",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kv_cache_gb(model_size_b: float, context_length: int) -> float:
    """Estimate KV cache memory in GB for a given model size and context."""
    # Find the largest threshold key that does not exceed model_size_b
    per_1k = 0.05  # default for tiny models
    for threshold in sorted(_KV_CACHE_PER_1K_TOKENS):
        if model_size_b >= threshold:
            per_1k = _KV_CACHE_PER_1K_TOKENS[threshold]
    return per_1k * (context_length / 1000.0)


def _model_memory_gb(model_size_b: float, quant: str) -> float:
    """Raw model weight memory in GB (before KV cache)."""
    return model_size_b * _BYTES_PER_PARAM[quant]


def _total_memory_gb(model_size_b: float, quant: str, context_length: int) -> float:
    """Total memory: model weights + KV cache."""
    return _model_memory_gb(model_size_b, quant) + _kv_cache_gb(
        model_size_b, context_length
    )


# ---------------------------------------------------------------------------
# Engine auto-detection
# ---------------------------------------------------------------------------


def _detect_engine() -> str:
    """Auto-detect the best available inference engine on PATH.

    Priority: edgeml > ollama > llama-server. Falls back to "edgeml".
    """
    if shutil.which("edgeml"):
        return "edgeml"
    if shutil.which("ollama"):
        return "ollama"
    if shutil.which("llama-server"):
        return "llama.cpp"
    return "edgeml"


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------


class ModelOptimizer:
    """Compute optimal inference configuration for the detected hardware.

    Engine-agnostic: works with edgeml serve, Ollama, llama.cpp, and other
    GGUF-based backends. All methods are pure computation. The class stores
    a reference to a ``HardwareProfile`` and derives usable VRAM / RAM
    budgets at init.
    """

    def __init__(self, hardware: HardwareProfile) -> None:
        self.hardware = hardware

        # --- Usable VRAM budget ---
        gpu = hardware.gpu
        is_metal = gpu is not None and gpu.backend == "metal"

        if is_metal:
            # Apple Silicon unified memory — treat total RAM as VRAM minus
            # a 4 GB OS reserve.
            self.usable_vram = max(hardware.total_ram_gb - 4.0, 0.0)
        elif gpu is not None and gpu.total_vram_gb > 0:
            # Discrete GPU(s): sum VRAM with 10% reserved for driver/OS
            self.usable_vram = gpu.total_vram_gb * 0.9
        else:
            self.usable_vram = 0.0

        # --- Usable system RAM budget (15% reserved) ---
        self.usable_ram = hardware.available_ram_gb * 0.85

        # --- Speed coefficient (tok/s per billion params at Q4_K_M) ---
        if gpu is not None and gpu.speed_coefficient > 0:
            self._speed_coeff = float(gpu.speed_coefficient)
        else:
            # Fallback: estimate from CPU GFLOPS — very rough
            self._speed_coeff = max(hardware.cpu.estimated_gflops / 50.0, 1.0)

        # --- Backend label ---
        self._backend = gpu.backend if gpu is not None else "cpu"

        # --- Auto-detect available inference engine ---
        self._engine = _detect_engine()

        logger.info(
            "ModelOptimizer init: vram=%.1f GB, ram=%.1f GB, backend=%s, engine=%s",
            self.usable_vram,
            self.usable_ram,
            self._backend,
            self._engine,
        )

    # ------------------------------------------------------------------
    # pick_quant_and_offload
    # ------------------------------------------------------------------

    def pick_quant_and_offload(
        self,
        model_size_b: float,
        context_length: int = 4096,
    ) -> QuantOffloadResult:
        """Pick the best quantization and GPU offload strategy.

        Tries quantizations from highest quality (Q8_0) down to Q2_K and
        returns the best quality that fits in available memory.
        """
        kv_cache = _kv_cache_gb(model_size_b, context_length)

        for quant in _QUANT_PREFERENCE_ORDER:
            model_gb = _model_memory_gb(model_size_b, quant) + kv_cache

            result = self._try_fit(model_size_b, quant, model_gb, kv_cache)
            if result is not None:
                return result

        # Nothing fits even at Q2_K — aggressive quantization fallback
        quant = "Q2_K"
        model_gb = _model_memory_gb(model_size_b, quant) + kv_cache
        total_budget = self.usable_vram + self.usable_ram

        if total_budget > 0 and model_gb <= total_budget:
            gpu_frac = self.usable_vram / model_gb if model_gb > 0 else 0.0
            gpu_layers = self._estimate_gpu_layers(model_size_b, gpu_frac)
            vram_used = min(model_gb, self.usable_vram)
            ram_used = model_gb - vram_used
            return QuantOffloadResult(
                quantization=quant,
                gpu_layers=gpu_layers,
                strategy=MemoryStrategy.AGGRESSIVE_QUANT,
                vram_gb=round(vram_used, 2),
                ram_gb=round(ram_used, 2),
                total_gb=round(model_gb, 2),
                warning="Model requires aggressive quantization (Q2_K). "
                "Expect noticeable quality degradation.",
            )

        # Truly does not fit — report AGGRESSIVE_QUANT with a warning
        return QuantOffloadResult(
            quantization=quant,
            gpu_layers=0,
            strategy=MemoryStrategy.AGGRESSIVE_QUANT,
            vram_gb=0.0,
            ram_gb=round(self.usable_ram, 2),
            total_gb=round(model_gb, 2),
            warning=(
                f"Model needs ~{model_gb:.1f} GB but only "
                f"{total_budget:.1f} GB available. Will likely OOM or swap heavily."
            ),
        )

    def _try_fit(
        self,
        model_size_b: float,
        quant: str,
        model_gb: float,
        kv_cache: float,
    ) -> QuantOffloadResult | None:
        """Attempt to fit model_gb into available memory.

        Returns a ``QuantOffloadResult`` if possible, otherwise ``None``.
        """
        # Case 1: fits entirely in VRAM
        if model_gb <= self.usable_vram:
            return QuantOffloadResult(
                quantization=quant,
                gpu_layers=-1,
                strategy=MemoryStrategy.FULL_GPU,
                vram_gb=round(model_gb, 2),
                ram_gb=0.0,
                total_gb=round(model_gb, 2),
            )

        # Case 2: split between VRAM and RAM
        if self.usable_vram > 0 and model_gb <= self.usable_vram + self.usable_ram:
            gpu_frac = self.usable_vram / model_gb
            gpu_layers = self._estimate_gpu_layers(model_size_b, gpu_frac)
            vram_used = min(model_gb, self.usable_vram)
            ram_used = model_gb - vram_used
            return QuantOffloadResult(
                quantization=quant,
                gpu_layers=gpu_layers,
                strategy=MemoryStrategy.PARTIAL_OFFLOAD,
                vram_gb=round(vram_used, 2),
                ram_gb=round(ram_used, 2),
                total_gb=round(model_gb, 2),
            )

        # Case 3: no GPU (or GPU too small) — CPU only
        if model_gb <= self.usable_ram:
            return QuantOffloadResult(
                quantization=quant,
                gpu_layers=0,
                strategy=MemoryStrategy.CPU_ONLY,
                vram_gb=0.0,
                ram_gb=round(model_gb, 2),
                total_gb=round(model_gb, 2),
            )

        return None

    @staticmethod
    def _estimate_gpu_layers(model_size_b: float, gpu_fraction: float) -> int:
        """Estimate the number of transformer layers to offload to GPU.

        Typical transformer layer counts by parameter size are
        approximated as ``ceil(model_size_b * 4)``, capped at 80.
        ``gpu_fraction`` (0.0–1.0) of those layers go on GPU.
        """
        total_layers = min(math.ceil(model_size_b * 4), 80)
        return max(1, round(total_layers * min(gpu_fraction, 1.0)))

    # ------------------------------------------------------------------
    # predict_speed
    # ------------------------------------------------------------------

    def predict_speed(
        self,
        model_size_b: float,
        config: QuantOffloadResult,
    ) -> SpeedEstimate:
        """Estimate inference speed in tokens/second.

        The estimate is derived from the hardware speed coefficient, the
        quantization speed factor, and a penalty for partial / CPU offload.
        """
        if model_size_b <= 0:
            return SpeedEstimate(
                tokens_per_second=0.0,
                backend=self._backend,
                strategy=config.strategy,
                confidence="low",
            )

        base_tps = self._speed_coeff / model_size_b
        quant_factor = _QUANT_SPEED_FACTORS.get(config.quantization, 1.0)
        tps = base_tps * quant_factor

        # Offload penalties
        if config.strategy == MemoryStrategy.PARTIAL_OFFLOAD:
            tps *= 0.6
        elif config.strategy == MemoryStrategy.CPU_ONLY:
            tps *= 0.15
        elif config.strategy == MemoryStrategy.AGGRESSIVE_QUANT:
            tps *= 0.12

        # Confidence
        if config.strategy == MemoryStrategy.FULL_GPU and self._backend != "cpu":
            confidence = "high"
        elif config.strategy == MemoryStrategy.PARTIAL_OFFLOAD:
            confidence = "medium"
        else:
            confidence = "low"

        return SpeedEstimate(
            tokens_per_second=round(max(tps, 0.1), 1),
            backend=self._backend,
            strategy=config.strategy,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def recommend(
        self,
        priority: str = "balanced",
    ) -> list[ModelRecommendation]:
        """Generate sorted list of model recommendations.

        Args:
            priority: One of ``"speed"``, ``"quality"``, or ``"balanced"``.
                - ``"speed"``: prefer smaller models with faster quants.
                - ``"quality"``: prefer larger models with better quants.
                - ``"balanced"``: best quality that runs at >= 10 tok/s.
        """
        recs: list[ModelRecommendation] = []

        for label, size_b in sorted(_MODEL_SIZES.items(), key=lambda kv: kv[1]):
            config = self.pick_quant_and_offload(size_b)
            speed = self.predict_speed(size_b, config)

            # Skip models that truly cannot run (OOM warning)
            if config.warning and "OOM" in config.warning:
                continue

            cmd = self.serve_command(label, config)

            reason = self._recommendation_reason(label, config, speed)

            recs.append(
                ModelRecommendation(
                    model_size=label,
                    quantization=config.quantization,
                    reason=reason,
                    config=config,
                    speed=speed,
                    serve_command=cmd,
                )
            )

        return self._sort_recommendations(recs, priority)

    @staticmethod
    def _recommendation_reason(
        label: str,
        config: QuantOffloadResult,
        speed: SpeedEstimate,
    ) -> str:
        parts = [f"{label} at {config.quantization}"]
        parts.append(f"~{speed.tokens_per_second} tok/s ({speed.confidence} conf)")
        parts.append(f"{config.strategy.value}")
        if config.warning:
            parts.append(f"warning: {config.warning}")
        return "; ".join(parts)

    @staticmethod
    def _sort_recommendations(
        recs: list[ModelRecommendation],
        priority: str,
    ) -> list[ModelRecommendation]:
        if priority == "speed":
            return sorted(recs, key=lambda r: -r.speed.tokens_per_second)

        if priority == "quality":
            # Larger model + higher-quality quant first; break ties by speed
            size_order = list(_MODEL_SIZES.keys())
            quant_order = list(reversed(_QUANT_PREFERENCE_ORDER))
            return sorted(
                recs,
                key=lambda r: (
                    -(
                        size_order.index(r.model_size)
                        if r.model_size in size_order
                        else 0
                    ),
                    -(
                        quant_order.index(r.quantization)
                        if r.quantization in quant_order
                        else 0
                    ),
                    -r.speed.tokens_per_second,
                ),
            )

        # balanced: best quality that still achieves >= 10 tok/s, then the rest
        fast_enough = [r for r in recs if r.speed.tokens_per_second >= 10.0]
        too_slow = [r for r in recs if r.speed.tokens_per_second < 10.0]

        size_order = list(_MODEL_SIZES.keys())
        quant_order = list(reversed(_QUANT_PREFERENCE_ORDER))

        fast_enough.sort(
            key=lambda r: (
                -(size_order.index(r.model_size) if r.model_size in size_order else 0),
                -(
                    quant_order.index(r.quantization)
                    if r.quantization in quant_order
                    else 0
                ),
            ),
        )
        too_slow.sort(key=lambda r: -r.speed.tokens_per_second)

        return fast_enough + too_slow

    # ------------------------------------------------------------------
    # env_vars
    # ------------------------------------------------------------------

    def env_vars(self, model_size_b: float | None = None) -> dict[str, str]:
        """Compute recommended runtime environment variables.

        Returns engine-agnostic recommendations. Keys use OLLAMA_ prefix
        for Ollama compatibility but apply conceptually to other engines.
        """
        threads = self.hardware.cpu.threads
        ram_gb = self.hardware.total_ram_gb

        num_parallel = min(4, max(1, threads // 4))
        max_loaded = max(1, int(ram_gb // 8))

        env: dict[str, str] = {
            "OLLAMA_NUM_PARALLEL": str(num_parallel),
            "OLLAMA_MAX_LOADED_MODELS": str(max_loaded),
            "OLLAMA_GPU_OVERHEAD": "256",
        }

        # Flash attention supported on CUDA and Metal
        if self._backend in ("cuda", "metal"):
            env["OLLAMA_FLASH_ATTENTION"] = "1"

        return env

    # ------------------------------------------------------------------
    # serve_command
    # ------------------------------------------------------------------

    def serve_command(
        self,
        model_tag: str,
        config: QuantOffloadResult,
        engine: str | None = None,
    ) -> str:
        """Generate a ready-to-paste serve/run command.

        Args:
            model_tag: Model identifier (e.g., ``"phi-mini"``, ``"llama3.1:8b"``).
            config: Quantization/offload configuration.
            engine: Override engine. ``None`` (default) auto-detects from PATH.
                Explicit values: ``"edgeml"``, ``"ollama"``, ``"llama.cpp"``.
        """
        resolved = engine or self._engine
        if resolved == "ollama":
            return self._ollama_command(model_tag, config)
        if resolved == "llama.cpp":
            return self._llamacpp_command(model_tag, config)
        return self._edgeml_command(model_tag, config)

    @staticmethod
    def _edgeml_command(model_tag: str, config: QuantOffloadResult) -> str:
        parts = ["edgeml", "serve", model_tag]
        if config.gpu_layers == 0:
            parts.extend(["--engine", "cpu"])
        return " ".join(parts)

    @staticmethod
    def _ollama_command(model_tag: str, config: QuantOffloadResult) -> str:
        parts = ["ollama", "run", model_tag]
        if config.gpu_layers == 0:
            parts.insert(0, "OLLAMA_NUM_GPU=0")
        elif config.gpu_layers > 0:
            parts.insert(0, f"OLLAMA_NUM_GPU={config.gpu_layers}")
        return " ".join(parts)

    @staticmethod
    def _llamacpp_command(model_tag: str, config: QuantOffloadResult) -> str:
        parts = ["llama-server", "-m", model_tag]
        if config.gpu_layers >= 0:
            parts.extend(["-ngl", str(config.gpu_layers)])
        return " ".join(parts)


def _resolve_model_size(model: str) -> float | None:
    """Resolve a model name or tag to parameter count in billions."""
    import re

    tag = model.lower().replace("-", "").replace("_", "")

    # Try NB pattern (e.g., "8b", "70b", "0.5b")
    match = re.search(r"(\d+\.?\d*)b", tag)
    if match:
        return float(match.group(1))

    # Known model name -> size
    _NAME_SIZES: dict[str, float] = {
        "phimini": 3.8,
        "phi4mini": 3.8,
        "phimedium": 14.0,
        "gemma1b": 1.0,
        "gemma3b": 3.0,
        "gemma4b": 4.0,
        "mistral": 7.0,
        "mixtral": 46.7,
        "llama2": 7.0,
        "llama3": 8.0,
        "qwen2": 7.0,
        "qwen3": 8.0,
        "deepseekcoderv2": 6.7,
        "smollm": 0.36,
        "smollm360m": 0.36,
        "whispertiny": 0.039,
        "whisperbase": 0.074,
        "whispersmall": 0.244,
        "whispermedium": 0.769,
        "whisperlargev3": 1.55,
    }

    for name, size in _NAME_SIZES.items():
        if name in tag:
            return size

    return None
