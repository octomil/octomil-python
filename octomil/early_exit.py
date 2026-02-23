"""Early exit / adaptive computation depth for inference.

Not all tokens need all transformer layers — easy tokens can exit early
when intermediate layer confidence is high enough.  This module provides:

- ``EarlyExitConfig``: threshold and preset configuration
- ``EarlyExitMonitor``: tracks per-request and aggregate early exit statistics
- ``SpeedQualityPreset``: predefined presets mapping to thresholds

For llama.cpp and MLX backends, early exit is backend-managed.  This module
focuses on **detection and telemetry**: tracking how many tokens exit early,
average layers used, and entropy at intermediate layers.

Research references:
- LayerSkip (Meta, 2024): skip schedules for different quality/speed tradeoffs
- CALM (Google, 2024): confident adaptive language modeling
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Speed / quality presets
# ---------------------------------------------------------------------------


class SpeedQualityPreset(str, Enum):
    """Predefined speed-quality tradeoff presets.

    Each preset maps to a confidence threshold for early exit decisions:
    - ``quality``: conservative — only exit when very confident (low threshold = fewer exits)
    - ``balanced``: moderate — exit when reasonably confident
    - ``fast``: aggressive — exit early more often for speed
    """

    QUALITY = "quality"
    BALANCED = "balanced"
    FAST = "fast"


# Mapping from preset to early exit entropy threshold.
# Lower entropy threshold → more tokens exit early (aggressive).
# Higher entropy threshold → fewer tokens exit early (conservative).
PRESET_THRESHOLDS: dict[SpeedQualityPreset, float] = {
    SpeedQualityPreset.QUALITY: 0.1,
    SpeedQualityPreset.BALANCED: 0.3,
    SpeedQualityPreset.FAST: 0.5,
}

# Mapping from preset to layer skip schedule.
# Expressed as fraction of total layers to evaluate before considering exit.
PRESET_MIN_LAYERS_FRACTION: dict[SpeedQualityPreset, float] = {
    SpeedQualityPreset.QUALITY: 0.75,
    SpeedQualityPreset.BALANCED: 0.5,
    SpeedQualityPreset.FAST: 0.25,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarlyExitConfig:
    """Configuration for early exit / adaptive computation depth.

    Parameters
    ----------
    enabled:
        Whether early exit monitoring is active.
    threshold:
        Entropy threshold below which a token is considered confident
        enough to exit early.  Range: 0.0 (never exit) to 1.0 (always exit).
        A token exits early when its intermediate logit entropy drops below
        this value.
    preset:
        Speed-quality preset that overrides threshold if set.
    min_layers_fraction:
        Minimum fraction of model layers to evaluate before considering
        early exit.  Prevents exiting too early on the first few layers
        where representations are not yet useful.
    total_layers:
        Total number of transformer layers in the model.  Used to compute
        actual layer counts from fractions.  When ``None``, layer-level
        stats are reported as fractions.
    """

    enabled: bool = False
    threshold: float = 0.3
    preset: Optional[SpeedQualityPreset] = None
    min_layers_fraction: float = 0.5
    total_layers: Optional[int] = None

    @property
    def effective_threshold(self) -> float:
        """Return the threshold, considering preset overrides."""
        if self.preset is not None:
            return PRESET_THRESHOLDS.get(self.preset, self.threshold)
        return self.threshold

    @property
    def effective_min_layers_fraction(self) -> float:
        """Return the minimum layers fraction, considering preset overrides."""
        if self.preset is not None:
            return PRESET_MIN_LAYERS_FRACTION.get(self.preset, self.min_layers_fraction)
        return self.min_layers_fraction

    @property
    def min_layers(self) -> Optional[int]:
        """Minimum number of layers to evaluate before early exit."""
        if self.total_layers is None:
            return None
        return max(1, int(self.total_layers * self.effective_min_layers_fraction))


def config_from_cli(
    *,
    early_exit_threshold: Optional[float] = None,
    speed_quality: Optional[str] = None,
) -> EarlyExitConfig:
    """Build an EarlyExitConfig from CLI arguments.

    Parameters
    ----------
    early_exit_threshold:
        Explicit entropy threshold (0.0-1.0).  When provided, early exit
        is enabled with this threshold.
    speed_quality:
        Preset name (``"quality"``, ``"balanced"``, ``"fast"``).  When
        provided, early exit is enabled with the preset's threshold.
        ``early_exit_threshold`` takes precedence if both are set.
    """
    if early_exit_threshold is None and speed_quality is None:
        return EarlyExitConfig(enabled=False)

    preset: Optional[SpeedQualityPreset] = None
    if speed_quality is not None:
        try:
            preset = SpeedQualityPreset(speed_quality)
        except ValueError:
            logger.warning(
                "Unknown speed-quality preset '%s', ignoring. "
                "Valid presets: quality, balanced, fast",
                speed_quality,
            )

    threshold = early_exit_threshold if early_exit_threshold is not None else 0.3
    if preset is not None and early_exit_threshold is None:
        threshold = PRESET_THRESHOLDS.get(preset, 0.3)

    return EarlyExitConfig(
        enabled=True,
        threshold=threshold,
        preset=preset,
    )


# ---------------------------------------------------------------------------
# Per-token exit record
# ---------------------------------------------------------------------------


@dataclass
class TokenExitRecord:
    """Record of a single token's exit decision.

    Parameters
    ----------
    layer_exited:
        The layer at which the token exited (1-indexed).
        ``None`` means the token went through all layers.
    total_layers:
        Total number of layers in the model.
    entropy:
        Logit entropy at the exit layer.
    exited_early:
        Whether the token exited before the final layer.
    """

    layer_exited: Optional[int] = None
    total_layers: Optional[int] = None
    entropy: float = 0.0
    exited_early: bool = False


# ---------------------------------------------------------------------------
# Per-request metrics
# ---------------------------------------------------------------------------


@dataclass
class EarlyExitRequestMetrics:
    """Aggregated early exit metrics for a single inference request.

    Parameters
    ----------
    total_tokens:
        Number of tokens generated in this request.
    early_exit_tokens:
        Number of tokens that exited early.
    avg_layers_used:
        Average number of layers used across all tokens.
    avg_entropy:
        Average entropy across all token exit points.
    min_layers_used:
        Minimum layers used by any token in this request.
    max_layers_used:
        Maximum layers used by any token in this request.
    """

    total_tokens: int = 0
    early_exit_tokens: int = 0
    avg_layers_used: float = 0.0
    avg_entropy: float = 0.0
    min_layers_used: int = 0
    max_layers_used: int = 0

    @property
    def exit_percentage(self) -> float:
        """Percentage of tokens that exited early (0.0-100.0)."""
        if self.total_tokens == 0:
            return 0.0
        return (self.early_exit_tokens / self.total_tokens) * 100.0

    def to_dict(self) -> dict[str, object]:
        """Serialize to a dict for telemetry/API responses."""
        return {
            "total_tokens": self.total_tokens,
            "early_exit_tokens": self.early_exit_tokens,
            "exit_percentage": round(self.exit_percentage, 2),
            "avg_layers_used": round(self.avg_layers_used, 2),
            "avg_entropy": round(self.avg_entropy, 4),
            "min_layers_used": self.min_layers_used,
            "max_layers_used": self.max_layers_used,
        }


# ---------------------------------------------------------------------------
# Entropy calculation
# ---------------------------------------------------------------------------


def compute_entropy(logits: list[float]) -> float:
    """Compute the Shannon entropy of a logit distribution.

    Parameters
    ----------
    logits:
        Raw logit values (pre-softmax).  Softmax is applied internally.

    Returns
    -------
    float
        Entropy in nats.  Normalized to [0, 1] range by dividing by
        log(vocab_size).  Returns 0.0 for empty or single-element inputs.
    """
    if len(logits) < 2:
        return 0.0

    # Numerical stability: subtract max before softmax
    max_logit = max(logits)
    shifted = [x - max_logit for x in logits]
    exp_vals = [math.exp(x) for x in shifted]
    total = sum(exp_vals)

    if total == 0:
        return 0.0

    probs = [e / total for e in exp_vals]

    # Shannon entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)

    # Normalize to [0, 1] by dividing by max possible entropy
    max_entropy = math.log(len(logits))
    if max_entropy > 0:
        entropy = entropy / max_entropy

    return entropy


def should_exit_early(
    entropy: float,
    config: EarlyExitConfig,
    current_layer: int,
) -> bool:
    """Determine whether a token should exit early based on entropy.

    Parameters
    ----------
    entropy:
        Normalized entropy (0-1) of the token's logit distribution at
        the current intermediate layer.
    config:
        Early exit configuration (threshold, min layers, etc.).
    current_layer:
        The current layer index (1-indexed).

    Returns
    -------
    bool
        ``True`` if the token should exit early at this layer.
    """
    if not config.enabled:
        return False

    # Check minimum layer requirement
    min_layers = config.min_layers
    if min_layers is not None and current_layer < min_layers:
        return False

    return entropy < config.effective_threshold


# ---------------------------------------------------------------------------
# Monitor — aggregate stats across requests
# ---------------------------------------------------------------------------


@dataclass
class EarlyExitStats:
    """Aggregate early exit statistics across all requests.

    Thread-safe — uses a lock for concurrent access.
    """

    total_requests: int = 0
    total_tokens: int = 0
    total_early_exit_tokens: int = 0
    total_layers_used: float = 0.0
    total_entropy_sum: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def exit_percentage(self) -> float:
        """Percentage of tokens that exited early (0.0-100.0)."""
        if self.total_tokens == 0:
            return 0.0
        return (self.total_early_exit_tokens / self.total_tokens) * 100.0

    @property
    def avg_layers_used(self) -> float:
        """Average number of layers used per token across all requests."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_layers_used / self.total_tokens

    @property
    def avg_entropy(self) -> float:
        """Average entropy at exit points across all tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_entropy_sum / self.total_tokens

    def to_dict(self) -> dict[str, object]:
        """Serialize to a dict for the stats endpoint."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_early_exit_tokens": self.total_early_exit_tokens,
            "exit_percentage": round(self.exit_percentage, 2),
            "avg_layers_used": round(self.avg_layers_used, 2),
            "avg_entropy": round(self.avg_entropy, 4),
        }


class EarlyExitMonitor:
    """Monitor and aggregate early exit metrics across inference requests.

    Thread-safe.  Designed to be attached to ``ServerState`` and queried
    by the ``/v1/early-exit/stats`` endpoint.
    """

    def __init__(self, config: EarlyExitConfig) -> None:
        self.config = config
        self._stats = EarlyExitStats()

    @property
    def stats(self) -> EarlyExitStats:
        return self._stats

    def record_request(self, metrics: EarlyExitRequestMetrics) -> None:
        """Record early exit metrics from a completed request.

        Parameters
        ----------
        metrics:
            Per-request early exit metrics to fold into aggregate stats.
        """
        with self._stats._lock:
            self._stats.total_requests += 1
            self._stats.total_tokens += metrics.total_tokens
            self._stats.total_early_exit_tokens += metrics.early_exit_tokens
            self._stats.total_layers_used += (
                metrics.avg_layers_used * metrics.total_tokens
            )
            self._stats.total_entropy_sum += metrics.avg_entropy * metrics.total_tokens

    def simulate_token_exits(
        self,
        token_count: int,
        total_layers: int,
    ) -> EarlyExitRequestMetrics:
        """Simulate early exit decisions for a batch of tokens.

        This is used when the backend does not natively report per-token
        layer usage.  It generates synthetic exit records based on a
        probability distribution derived from the threshold.

        In production with backends that support early exit natively
        (e.g., modified llama.cpp or MLX builds), this would be replaced
        by actual per-token layer data from the backend.

        Parameters
        ----------
        token_count:
            Number of tokens to simulate.
        total_layers:
            Total transformer layers in the model.

        Returns
        -------
        EarlyExitRequestMetrics
            Simulated metrics for this request.
        """
        if token_count == 0 or not self.config.enabled:
            return EarlyExitRequestMetrics(
                total_tokens=token_count,
                max_layers_used=total_layers,
                min_layers_used=total_layers,
                avg_layers_used=float(total_layers),
            )

        threshold = self.config.effective_threshold
        min_frac = self.config.effective_min_layers_fraction
        min_layers = max(1, int(total_layers * min_frac))

        # Estimate: tokens with low complexity (predictable continuations)
        # exit early.  We use the threshold as a rough proxy for exit rate.
        # Higher threshold → more tokens exit early.
        exit_rate = threshold  # e.g., 0.3 threshold → ~30% of tokens exit early

        early_count = int(token_count * exit_rate)
        normal_count = token_count - early_count

        # Early-exit tokens use approximately (min_layers + total_layers) / 2 layers
        early_avg_layers = (min_layers + total_layers) / 2.0
        normal_avg_layers = float(total_layers)

        total_layers_sum = (early_count * early_avg_layers) + (
            normal_count * normal_avg_layers
        )
        avg_layers = total_layers_sum / token_count if token_count > 0 else 0.0

        # Synthetic entropy: early-exit tokens have low entropy, normal tokens vary
        early_entropy = threshold * 0.5  # well below threshold
        normal_entropy = 0.6  # moderate entropy
        avg_entropy = (
            (early_count * early_entropy + normal_count * normal_entropy) / token_count
            if token_count > 0
            else 0.0
        )

        return EarlyExitRequestMetrics(
            total_tokens=token_count,
            early_exit_tokens=early_count,
            avg_layers_used=avg_layers,
            avg_entropy=avg_entropy,
            min_layers_used=min_layers if early_count > 0 else total_layers,
            max_layers_used=total_layers,
        )

    def get_stats_dict(self) -> dict[str, object]:
        """Return current aggregate stats as a dict."""
        config_dict: dict[str, object] = {
            "enabled": self.config.enabled,
            "threshold": self.config.effective_threshold,
            "min_layers_fraction": self.config.effective_min_layers_fraction,
        }
        if self.config.preset is not None:
            config_dict["preset"] = self.config.preset.value
        if self.config.total_layers is not None:
            config_dict["total_layers"] = self.config.total_layers
            min_l = self.config.min_layers
            if min_l is not None:
                config_dict["min_layers"] = min_l

        return {
            "config": config_dict,
            "stats": self._stats.to_dict(),
        }
