"""Extensible filter pipeline for federated learning weight updates.

Provides an abstract base class :class:`DeltaFilter` that users can subclass
to create custom filters, a :class:`FilterRegistry` for registering filters
by name, and :func:`apply_filters` to run a composable pipeline with an
audit trail.

Built-in filters:
  - :class:`GradientClipFilter`
  - :class:`GaussianNoiseFilter`
  - :class:`NormValidationFilter`
  - :class:`SparsificationFilter`
  - :class:`QuantizationFilter`
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data kinds
# ---------------------------------------------------------------------------


class DataKind(str, Enum):
    """Kind of data a filter operates on."""

    WEIGHTS = "weights"
    WEIGHT_DIFF = "weight_diff"
    METRICS = "metrics"
    ANY = "any"


# ---------------------------------------------------------------------------
# Filter result with audit trail
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Result of running a filter pipeline.

    Attributes:
        delta: The filtered state-dict delta.
        audit_trail: Ordered list of ``(filter_name, data_kind)`` tuples
            recording which filters were applied.
    """

    delta: Dict[str, Any]
    audit_trail: List[tuple] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class DeltaFilter(ABC):
    """Abstract base class for state-dict delta filters.

    Subclasses must implement :meth:`process` which receives a state-dict
    delta and returns the filtered delta (or *None* to indicate no change).

    Filters optionally declare which :class:`DataKind` values they support
    via ``supported_data_kinds``.  When set, the pipeline will only apply the
    filter when the current data kind matches.

    Example::

        class MyClipFilter(DeltaFilter):
            supported_data_kinds = [DataKind.WEIGHT_DIFF]

            def __init__(self, threshold: float = 5.0):
                super().__init__()
                self.threshold = threshold

            def process(self, delta, config=None):
                import torch
                return {
                    k: torch.clamp(v, -self.threshold, self.threshold)
                    if torch.is_tensor(v) else v
                    for k, v in delta.items()
                }

        FilterRegistry.register("my_clip", MyClipFilter)
    """

    #: Data kinds this filter supports.  Empty or ``None`` means all kinds.
    supported_data_kinds: Optional[List[DataKind]] = None

    def __init__(self) -> None:
        self._name: str = self.__class__.__name__

    @property
    def name(self) -> str:
        """Human-readable filter name (defaults to class name)."""
        return self._name

    @abstractmethod
    def process(
        self,
        delta: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Apply this filter to a state-dict delta.

        Parameters
        ----------
        delta:
            The state-dict delta to filter.  Implementations should NOT
            mutate *delta* in-place -- return a new dict instead.
        config:
            Optional config dict (from the server round config).

        Returns
        -------
        The filtered delta, or *None* if the filter did not change anything.
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class FilterRegistry:
    """Global registry mapping filter type names to :class:`DeltaFilter` subclasses.

    Built-in filters are pre-registered.  Users can register custom filters::

        FilterRegistry.register("my_filter", MyFilterClass)
    """

    _registry: Dict[str, Type[DeltaFilter]] = {}

    @classmethod
    def register(cls, name: str, filter_class: Type[DeltaFilter]) -> None:
        """Register a filter class under *name*.

        Raises :class:`ValueError` if *filter_class* is not a
        :class:`DeltaFilter` subclass.
        """
        if not (isinstance(filter_class, type) and issubclass(filter_class, DeltaFilter)):
            raise ValueError(
                f"filter_class must be a DeltaFilter subclass, got {filter_class}"
            )
        cls._registry[name] = filter_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[DeltaFilter]]:
        """Look up a filter class by name, or *None* if not found."""
        return cls._registry.get(name)

    @classmethod
    def list_filters(cls) -> List[str]:
        """Return all registered filter names."""
        return sorted(cls._registry.keys())

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Remove a filter by name. Returns True if it existed."""
        return cls._registry.pop(name, None) is not None


# ---------------------------------------------------------------------------
# Built-in filters
# ---------------------------------------------------------------------------


def _require_torch():
    """Import and return torch, raising a clear error if unavailable."""
    try:
        import torch  # type: ignore
        return torch
    except Exception as exc:
        from .api_client import OctomilClientError
        raise OctomilClientError("torch is required for filter pipeline") from exc


class GradientClipFilter(DeltaFilter):
    """Clip per-tensor norms to ``max_norm``."""

    supported_data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

    def __init__(self, max_norm: float = 1.0) -> None:
        super().__init__()
        self.max_norm = max_norm

    def process(self, delta, config=None):
        torch = _require_torch()
        max_norm = float((config or {}).get("max_norm", self.max_norm))
        result = {}
        for key, tensor in delta.items():
            if not torch.is_tensor(tensor):
                result[key] = tensor
                continue
            norm = torch.norm(tensor.float().flatten(), dim=0)
            if norm > max_norm:
                result[key] = tensor * (max_norm / norm)
            else:
                result[key] = tensor
        return result


class GaussianNoiseFilter(DeltaFilter):
    """Add N(0, stddev^2) noise to each tensor."""

    supported_data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

    def __init__(self, stddev: float = 0.01) -> None:
        super().__init__()
        self.stddev = stddev

    def process(self, delta, config=None):
        torch = _require_torch()
        stddev = float((config or {}).get("stddev", self.stddev))
        result = {}
        for key, tensor in delta.items():
            if not torch.is_tensor(tensor):
                result[key] = tensor
                continue
            result[key] = tensor + torch.randn_like(tensor.float()) * stddev
        return result


class NormValidationFilter(DeltaFilter):
    """Drop tensors whose norm exceeds ``max_norm``."""

    supported_data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

    def __init__(self, max_norm: float = 10.0) -> None:
        super().__init__()
        self.max_norm = max_norm

    def process(self, delta, config=None):
        torch = _require_torch()
        max_norm = float((config or {}).get("max_norm", self.max_norm))
        result = {}
        for key, tensor in delta.items():
            if not torch.is_tensor(tensor):
                result[key] = tensor
                continue
            if torch.norm(tensor.float().flatten(), dim=0) <= max_norm:
                result[key] = tensor
        return result


class SparsificationFilter(DeltaFilter):
    """Zero out values below the top-k% by magnitude."""

    supported_data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

    def __init__(self, top_k_percent: float = 10.0) -> None:
        super().__init__()
        self.top_k_percent = top_k_percent

    def process(self, delta, config=None):
        torch = _require_torch()
        top_k_percent = float((config or {}).get("top_k_percent", self.top_k_percent))
        result = {}
        for key, tensor in delta.items():
            if not torch.is_tensor(tensor):
                result[key] = tensor
                continue
            flat = tensor.float().abs().flatten()
            k = max(1, int(math.ceil(flat.numel() * top_k_percent / 100.0)))
            threshold = torch.topk(flat, k).values[-1]
            mask = tensor.abs() >= threshold
            result[key] = tensor * mask
        return result


class QuantizationFilter(DeltaFilter):
    """Round tensor values to ``bits``-bit resolution."""

    supported_data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

    def __init__(self, bits: int = 8) -> None:
        super().__init__()
        self.bits = bits

    def process(self, delta, config=None):
        torch = _require_torch()
        bits = int((config or {}).get("bits", self.bits))
        levels = (1 << bits) - 1
        result = {}
        for key, tensor in delta.items():
            if not torch.is_tensor(tensor):
                result[key] = tensor
                continue
            t_min = tensor.min()
            t_max = tensor.max()
            if t_min == t_max:
                result[key] = tensor
                continue
            scale = (t_max - t_min) / levels
            result[key] = (torch.round((tensor - t_min) / scale) * scale) + t_min
        return result


# ---------------------------------------------------------------------------
# Register built-in filters
# ---------------------------------------------------------------------------

FilterRegistry.register("gradient_clip", GradientClipFilter)
FilterRegistry.register("gaussian_noise", GaussianNoiseFilter)
FilterRegistry.register("norm_validation", NormValidationFilter)
FilterRegistry.register("sparsification", SparsificationFilter)
FilterRegistry.register("quantization", QuantizationFilter)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def apply_filters(
    delta: Dict[str, Any],
    filters: List[Union[Dict[str, Any], DeltaFilter]],
    data_kind: DataKind = DataKind.WEIGHT_DIFF,
) -> FilterResult:
    """Apply a composable filter pipeline to a state-dict delta.

    Accepts a mix of:
      - **dict configs** (``{"type": "gradient_clip", "max_norm": 1.0}``)
        which are resolved via :class:`FilterRegistry`
      - **DeltaFilter instances** which are called directly

    Returns a :class:`FilterResult` containing the filtered delta and an
    audit trail of applied filters.

    Parameters
    ----------
    delta:
        The state-dict delta to filter.
    filters:
        List of filter configs (dicts) or DeltaFilter instances.
    data_kind:
        The kind of data being filtered (used for data-kind routing).

    Returns
    -------
    FilterResult
        The filtered delta and audit trail.
    """
    torch = _require_torch()

    # Clone tensors so we never mutate the caller's data.
    current = {k: v.clone() if torch.is_tensor(v) else v for k, v in delta.items()}
    audit_trail: List[tuple] = []

    for f in filters:
        if isinstance(f, DeltaFilter):
            filter_instance = f
            config = None
        elif isinstance(f, dict):
            filter_type = f.get("type", "")
            filter_class = FilterRegistry.get(filter_type)
            if filter_class is None:
                logger.warning("Unknown filter type '%s', skipping", filter_type)
                continue
            filter_instance = filter_class()
            config = f
        else:
            logger.warning("Invalid filter entry %r, skipping", f)
            continue

        # Data-kind routing: skip if this filter doesn't support the current kind.
        supported = filter_instance.supported_data_kinds
        if supported and data_kind not in supported and DataKind.ANY not in supported:
            logger.debug(
                "Skipping filter '%s': data kind '%s' not in %s",
                filter_instance.name, data_kind.value, [dk.value for dk in supported],
            )
            continue

        result = filter_instance.process(current, config=config)
        if result is not None:
            current = result
            audit_trail.append((filter_instance.name, data_kind.value))

    return FilterResult(delta=current, audit_trail=audit_trail)
