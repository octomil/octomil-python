"""Quantization pipeline (float <-> integer) for SecAgg+.

Stochastic quantization/dequantization following the Flower SecAgg+ scheme.
"""

from __future__ import annotations

from typing import List


def _stochastic_round(values: List[float]) -> List[int]:
    """Stochastic rounding: ``ceil(x)`` with probability ``x - floor(x)``.

    Matches the Flower SecAgg+ stochastic rounding implementation.
    """
    import math
    import random

    result: List[int] = []
    for v in values:
        c = math.ceil(v)
        # Probability of rounding down = ceil(v) - v
        if random.random() < (c - v):
            result.append(c - 1)
        else:
            result.append(c)
    return result


def quantize(
    values: List[float],
    clipping_range: float,
    target_range: int,
) -> List[int]:
    """Stochastic quantize floats to integers in ``[0, target_range]``.

    Follows the Flower SecAgg+ quantization scheme:
      1. Clip values to ``[-clipping_range, +clipping_range]``
      2. Shift to ``[0, 2 * clipping_range]``
      3. Scale to ``[0, target_range]``
      4. Stochastic round to integers

    The inverse is :func:`dequantize`.
    """
    if not values:
        return []

    quantizer = target_range / (2.0 * clipping_range) if clipping_range != 0 else 0.0
    pre_quantized = [
        (max(-clipping_range, min(clipping_range, v)) + clipping_range) * quantizer for v in values
    ]
    return _stochastic_round(pre_quantized)


def dequantize(
    quantized: List[int],
    clipping_range: float,
    target_range: int,
) -> List[float]:
    """Reverse :func:`quantize` — map integers back to floats in
    ``[-clipping_range, +clipping_range]``.
    """
    if not quantized:
        return []

    scale = (2.0 * clipping_range) / target_range if target_range != 0 else 0.0
    shift = -clipping_range
    return [q * scale + shift for q in quantized]
