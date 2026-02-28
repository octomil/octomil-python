"""FP16 alpha-scaling utility for NPU-safe model weights.

Models like Gemma 3 overflow FP16 range (+-65504) on NPUs. This applies
ANEMLL-style alpha-scaling: divide embedding weights by alpha, multiply
post-norm gains to compensate. Net effect is mathematically identical
inference with all weights safely within FP16 range.

Scale factors are stored as ONNX metadata_props so the transformation
is traceable and reversible.
"""

from __future__ import annotations

import re
from typing import Sequence

import onnx
from onnx import numpy_helper


def apply_fp16_scaling(
    model_path: str,
    output_path: str,
    alpha: float,
    weight_patterns: Sequence[str] | None = None,
) -> dict[str, float]:
    """Scale model weights to fit within FP16 range.

    Parameters
    ----------
    model_path:
        Path to input ONNX model.
    output_path:
        Path to write the scaled model.
    alpha:
        Scaling factor. Embedding weights are divided by alpha, post-norm
        weights are adjusted to compensate: ``new_gain = alpha * (1 + w) - 1``
        where ``w`` is the original weight value.
    weight_patterns:
        Optional list of regex patterns for weight names to scale.
        Defaults to common embedding and norm patterns.

    Returns
    -------
    dict mapping scaled initializer names to their applied scale factor.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    model = onnx.load(model_path)

    if weight_patterns is None:
        weight_patterns = [
            r".*embed.*weight",
            r".*embedding.*weight",
            r".*wte.*",
        ]
    norm_patterns = [
        r".*layernorm.*weight",
        r".*layer_norm.*weight",
        r".*norm.*weight",
        r".*ln_.*weight",
    ]

    embed_re = [re.compile(p, re.IGNORECASE) for p in weight_patterns]
    norm_re = [re.compile(p, re.IGNORECASE) for p in norm_patterns]

    scaled: dict[str, float] = {}

    for initializer in model.graph.initializer:
        name = initializer.name
        arr = numpy_helper.to_array(initializer).copy()

        # Scale embedding weights: divide by alpha
        if any(r.match(name) for r in embed_re):
            arr = arr / alpha
            new_init = numpy_helper.from_array(arr, name=name)
            initializer.CopyFrom(new_init)
            scaled[name] = 1.0 / alpha

        # Adjust post-norm weights to compensate
        elif any(r.match(name) for r in norm_re):
            # new_gain = alpha * (1 + w) - 1
            arr = alpha * (1.0 + arr) - 1.0
            new_init = numpy_helper.from_array(arr, name=name)
            initializer.CopyFrom(new_init)
            scaled[name] = alpha

    # Store scale factors in metadata
    model.metadata_props.append(
        onnx.StringStringEntryProto(key="octomil.fp16_scale.alpha", value=str(alpha))
    )
    model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="octomil.fp16_scale.scaled_weights",
            value=",".join(sorted(scaled.keys())),
        )
    )

    onnx.save(model, output_path)
    return scaled
