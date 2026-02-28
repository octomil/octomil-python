"""Append in-graph sampling nodes to ONNX models.

Reduces data transfer from vocab-sized float32 arrays (e.g. 128K × 4 = 512KB)
to a single int64 scalar (4 bytes) per token by moving argmax into the graph.
"""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper


def append_argmax(
    model_path: str,
    output_path: str,
    axis: int = -1,
    keepdims: bool = False,
) -> None:
    """Append an ArgMax node after the final output of an ONNX model.

    The original output (typically logits with shape [batch, seq, vocab]) is
    replaced by an int64 output containing the argmax index.

    Sets ONNX metadata ``octomil.sampling = "argmax"`` so consumers can
    detect that sampling is already embedded in the graph.

    Parameters
    ----------
    model_path:
        Path to the input ONNX model.
    output_path:
        Path to write the modified model.
    axis:
        Axis along which to compute argmax. Default -1 (vocab dimension).
    keepdims:
        Whether to keep the reduced dimension. Default False.
    """
    model = onnx.load(model_path)
    graph = model.graph

    # Get the last output
    original_output = graph.output[-1]
    original_output_name = original_output.name

    # Create intermediate name for original logits
    logits_name = original_output_name + "_logits"

    # Rename the original output in producing nodes
    for node in graph.node:
        for i, out in enumerate(node.output):
            if out == original_output_name:
                node.output[i] = logits_name

    # Create ArgMax node
    argmax_node = helper.make_node(
        "ArgMax",
        inputs=[logits_name],
        outputs=[original_output_name],
        axis=axis,
        keepdims=int(keepdims),
    )
    graph.node.append(argmax_node)

    # Replace output type info: int64 instead of float
    graph.output.remove(original_output)
    new_output = helper.make_tensor_value_info(
        original_output_name, TensorProto.INT64, None
    )
    graph.output.append(new_output)

    # Tag model metadata so consumers can detect embedded sampling
    model.metadata_props.append(
        onnx.StringStringEntryProto(key="octomil.sampling", value="argmax")
    )

    onnx.save(model, output_path)
