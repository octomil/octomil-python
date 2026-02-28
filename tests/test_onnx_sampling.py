"""Tests for octomil.utils.onnx_sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from octomil.utils.onnx_sampling import append_argmax


def _make_dummy_model(output_path: str, vocab_size: int = 100) -> None:
    """Create a minimal ONNX model that outputs random logits."""
    # Input: [1, vocab_size]
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, vocab_size])

    # Identity node -- pass through input as "logits"
    identity = helper.make_node("Identity", inputs=["input"], outputs=["logits"])

    # Output: [1, vocab_size]
    Y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, vocab_size])

    graph = helper.make_graph([identity], "test_model", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, output_path)


class TestAppendArgmax:
    def test_output_is_int64(self, tmp_path: Path) -> None:
        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "model_argmax.onnx")
        _make_dummy_model(src)

        append_argmax(src, dst)

        model = onnx.load(dst)
        assert model.graph.output[-1].type.tensor_type.elem_type == TensorProto.INT64

    def test_metadata_tag_set(self, tmp_path: Path) -> None:
        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "model_argmax.onnx")
        _make_dummy_model(src)

        append_argmax(src, dst)

        model = onnx.load(dst)
        meta = {p.key: p.value for p in model.metadata_props}
        assert meta.get("octomil.sampling") == "argmax"

    def test_argmax_node_added(self, tmp_path: Path) -> None:
        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "model_argmax.onnx")
        _make_dummy_model(src)

        append_argmax(src, dst)

        model = onnx.load(dst)
        argmax_nodes = [n for n in model.graph.node if n.op_type == "ArgMax"]
        assert len(argmax_nodes) == 1

    def test_inference_returns_scalar(self, tmp_path: Path) -> None:
        vocab_size = 100
        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "model_argmax.onnx")
        _make_dummy_model(src, vocab_size=vocab_size)

        append_argmax(src, dst)

        session = ort.InferenceSession(dst)
        # Create input with known argmax at index 42
        logits = np.zeros((1, vocab_size), dtype=np.float32)
        logits[0, 42] = 10.0
        result = session.run(None, {"input": logits})
        assert result[0].flatten()[0] == 42
