"""Tests for octomil.utils.fp16_scaling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from octomil.utils.fp16_scaling import apply_fp16_scaling


def _make_model_with_weights(
    output_path: str,
    embed_values: np.ndarray,
    norm_values: np.ndarray,
) -> None:
    """Create a minimal ONNX model with named embedding and norm weights."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])

    embed_init = numpy_helper.from_array(embed_values, name="model.embed_tokens.weight")
    norm_init = numpy_helper.from_array(norm_values, name="model.layernorm.weight")

    # Simple matmul + add to use both initializers
    matmul = helper.make_node(
        "MatMul", ["input", "model.embed_tokens.weight"], ["hidden"]
    )
    add = helper.make_node("Add", ["hidden", "model.layernorm.weight"], ["output"])

    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

    graph = helper.make_graph(
        [matmul, add],
        "test_model",
        [X],
        [Y],
        initializer=[embed_init, norm_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, output_path)


class TestFP16Scaling:
    def test_embedding_scaled_down(self, tmp_path: Path) -> None:
        embed = np.array([[60000.0, -60000.0, 100.0, 200.0]], dtype=np.float32)
        norm = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "scaled.onnx")
        _make_model_with_weights(src, embed, norm)

        scaled = apply_fp16_scaling(src, dst, alpha=2.0)

        model = onnx.load(dst)
        for init in model.graph.initializer:
            if init.name == "model.embed_tokens.weight":
                arr = numpy_helper.to_array(init)
                np.testing.assert_allclose(arr, embed / 2.0, rtol=1e-5)
        assert "model.embed_tokens.weight" in scaled

    def test_norm_compensated(self, tmp_path: Path) -> None:
        embed = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)
        norm = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "scaled.onnx")
        _make_model_with_weights(src, embed, norm)

        apply_fp16_scaling(src, dst, alpha=2.0)

        model = onnx.load(dst)
        for init in model.graph.initializer:
            if init.name == "model.layernorm.weight":
                arr = numpy_helper.to_array(init)
                expected = 2.0 * (1.0 + norm) - 1.0
                np.testing.assert_allclose(arr, expected, rtol=1e-5)

    def test_weights_within_fp16_range(self, tmp_path: Path) -> None:
        # Create weights that overflow FP16
        embed = np.array([[70000.0, -70000.0, 50000.0, -50000.0]], dtype=np.float32)
        norm = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "scaled.onnx")
        _make_model_with_weights(src, embed, norm)

        apply_fp16_scaling(src, dst, alpha=2.0)

        model = onnx.load(dst)
        for init in model.graph.initializer:
            if init.name == "model.embed_tokens.weight":
                arr = numpy_helper.to_array(init)
                assert np.all(np.abs(arr) <= 65504.0)

    def test_metadata_stored(self, tmp_path: Path) -> None:
        embed = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)
        norm = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        src = str(tmp_path / "model.onnx")
        dst = str(tmp_path / "scaled.onnx")
        _make_model_with_weights(src, embed, norm)

        apply_fp16_scaling(src, dst, alpha=2.0)

        model = onnx.load(dst)
        meta = {p.key: p.value for p in model.metadata_props}
        assert meta["octomil.fp16_scale.alpha"] == "2.0"
        assert "model.embed_tokens.weight" in meta["octomil.fp16_scale.scaled_weights"]

    def test_invalid_alpha_raises(self, tmp_path: Path) -> None:
        embed = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        norm = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        src = str(tmp_path / "model.onnx")
        _make_model_with_weights(src, embed, norm)

        with pytest.raises(ValueError, match="alpha must be positive"):
            apply_fp16_scaling(src, str(tmp_path / "out.onnx"), alpha=-1.0)
