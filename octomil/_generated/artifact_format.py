"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ArtifactFormat(str, Enum):
    COREML = "coreml"
    """Apple Core ML package (.mlpackage / .mlmodel)"""
    TFLITE = "tflite"
    """TensorFlow Lite flatbuffer (.tflite)"""
    ONNX = "onnx"
    """ONNX optimized for deployment (.onnx)"""
    GGUF = "gguf"
    """GGUF quantized model (.gguf)"""
    MLX = "mlx"
    """MLX weights directory"""
    MNN = "mnn"
    """MNN optimized model (.mnn)"""
    TRANSFORMERSJS = "transformersjs"
    """Transformers.js package (ONNX + tokenizer bundle)"""
    CLOUD = "cloud"
    """Cloud-hosted inference endpoint. No local artifact."""
