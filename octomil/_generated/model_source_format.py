"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ModelSourceFormat(str, Enum):
    SAFETENSORS = "safetensors"
    """Hugging Face Safetensors format"""
    PYTORCH = "pytorch"
    """PyTorch state_dict or TorchScript"""
    TENSORFLOW = "tensorflow"
    """TensorFlow SavedModel or Keras"""
    ONNX = "onnx"
    """ONNX (source / unoptimized)"""
    GGUF = "gguf"
    """GGUF quantized weights (llama.cpp ecosystem)"""
    CUSTOM = "custom"
    """Vendor-specific or proprietary format"""
