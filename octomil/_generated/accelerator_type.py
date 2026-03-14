"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class AcceleratorType(str, Enum):
    GPU = "gpu"
    """Generic GPU (Metal, Vulkan, OpenCL, OpenGL ES)"""
    NPU = "npu"
    """Generic neural processing unit"""
    ANE = "ane"
    """Apple Neural Engine"""
    NNAPI = "nnapi"
    """Android NNAPI delegate (Qualcomm QNN, Samsung Eden, MediaTek NeuroPilot)"""
    WEBGPU = "webgpu"
    """WebGPU API in browser"""
