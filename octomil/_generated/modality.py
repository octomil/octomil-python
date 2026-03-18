"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class Modality(str, Enum):
    TEXT = "text"
    """Text input and/or output"""
    IMAGE = "image"
    """Image input and/or output"""
    AUDIO = "audio"
    """Audio input and/or output"""
    VIDEO = "video"
    """Video input and/or output"""
