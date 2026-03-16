"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class InputModality(str, Enum):
    TEXT = "text"
    """Text input"""
    IMAGE = "image"
    """Image input"""
    AUDIO = "audio"
    """Audio input"""
    VIDEO = "video"
    """Video input"""
