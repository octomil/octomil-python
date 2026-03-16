"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ContentPartType(str, Enum):
    INPUT_TEXT = "input_text"
    """Plain text content"""
    INPUT_IMAGE = "input_image"
    """Image content (JPEG, PNG, WebP, GIF)"""
    INPUT_AUDIO = "input_audio"
    """Audio content (MPEG, WAV, OGG)"""
    INPUT_VIDEO = "input_video"
    """Video content (MP4, WebM)"""
