"""Whisper.cpp engine — on-device speech-to-text."""

from octomil.runtime.engines.whisper.engine import WhisperCppEngine, is_whisper_model

TIER = "supported"

__all__ = ["WhisperCppEngine", "TIER", "is_whisper_model"]
