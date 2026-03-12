"""Ollama engine — zero-pip fallback via local Ollama server."""

from octomil.runtime.engines.ollama.engine import OllamaEngine

TIER = "supported"

__all__ = ["OllamaEngine", "TIER"]
