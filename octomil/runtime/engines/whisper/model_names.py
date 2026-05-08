"""Whisper model-name detection — non-legacy, product-path safe.

This module owns the canonical whisper model-name set and the
:func:`is_whisper_model` predicate. It is deliberately separate from
``_legacy_pywhisper.py`` so the product path can detect whisper model
names without pulling the legacy pywhispercpp engine into
``sys.modules``.

Importing this module is cheap and side-effect-free — no engine
plugins, no native bindings, no inference machinery. The legacy
module re-exports from here for backwards compatibility with parity /
benchmark code paths.
"""

from __future__ import annotations

# Whisper model sizes — display name to whisper.cpp / HuggingFace size id.
_WHISPER_MODELS: dict[str, str] = {
    "whisper-tiny": "tiny",
    "whisper-base": "base",
    "whisper-small": "small",
    "whisper-medium": "medium",
    "whisper-large-v3": "large-v3",
}


def is_whisper_model(model_name: str) -> bool:
    """Check if a model name refers to a Whisper speech-to-text model.

    Pure name-prefix detection — does NOT load any engine, model, or
    artifact. Safe to call on product paths.
    """
    return model_name.lower() in _WHISPER_MODELS


__all__ = ["_WHISPER_MODELS", "is_whisper_model"]
