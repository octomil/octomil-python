"""ModelRef — discriminated union for model resolution.

A ModelRef identifies a model either by its direct ID or by a
capability tag. Used in responses.create(), audio, text, and
anywhere a model is referenced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from octomil._generated.model_capability import ModelCapability


@dataclass(frozen=True)
class _ModelRefId:
    """Resolve by direct model ID."""

    model_id: str


@dataclass(frozen=True)
class _ModelRefCapability:
    """Resolve by capability tag."""

    capability: ModelCapability


# Public union type
ModelRef = Union[_ModelRefId, _ModelRefCapability]


def model_ref_id(model_id: str) -> ModelRef:
    """Create a ModelRef that resolves by direct model ID."""
    return _ModelRefId(model_id=model_id)


def model_ref_capability(capability: ModelCapability) -> ModelRef:
    """Create a ModelRef that resolves by capability tag."""
    return _ModelRefCapability(capability=capability)


# Convenience namespace for cleaner call sites:
#   ModelRef.id("gemma-2b")
#   ModelRef.capability(ModelCapability.CHAT)
class ModelRefFactory:
    """Factory with .id() and .capability() class methods."""

    @staticmethod
    def id(model_id: str) -> ModelRef:  # noqa: A003
        return model_ref_id(model_id)

    @staticmethod
    def capability(cap: ModelCapability) -> ModelRef:
        return model_ref_capability(cap)


def is_id_ref(ref: ModelRef) -> bool:
    """Check if a ModelRef is an ID reference."""
    return isinstance(ref, _ModelRefId)


def is_capability_ref(ref: ModelRef) -> bool:
    """Check if a ModelRef is a capability reference."""
    return isinstance(ref, _ModelRefCapability)


def get_model_id(ref: ModelRef) -> str | None:
    """Extract model ID from a ModelRef, or None if capability-based."""
    if isinstance(ref, _ModelRefId):
        return ref.model_id
    return None


def get_capability(ref: ModelRef) -> ModelCapability | None:
    """Extract capability from a ModelRef, or None if ID-based."""
    if isinstance(ref, _ModelRefCapability):
        return ref.capability
    return None
