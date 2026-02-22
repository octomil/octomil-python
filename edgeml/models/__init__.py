"""Unified model resolution with ``model:variant`` syntax.

Parse Ollama-style model specifiers, resolve to engine-specific artifacts,
and provide a single catalog replacing per-engine hardcoded sets.

Also re-exports legacy data classes (DeploymentPlan, etc.) that previously
lived in ``edgeml/models.py`` so existing imports remain valid.

Usage::

    from edgeml.models import parse, resolve, list_models

    parsed = parse("gemma-3b:4bit")
    resolved = resolve("gemma-3b:4bit", available_engines=["mlx-lm", "llama.cpp"])
"""

from .catalog import CATALOG, list_models
from .parser import ParsedModel, parse
from .resolver import ResolvedModel, resolve

# Re-export legacy data classes from the old models.py
from ._types import (
    DeploymentPlan,
    DeploymentResult,
    DeviceDeployment,
    DeviceDeploymentStatus,
    RollbackResult,
    TrainingSession,
)

__all__ = [
    # New model resolution API
    "CATALOG",
    "ParsedModel",
    "ResolvedModel",
    "list_models",
    "parse",
    "resolve",
    # Legacy data classes
    "DeploymentPlan",
    "DeploymentResult",
    "DeviceDeployment",
    "DeviceDeploymentStatus",
    "RollbackResult",
    "TrainingSession",
]
