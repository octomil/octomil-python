"""Model reference parser — canonical vocabulary for model_ref_kind.

Parses model reference strings into structured descriptors so route metadata
can record what kind of reference the caller supplied. The actual resolution
happens server-side; this module only classifies the reference.

Canonical vocabulary (shared across all SDKs):
    model       — plain model ID/slug
    app         — @app/{slug}/{capability}
    capability  — @capability/{capability}
    deployment  — deploy_{id_or_key}
    experiment  — exp_{experiment_id}/{variant_id}
    alias       — named alias
    default     — empty/unset model
    unknown     — unrecognized format
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

# Canonical kind vocabulary — the source of truth.
ModelRefKind = Literal[
    "model",
    "app",
    "capability",
    "deployment",
    "experiment",
    "alias",
    "default",
    "unknown",
]

_APP_RE = re.compile(r"^@app/([^/]+)/([^/]+)$")
_CAPABILITY_RE = re.compile(r"^@capability/([^/]+)$")
_DEPLOY_RE = re.compile(r"^deploy_(.+)$")
_EXPERIMENT_RE = re.compile(r"^(exp[^/]*)/([^/]+)$")


@dataclass(frozen=True)
class ParsedModelRef:
    """Structured model reference with a canonical kind for route metadata."""

    raw: str
    kind: ModelRefKind
    app_slug: Optional[str] = None
    capability: Optional[str] = None
    deployment_id: Optional[str] = None
    experiment_id: Optional[str] = None
    variant_id: Optional[str] = None


def parse_model_ref(model: str) -> ParsedModelRef:
    """Parse a model reference string into a structured descriptor.

    This is a pure function with no side effects.

    Parameters
    ----------
    model:
        The model reference string as provided by the caller.

    Returns
    -------
    ParsedModelRef
        Structured descriptor with a canonical ``kind`` field.
    """
    trimmed = model.strip()

    # Empty or missing → "default"
    if not trimmed:
        return ParsedModelRef(raw=trimmed, kind="default")

    # @app/<slug>/<capability>
    m = _APP_RE.match(trimmed)
    if m:
        return ParsedModelRef(
            raw=trimmed,
            kind="app",
            app_slug=m.group(1),
            capability=m.group(2),
        )

    # @capability/<cap>
    m = _CAPABILITY_RE.match(trimmed)
    if m:
        return ParsedModelRef(
            raw=trimmed,
            kind="capability",
            capability=m.group(1),
        )

    # deploy_<id>
    m = _DEPLOY_RE.match(trimmed)
    if m:
        return ParsedModelRef(
            raw=trimmed,
            kind="deployment",
            deployment_id=m.group(1),
        )

    # exp_<id>/<variant> or exp/<variant>
    m = _EXPERIMENT_RE.match(trimmed)
    if m:
        return ParsedModelRef(
            raw=trimmed,
            kind="experiment",
            experiment_id=m.group(1),
            variant_id=m.group(2),
        )

    # Plain model ID
    return ParsedModelRef(raw=trimmed, kind="model")
