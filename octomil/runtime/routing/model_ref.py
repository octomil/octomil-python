"""Canonical model reference classification for routing metadata."""

from __future__ import annotations

from dataclasses import dataclass

from octomil._generated.model_ref_kind import ModelRefKind

CANONICAL_MODEL_REF_KINDS = frozenset(kind.value for kind in ModelRefKind)


@dataclass(frozen=True)
class ParsedModelRef:
    """Structured classification of a user-provided model reference."""

    raw: str
    kind: str
    model_slug: str | None = None
    app_slug: str | None = None
    capability: str | None = None
    deployment_id: str | None = None
    experiment_id: str | None = None
    variant_id: str | None = None


def parse_model_ref(model: str | None) -> ParsedModelRef:
    """Classify a model reference without resolving it.

    Resolution is still owned by the server/runtime planner. This helper only
    provides the contract-level ``model_ref_kind`` used in route metadata and
    telemetry.
    """

    raw = (model or "").strip()
    if not raw:
        return ParsedModelRef(raw=raw, kind=ModelRefKind.DEFAULT.value)

    if raw.startswith("@app/"):
        parts = raw.removeprefix("@app/").split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return ParsedModelRef(
                raw=raw,
                kind=ModelRefKind.APP.value,
                app_slug=parts[0],
                capability=parts[1],
            )
        return ParsedModelRef(raw=raw, kind=ModelRefKind.UNKNOWN.value)

    if raw.startswith("@capability/"):
        capability = raw.removeprefix("@capability/")
        if capability:
            return ParsedModelRef(
                raw=raw,
                kind=ModelRefKind.CAPABILITY.value,
                capability=capability,
            )
        return ParsedModelRef(raw=raw, kind=ModelRefKind.UNKNOWN.value)

    if raw.startswith("deploy_") and len(raw) > len("deploy_"):
        return ParsedModelRef(
            raw=raw,
            kind=ModelRefKind.DEPLOYMENT.value,
            deployment_id=raw,
        )
    if raw == "deploy_":
        return ParsedModelRef(raw=raw, kind=ModelRefKind.UNKNOWN.value)

    if raw.startswith("exp_") and "/" in raw:
        experiment_id, variant_id = raw.split("/", 1)
        if experiment_id and variant_id:
            return ParsedModelRef(
                raw=raw,
                kind=ModelRefKind.EXPERIMENT.value,
                experiment_id=experiment_id,
                variant_id=variant_id,
            )
        return ParsedModelRef(raw=raw, kind=ModelRefKind.UNKNOWN.value)

    if raw.startswith("alias:") and raw.removeprefix("alias:"):
        return ParsedModelRef(raw=raw, kind=ModelRefKind.ALIAS.value)
    if raw == "alias:":
        return ParsedModelRef(raw=raw, kind=ModelRefKind.UNKNOWN.value)

    if raw.startswith("@") or "://" in raw:
        return ParsedModelRef(raw=raw, kind=ModelRefKind.UNKNOWN.value)

    return ParsedModelRef(
        raw=raw,
        kind=ModelRefKind.MODEL.value,
        model_slug=raw,
    )
