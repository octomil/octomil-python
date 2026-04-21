"""Canonical model reference classification for routing metadata."""

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_MODEL_REF_KINDS = frozenset(
    {"model", "app", "capability", "deployment", "experiment", "alias", "default", "unknown"}
)


@dataclass(frozen=True)
class ParsedModelRef:
    """Structured classification of a user-provided model reference."""

    raw: str
    kind: str
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
        return ParsedModelRef(raw=raw, kind="default")

    if raw.startswith("@app/"):
        parts = raw.removeprefix("@app/").split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return ParsedModelRef(raw=raw, kind="app", app_slug=parts[0], capability=parts[1])
        return ParsedModelRef(raw=raw, kind="unknown")

    if raw.startswith("@capability/"):
        capability = raw.removeprefix("@capability/")
        if capability:
            return ParsedModelRef(raw=raw, kind="capability", capability=capability)
        return ParsedModelRef(raw=raw, kind="unknown")

    if raw.startswith("deploy_") and len(raw) > len("deploy_"):
        return ParsedModelRef(raw=raw, kind="deployment", deployment_id=raw)

    if raw.startswith("exp_") and "/" in raw:
        experiment_id, variant_id = raw.split("/", 1)
        if experiment_id and variant_id:
            return ParsedModelRef(
                raw=raw,
                kind="experiment",
                experiment_id=experiment_id,
                variant_id=variant_id,
            )
        return ParsedModelRef(raw=raw, kind="unknown")

    if raw.startswith("alias:") and raw.removeprefix("alias:"):
        return ParsedModelRef(raw=raw, kind="alias")

    if raw.startswith("@") or "://" in raw:
        return ParsedModelRef(raw=raw, kind="unknown")

    return ParsedModelRef(raw=raw, kind="model")
