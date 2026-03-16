"""App manifest types for declaring model requirements.

An :class:`AppManifest` describes the models an application needs.
It is consumed by iOS and Android SDKs to provision local or cloud-based
inference at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from octomil._generated.delivery_mode import DeliveryMode
from octomil._generated.model_capability import ModelCapability
from octomil._generated.routing_policy import RoutingPolicy as AppRoutingPolicy

# ---------------------------------------------------------------------------
# Known capability values (derived from contracts-generated enum)
# ---------------------------------------------------------------------------

KNOWN_CAPABILITIES: frozenset[str] = frozenset(m.value for m in ModelCapability)


# ---------------------------------------------------------------------------
# Model entry
# ---------------------------------------------------------------------------


@dataclass
class AppModelEntry:
    """A single model declaration inside an :class:`AppManifest`."""

    id: str
    capability: str
    delivery: DeliveryMode
    routing_policy: Optional[AppRoutingPolicy] = None
    bundled_path: Optional[str] = None
    required: bool = False

    @property
    def effective_routing_policy(self) -> AppRoutingPolicy:
        if self.routing_policy is not None:
            return self.routing_policy
        if self.delivery in (DeliveryMode.BUNDLED, DeliveryMode.MANAGED):
            return AppRoutingPolicy.LOCAL_FIRST
        return AppRoutingPolicy.CLOUD_ONLY

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict, omitting ``None`` optional fields."""
        d: dict[str, Any] = {
            "id": self.id,
            "capability": self.capability,
            "delivery": self.delivery.value,
        }
        if self.routing_policy is not None:
            d["routing_policy"] = self.routing_policy.value
        if self.bundled_path is not None:
            d["bundled_path"] = self.bundled_path
        if self.required:
            d["required"] = self.required
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppModelEntry:
        delivery = DeliveryMode(data["delivery"])
        rp_raw = data.get("routing_policy")
        routing_policy = AppRoutingPolicy(rp_raw) if rp_raw else None
        return cls(
            id=data["id"],
            capability=data["capability"],
            delivery=delivery,
            routing_policy=routing_policy,
            bundled_path=data.get("bundled_path"),
            required=data.get("required", False),
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class AppManifest:
    """Top-level manifest describing the models an app requires."""

    models: list[AppModelEntry] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "models": [m.to_dict() for m in self.models],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        """Serialise to YAML.  Uses PyYAML if available, otherwise a simple emitter."""
        try:
            import yaml  # type: ignore[import-untyped]

            return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        except ImportError:  # pragma: no cover
            return _simple_yaml_dump(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppManifest:
        version = data.get("version", 1)
        models = [AppModelEntry.from_dict(m) for m in data.get("models", [])]
        return cls(models=models, version=version)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppManifest:
        """Load a manifest from a YAML file."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML manifests: pip install pyyaml") from exc

        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping at top level, got {type(data).__name__}")
        return cls.from_dict(data)

    def validate(self) -> list[str]:
        """Return a list of validation errors.  Empty list means valid."""
        errors: list[str] = []
        seen_capabilities: set[str] = set()

        for i, entry in enumerate(self.models):
            prefix = f"models[{i}] (id={entry.id!r})"

            # Validate delivery enum
            if not isinstance(entry.delivery, DeliveryMode):
                try:
                    DeliveryMode(entry.delivery)
                except ValueError:
                    errors.append(f"{prefix}: invalid delivery mode {entry.delivery!r}")

            # Bundled entries must have bundled_path
            if entry.delivery == DeliveryMode.BUNDLED and not entry.bundled_path:
                errors.append(f"{prefix}: bundled delivery requires 'bundled_path'")

            # Capability must be a known value
            if entry.capability not in KNOWN_CAPABILITIES:
                errors.append(
                    f"{prefix}: unknown capability {entry.capability!r}; "
                    f"expected one of {sorted(KNOWN_CAPABILITIES)}"
                )

            # Duplicate capabilities
            if entry.capability in seen_capabilities:
                errors.append(f"{prefix}: duplicate capability {entry.capability!r}")
            seen_capabilities.add(entry.capability)

        return errors


# ---------------------------------------------------------------------------
# Minimal YAML emitter (no PyYAML dependency)
# ---------------------------------------------------------------------------


def _simple_yaml_dump(data: dict[str, Any]) -> str:
    """Best-effort YAML serialisation for simple manifest dicts."""
    lines: list[str] = []
    _dump_value(data, lines, indent=0)
    return "\n".join(lines) + "\n"


def _dump_value(value: Any, lines: list[str], indent: int) -> None:
    prefix = "  " * indent
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                _dump_value(v, lines, indent + 1)
            else:
                lines.append(f"{prefix}{k}: {_scalar(v)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                first = True
                for k, v in item.items():
                    bullet = "- " if first else "  "
                    first = False
                    lines.append(f"{prefix}{bullet}{k}: {_scalar(v)}")
            else:
                lines.append(f"{prefix}- {_scalar(item)}")
    else:
        lines.append(f"{prefix}{_scalar(value)}")


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if any(c in value for c in ":#{}[]&*?|>-!%@`"):
            return f'"{value}"'
        return value
    return str(value)
