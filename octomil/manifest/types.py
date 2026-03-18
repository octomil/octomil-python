"""Manifest data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from octomil._generated.artifact_resource_kind import ArtifactResourceKind
from octomil._generated.delivery_mode import DeliveryMode
from octomil._generated.modality import Modality
from octomil._generated.model_capability import ModelCapability


@dataclass
class ResourceBinding:
    """Maps an ArtifactResourceKind to a resolved file path and metadata."""

    kind: ArtifactResourceKind
    uri: str
    path: str = ""
    size_bytes: int = 0
    checksum_sha256: str = ""
    required: bool = True
    load_order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppModelEntry:
    """A single model declared in the app manifest."""

    id: str
    capability: ModelCapability
    delivery: DeliveryMode
    input_modalities: list[Modality]
    output_modalities: list[Modality]
    required: bool = True
    bundled_path: Optional[str] = None
    download_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    resource_bindings: list[ResourceBinding] = field(default_factory=list)
    engine_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def binding_for(self, kind: ArtifactResourceKind) -> Optional[ResourceBinding]:
        """Find the first resource binding matching a kind."""
        for binding in self.resource_bindings:
            if binding.kind == kind:
                return binding
        return None

    def bindings_for(self, kind: ArtifactResourceKind) -> list[ResourceBinding]:
        """Find all resource bindings matching a kind."""
        return [b for b in self.resource_bindings if b.kind == kind]


@dataclass
class AppManifest:
    """Declarative manifest describing all models the app needs."""

    models: list[AppModelEntry] = field(default_factory=list)

    def entry_for(self, capability: ModelCapability) -> Optional[AppModelEntry]:
        """Find the first entry matching a capability."""
        for entry in self.models:
            if entry.capability == capability:
                return entry
        return None

    def entry_by_id(self, model_id: str) -> Optional[AppModelEntry]:
        """Find an entry by model ID."""
        for entry in self.models:
            if entry.id == model_id:
                return entry
        return None
