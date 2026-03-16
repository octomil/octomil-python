"""Manifest data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from octomil._generated.delivery_mode import DeliveryMode
from octomil._generated.model_capability import ModelCapability


@dataclass
class AppModelEntry:
    """A single model declared in the app manifest."""

    id: str
    capability: ModelCapability
    delivery: DeliveryMode
    required: bool = True
    bundled_path: Optional[str] = None
    download_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    metadata: dict[str, str] = field(default_factory=dict)


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
