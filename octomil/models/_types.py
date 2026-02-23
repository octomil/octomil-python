"""Data classes for deploy orchestration, training, and rollback results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DeviceDeployment:
    """Deployment plan for a single device."""

    device_id: str
    format: str
    executor: str
    quantization: str
    download_url: str | None = None
    conversion_needed: bool = False
    runtime_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentPlan:
    """Dry-run result showing what a deployment would do."""

    model_name: str
    model_version: str
    deployments: list[DeviceDeployment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceDeploymentStatus:
    """Status of a deployment to a single device."""

    device_id: str
    status: str
    download_url: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentResult:
    """Result of executing a deployment."""

    deployment_id: str
    model_name: str
    model_version: str
    status: str
    device_statuses: list[DeviceDeploymentStatus] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingSession:
    """Result of creating a federated training session."""

    session_id: str
    model_name: str
    group: str
    strategy: str
    rounds: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RollbackResult:
    """Result of a model rollback."""

    model_name: str
    from_version: str
    to_version: str
    rollout_id: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
