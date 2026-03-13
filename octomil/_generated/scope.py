"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class Scope(str, Enum):
    CATALOG_READ = "catalog:read"
    """Read model catalog and metadata"""
    MODELS_READ = "models:read"
    """Read model artifacts and versions"""
    MODELS_WRITE = "models:write"
    """Create, update, or delete models and versions"""
    DEVICES_REGISTER = "devices:register"
    """Register a new device"""
    DEVICES_HEARTBEAT = "devices:heartbeat"
    """Send device heartbeats"""
    CONTROL_REFRESH = "control:refresh"
    """Refresh control plane assignments"""
    TELEMETRY_WRITE = "telemetry:write"
    """Write telemetry data (spans, metrics, logs)"""
    ROLLOUTS_READ = "rollouts:read"
    """Read rollout configurations"""
    ROLLOUTS_WRITE = "rollouts:write"
    """Create, update, or delete rollouts"""
    BENCHMARKS_WRITE = "benchmarks:write"
    """Submit benchmark results"""
    EVALS_WRITE = "evals:write"
    """Submit evaluation results"""
