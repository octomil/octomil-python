"""
Octomil Python SDK.

Serve, deploy, and observe ML models on edge devices.

Primary SDK code lives in `octomil/python/octomil`.
Submodules are aliased here so ``from octomil.secagg import …`` works.
"""

import importlib as _importlib
import sys as _sys

from .client import Client
from .models import (
    DeploymentPlan,
    DeploymentResult,
    DeviceDeployment,
    DeviceDeploymentStatus,
    RollbackResult,
    TrainingSession,
)
from .python.octomil import (
    Octomil,
    OctomilClientError,
    ExperimentsAPI,
    FederatedAnalyticsAPI,
    Federation,
    FederatedClient,
    ModelRegistry,
    RolloutsAPI,
    DeviceAuthClient,
    compute_state_dict_delta,
    apply_filters,
    DataKind,
    DeltaFilter,
    FilterRegistry,
    FilterResult,
    ECKeyPair,
    SecAggClient,
    SecAggConfig,
    SecAggPlusClient,
    SecAggPlusConfig,
    SECAGG_PLUS_MOD_RANGE,
    HKDF_INFO_PAIRWISE_MASK,
    HKDF_INFO_SHARE_ENCRYPTION,
    HKDF_INFO_SELF_MASK,
)

# Alias inner submodules so ``from octomil.secagg import …`` works without
# requiring users to know about the nested ``octomil.python.octomil`` layout.
_SUBMODULES = [
    "api_client",
    "auth",
    "control_plane",
    "data_loader",
    "edge",
    "feature_alignment",
    "feature_alignment.aligner",
    "federated_client",
    "federation",
    "filters",
    "inference",
    "registry",
    "secagg",
]

for _name in _SUBMODULES:
    _fq = f"octomil.python.octomil.{_name}"
    if _fq not in _sys.modules:
        try:
            _importlib.import_module(_fq)
        except ImportError:
            continue
    _mod = _sys.modules[_fq]
    _sys.modules[f"octomil.{_name}"] = _mod
    # Also set as attribute on parent module so getattr() works (required by
    # unittest.mock._dot_lookup on Python <3.12).
    _parts = _name.split(".")
    _parent = _sys.modules[__name__]
    for _part in _parts[:-1]:
        _parent = getattr(_parent, _part, _parent)
    setattr(_parent, _parts[-1], _mod)

__all__ = [
    "Client",
    "DeploymentPlan",
    "DeploymentResult",
    "DeviceDeployment",
    "DeviceDeploymentStatus",
    "RollbackResult",
    "TrainingSession",
    "Octomil",
    "OctomilClientError",
    "Federation",
    "FederatedClient",
    "ModelRegistry",
    "RolloutsAPI",
    "ExperimentsAPI",
    "FederatedAnalyticsAPI",
    "compute_state_dict_delta",
    "apply_filters",
    "DeviceAuthClient",
    "DataKind",
    "DeltaFilter",
    "FilterRegistry",
    "FilterResult",
    "ECKeyPair",
    "SecAggClient",
    "SecAggConfig",
    "SecAggPlusClient",
    "SecAggPlusConfig",
    "SECAGG_PLUS_MOD_RANGE",
    "HKDF_INFO_PAIRWISE_MASK",
    "HKDF_INFO_SHARE_ENCRYPTION",
    "HKDF_INFO_SELF_MASK",
]
