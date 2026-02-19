"""
EdgeML Python SDK.

Serve, deploy, and observe ML models on edge devices.

Primary SDK code lives in `edgeml/python/edgeml`.
Submodules are aliased here so ``from edgeml.secagg import …`` works.
"""

import importlib as _importlib
import sys as _sys

from .client import Client
from .python.edgeml import (
    EdgeML,
    EdgeMLClientError,
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

# Alias inner submodules so ``from edgeml.secagg import …`` works without
# requiring users to know about the nested ``edgeml.python.edgeml`` layout.
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
    _fq = f"edgeml.python.edgeml.{_name}"
    if _fq not in _sys.modules:
        try:
            _importlib.import_module(_fq)
        except ImportError:
            continue
    _sys.modules[f"edgeml.{_name}"] = _sys.modules[_fq]

__all__ = [
    "Client",
    "EdgeML",
    "EdgeMLClientError",
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
