"""
Compatibility exports for the Python SDK package layout.

Primary SDK code lives in `octomil/python/octomil`.
"""

from .python.octomil import (
    Octomil,
    OctomilClientError,
    ExperimentsAPI,
    Federation,
    FederatedClient,
    ModelRegistry,
    RolloutsAPI,
    DeviceAuthClient,
    compute_state_dict_delta,
    apply_filters,
)

__all__ = [
    "Octomil",
    "OctomilClientError",
    "Federation",
    "FederatedClient",
    "ModelRegistry",
    "RolloutsAPI",
    "ExperimentsAPI",
    "compute_state_dict_delta",
    "apply_filters",
    "DeviceAuthClient",
]
