"""
Compatibility exports for the Python SDK package layout.

Primary SDK code lives in `edgeml/python/edgeml`.
"""

from .python.edgeml import (
    EdgeML,
    EdgeMLClientError,
    ExperimentsAPI,
    Federation,
    FederatedClient,
    ModelRegistry,
    RolloutsAPI,
    DeviceAuthClient,
    compute_state_dict_delta,
)

__all__ = [
    "EdgeML",
    "EdgeMLClientError",
    "Federation",
    "FederatedClient",
    "ModelRegistry",
    "RolloutsAPI",
    "ExperimentsAPI",
    "compute_state_dict_delta",
    "DeviceAuthClient",
]
