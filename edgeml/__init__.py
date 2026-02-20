"""
EdgeML Python SDK.

Serve, deploy, and observe ML models on edge devices.

Primary SDK code lives in `edgeml/python/edgeml`.
Submodules are aliased here so ``from edgeml.secagg import …`` works.
"""

from __future__ import annotations

import importlib as _importlib
import logging as _logging
import os as _os
import sys as _sys
from typing import Optional as _Optional

from .telemetry import TelemetryReporter

from .client import Client
from .enterprise import (
    COMPLIANCE_PRESETS,
    EnterpriseClient,
    EnterpriseClientError,
    get_org_id,
    load_config,
    save_config,
)
from .models import (
    DeploymentPlan,
    DeploymentResult,
    DeviceDeployment,
    DeviceDeploymentStatus,
    RollbackResult,
    TrainingSession,
)
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
    _mod = _sys.modules[_fq]
    _sys.modules[f"edgeml.{_name}"] = _mod
    # Also set as attribute on parent module so getattr() works (required by
    # unittest.mock._dot_lookup on Python <3.12).
    _parts = _name.split(".")
    _parent = _sys.modules[__name__]
    for _part in _parts[:-1]:
        _parent = getattr(_parent, _part, _parent)
    setattr(_parent, _parts[-1], _mod)

# ---------------------------------------------------------------------------
# Module-level telemetry state
# ---------------------------------------------------------------------------

_logger = _logging.getLogger(__name__)

_config: dict[str, str] = {}
_reporter: _Optional[TelemetryReporter] = None


def init(
    api_key: _Optional[str] = None,
    org_id: _Optional[str] = None,
    api_base: _Optional[str] = None,
) -> None:
    """Initialise EdgeML telemetry.

    Call this once at program start to enable automatic inference
    telemetry.  Configuration falls back to environment variables
    ``EDGEML_API_KEY``, ``EDGEML_ORG_ID``, and ``EDGEML_API_BASE``.

    Raises
    ------
    ValueError
        If no API key is provided and ``EDGEML_API_KEY`` is not set.
    """
    global _config, _reporter  # noqa: PLW0603

    resolved_key = api_key if api_key else _os.environ.get("EDGEML_API_KEY", "")
    resolved_org = org_id if org_id else _os.environ.get("EDGEML_ORG_ID", "default")
    resolved_base = (
        api_base
        if api_base
        else _os.environ.get("EDGEML_API_BASE", "https://api.edgeml.io/api/v1")
    )

    if not resolved_key:
        raise ValueError(
            "EdgeML API key required. Pass api_key= or set EDGEML_API_KEY."
        )

    _config = {
        "api_key": resolved_key,
        "org_id": resolved_org,
        "api_base": resolved_base,
    }

    # Validate the key with a lightweight health check
    import httpx as _hx

    try:
        with _hx.Client(timeout=5.0) as _client:
            resp = _client.get(
                f"{resolved_base.rstrip('/')}/health",
                headers={"Authorization": f"Bearer {resolved_key}"},
            )
    except (
        _hx.ConnectError,
        _hx.TimeoutException,
        OSError,
    ):
        _logger.warning(
            "Could not reach EdgeML API at %s — telemetry will retry at event time.",
            resolved_base,
        )
    else:
        if resp.status_code in (401, 403):
            raise ValueError(
                f"Invalid EdgeML API key (HTTP {resp.status_code}). "
                "Check your EDGEML_API_KEY."
            )

    _reporter = TelemetryReporter(
        api_key=resolved_key,
        api_base=resolved_base,
        org_id=resolved_org,
    )
    _logger.info("EdgeML telemetry initialised (org=%s)", resolved_org)


def get_reporter() -> _Optional[TelemetryReporter]:
    """Return the global ``TelemetryReporter``, or ``None`` if :func:`init` has not been called."""
    return _reporter


__all__ = [
    "Client",
    "COMPLIANCE_PRESETS",
    "EnterpriseClient",
    "EnterpriseClientError",
    "get_org_id",
    "load_config",
    "save_config",
    "DeploymentPlan",
    "DeploymentResult",
    "DeviceDeployment",
    "DeviceDeploymentStatus",
    "RollbackResult",
    "TrainingSession",
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
    "TelemetryReporter",
    "init",
    "get_reporter",
]
