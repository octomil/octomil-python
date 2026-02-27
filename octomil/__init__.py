"""
Octomil Python SDK.

Serve, deploy, and observe ML models on edge devices.

Primary SDK code lives in `octomil/python/octomil`.
Submodules are aliased here so ``from octomil.secagg import …`` works.
"""

from __future__ import annotations

__version__ = "2.6.0"

import importlib as _importlib
import logging as _logging
import os as _os
import sys as _sys
from typing import Optional as _Optional

from .client import Client
from .model import Model, ModelMetadata, Prediction
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
    MoEMetadata,
    RollbackResult,
    TrainingSession,
    get_moe_metadata,
    is_moe_model,
    list_moe_models,
)
from .decomposer import (
    DecompositionResult,
    QueryDecomposer,
    ResultMerger,
    SubTask,
    SubTaskResult,
)
from .routing import (
    DecomposedRoutingDecision,
    ModelInfo,
    QueryRouter,
    RoutingDecision,
    assign_tiers,
)
from .embeddings import (
    EmbeddingResult,
    EmbeddingUsage,
    embed,
)
from .streaming import (
    StreamToken,
    stream_inference,
    stream_inference_async,
)

# The inner SDK package has heavy optional deps (torch, cryptography, etc.)
# that are not bundled in the standalone CLI binary (PyInstaller).
# Only suppress ImportError when running as a frozen binary.
_FROZEN = getattr(_sys, "frozen", False)

try:
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
except ImportError:
    if not _FROZEN:
        raise

# ---------------------------------------------------------------------------
# Lazy submodule aliasing
# ---------------------------------------------------------------------------
# ``from octomil.secagg import …`` must resolve to
# ``octomil.python.octomil.secagg``.  Rather than eagerly importing every
# submodule (~240ms for pandas + httpx), we register lazy aliases that only
# import the real module on first attribute access.

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
    "gradient_cache",
    "inference",
    "registry",
    "resilience",
    "secagg",
]


class _LazyModule:
    """Proxy that imports the real module on first attribute access.

    Installed into ``sys.modules`` so ``from octomil.secagg import X`` works
    without eagerly importing ``octomil.python.octomil.secagg``.
    """

    def __init__(self, alias: str, real_fq: str) -> None:
        self.__alias = alias
        self.__real_fq = real_fq
        self.__mod = None

    def _load(self) -> object:
        if self.__mod is None:
            self.__mod = _importlib.import_module(self.__real_fq)
            # Replace ourselves in sys.modules with the real module
            _sys.modules[self.__alias] = self.__mod
        return self.__mod

    def __getattr__(self, name: str) -> object:
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        return f"<lazy alias {self.__alias!r} -> {self.__real_fq!r}>"


for _name in _SUBMODULES:
    _alias = f"octomil.{_name}"
    _fq = f"octomil.python.octomil.{_name}"
    # If already eagerly imported (e.g. via the from .python.octomil import block),
    # just alias it.  Otherwise install a lazy proxy.
    if _fq in _sys.modules:
        _sys.modules[_alias] = _sys.modules[_fq]
    else:
        _sys.modules[_alias] = _LazyModule(_alias, _fq)  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Lazy attribute access for deferred imports
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> object:
    if name == "TelemetryReporter":
        from .telemetry import TelemetryReporter

        globals()["TelemetryReporter"] = TelemetryReporter
        return TelemetryReporter
    # Support lazy submodule access as attributes (e.g. octomil.secagg)
    _alias = f"octomil.{name}"
    if _alias in _sys.modules:
        _mod = _sys.modules[_alias]
        if isinstance(_mod, _LazyModule):
            _mod = _mod._load()
        globals()[name] = _mod
        return _mod
    raise AttributeError(f"module 'octomil' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Module-level telemetry state
# ---------------------------------------------------------------------------

_logger = _logging.getLogger(__name__)

_config: dict[str, str] = {}
_reporter: _Optional["TelemetryReporter"] = None


def init(
    api_key: _Optional[str] = None,
    org_id: _Optional[str] = None,
    api_base: _Optional[str] = None,
) -> None:
    """Initialise Octomil telemetry.

    Call this once at program start to enable automatic inference
    telemetry.  Configuration falls back to environment variables
    ``OCTOMIL_API_KEY``, ``OCTOMIL_ORG_ID``, and ``OCTOMIL_API_BASE``.

    Raises
    ------
    ValueError
        If no API key is provided and ``OCTOMIL_API_KEY`` is not set.
    """
    global _config, _reporter  # noqa: PLW0603

    resolved_key = api_key if api_key else _os.environ.get("OCTOMIL_API_KEY", "")
    resolved_org = org_id if org_id else _os.environ.get("OCTOMIL_ORG_ID", "default")
    resolved_base = (
        api_base
        if api_base
        else _os.environ.get("OCTOMIL_API_BASE", "https://api.octomil.com/api/v1")
    )

    if not resolved_key:
        raise ValueError(
            "Octomil API key required. Pass api_key= or set OCTOMIL_API_KEY."
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
            "Could not reach Octomil API at %s — telemetry will retry at event time.",
            resolved_base,
        )
    else:
        if resp.status_code in (401, 403):
            raise ValueError(
                f"Invalid Octomil API key (HTTP {resp.status_code}). "
                "Check your OCTOMIL_API_KEY."
            )

    from .telemetry import TelemetryReporter

    _reporter = TelemetryReporter(
        api_key=resolved_key,
        api_base=resolved_base,
        org_id=resolved_org,
    )
    _logger.info("Octomil telemetry initialised (org=%s)", resolved_org)


def get_reporter() -> _Optional["TelemetryReporter"]:
    """Return the global ``TelemetryReporter``, or ``None`` if :func:`init` has not been called."""
    return _reporter


__all__ = [
    "__version__",
    "Client",
    "Model",
    "ModelMetadata",
    "Prediction",
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
    "MoEMetadata",
    "get_moe_metadata",
    "is_moe_model",
    "list_moe_models",
    "TelemetryReporter",
    "init",
    "get_reporter",
    "DecomposedRoutingDecision",
    "DecompositionResult",
    "ModelInfo",
    "QueryDecomposer",
    "QueryRouter",
    "ResultMerger",
    "RoutingDecision",
    "SubTask",
    "SubTaskResult",
    "assign_tiers",
    "StreamToken",
    "stream_inference",
    "stream_inference_async",
    "EmbeddingResult",
    "EmbeddingUsage",
    "embed",
]
