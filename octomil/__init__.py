"""
Octomil Python SDK.

Serve, deploy, and observe ML models on edge devices.

Primary SDK code lives in `octomil/python/octomil`.
Submodules are aliased here so ``from octomil.secagg import …`` works.
"""

from __future__ import annotations

__version__ = "4.14.0"

import importlib as _importlib
import logging as _logging
import os as _os
import sys as _sys
from typing import Optional as _Optional

from . import _generated as contracts  # noqa: F401
from . import responses  # noqa: F401
from .auth import AuthConfig, DeviceTokenAuth, OrgApiKeyAuth
from .auth_config import AnonymousAuth, BootstrapTokenAuth, DeviceAuthConfig, PublishableKeyAuth
from .capabilities_client import CapabilitiesClient, CapabilityProfile
from .chat_client import ChatChunk, ChatClient, ChatCompletion
from .client import OctomilClient
from .configure import configure, get_device_context
from .control import DeviceRegistration, HeartbeatResponse, OctomilControl
from .decomposer import (
    DecompositionResult,
    QueryDecomposer,
    ResultMerger,
    SubTask,
    SubTaskResult,
)
from .device_context import DeviceContext, RegistrationState, TokenState
from .embeddings import (
    EmbeddingResult,
    EmbeddingUsage,
    embed,
)
from .enterprise import (
    COMPLIANCE_PRESETS,
    EnterpriseClient,
    EnterpriseClientError,
    get_org_id,
    load_config,
    save_config,
)
from .errors import OctomilError, OctomilErrorCode
from .facade import FacadeEmbeddings, OctomilNotInitializedError  # noqa: F811 — unified facade re-export
from .facade import Octomil as Octomil
from .model import Model, ModelMetadata, Prediction
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
from .monitoring_config import MonitoringConfig
from .routing import (
    DecomposedRoutingDecision,
    ModelInfo,
    QueryRouter,
    RoutingDecision,
    assign_tiers,
)
from .smart_router import RouterConfig, SmartRouter
from .streaming import (
    StreamToken,
    stream_inference,
    stream_inference_async,
)
from .telemetry import TelemetryReporter
from .telemetry_client import TelemetryClient
from .types import (  # noqa: F401
    ArtifactCache,
    FallbackInfo,
    PlannerInfo,
    RouteArtifact,
    RouteExecution,
    RouteMetadata,
    RouteModel,
    RouteModelRequested,
    RouteModelResolved,
    RouteReason,
)

# The inner SDK package has heavy optional deps (torch, cryptography,
# pandas, pyarrow, …) that thin clients (TTS-only embedded callers,
# Ren'Py games, PyInstaller binaries) don't need and often can't
# import — pandas in particular calls ``sysconfig.get_config_var``
# at import time, which crashes inside Ren'Py's bundled CPython.
#
# PR C lazy-loads these legacy/FL symbols via module-level
# ``__getattr__``. Plain ``import octomil`` no longer triggers
# pandas / pyarrow / torch; only an explicit
# ``from octomil import FederatedClient`` (or peeking at
# ``octomil.FederatedClient``) does.
_FROZEN = getattr(_sys, "frozen", False)

# Names exported lazily from ``octomil.python.octomil``. The keys are
# the public names callers see on the ``octomil`` namespace; the
# lazy loader imports the inner package on first access and re-exposes
# the attribute at module level so subsequent lookups skip the import.
_LAZY_LEGACY_EXPORTS = {
    "HKDF_INFO_PAIRWISE_MASK",
    "HKDF_INFO_SELF_MASK",
    "HKDF_INFO_SHARE_ENCRYPTION",
    "SECAGG_PLUS_MOD_RANGE",
    "DataKind",
    "DeltaFilter",
    "DeviceAuthClient",
    "ECKeyPair",
    "ExperimentsAPI",
    "FederatedAnalyticsAPI",
    "FederatedAnalyticsClient",
    "FederatedClient",
    "Federation",
    "FilterRegistry",
    "FilterResult",
    "ModelRegistry",
    "OctomilClientError",
    "RolloutsAPI",
    "SecAggClient",
    "SecAggConfig",
    "SecAggPlusClient",
    "SecAggPlusConfig",
    "apply_filters",
    "compute_state_dict_delta",
    # ``LegacyOctomil`` is the inner package's ``Octomil`` class; we
    # surface it under the legacy alias so prior consumers still work.
    "LegacyOctomil",
}


def __getattr__(name: str):  # noqa: D401 (module-level dunder)
    """Lazy attribute resolver for legacy / FL exports + submodules.

    Triggered ONLY on first access to one of the symbols in
    ``_LAZY_LEGACY_EXPORTS`` or one of the lazy submodules (e.g.
    ``federated_client``). The inner ``octomil.python.octomil``
    package (and its pandas/pyarrow/torch surface) is imported at
    that moment, never at top-level ``import octomil``.
    """
    # Lazy submodule aliasing: ``octomil.federated_client`` resolves
    # to ``octomil.python.octomil.federated_client`` on first
    # access. Subsequent imports go through ``sys.modules`` directly.
    if name in _LAZY_SUBMODULES:
        try:
            return _resolve_lazy_submodule(name)
        except ImportError:
            if _FROZEN:
                raise AttributeError(
                    f"octomil.{name} is not available in this build (frozen "
                    "binary without [fl] extras). Install 'octomil[fl]' to use it."
                ) from None
            raise

    if name not in _LAZY_LEGACY_EXPORTS:
        raise AttributeError(f"module 'octomil' has no attribute {name!r}")

    try:
        from . import python as _python_pkg  # noqa: F401  (forces submodule load)
        from .python import octomil as _legacy_pkg
    except ImportError:
        if _FROZEN:
            # PyInstaller / frozen binaries deliberately strip the
            # heavy deps. Re-raise as AttributeError so feature
            # detection (``hasattr(octomil, 'FederatedClient')``)
            # works cleanly.
            raise AttributeError(
                f"octomil.{name} is not available in this build (frozen binary "
                "without [fl] extras). Install 'octomil[fl]' to use it."
            ) from None
        raise

    if name == "LegacyOctomil":
        value = getattr(_legacy_pkg, "Octomil")
    else:
        value = getattr(_legacy_pkg, name)
    # Cache at module level so the next lookup skips this function.
    globals()[name] = value
    return value


# Submodule aliases for ``from octomil.secagg import …`` ergonomics.
#
# Pre-PR-C, the SDK eagerly imported every entry below at top-level
# ``import octomil``. The trouble is that even a "lightweight" alias
# like ``api_client`` triggers
# ``import octomil.python.octomil.api_client``, which makes Python
# execute the inner package's ``__init__.py`` first — and that
# ``__init__`` itself imports ``federated_client`` + ``data_loader``,
# which import pandas + pyarrow at module load, which crashes Ren'Py
# / certain PyInstaller builds via ``sysconfig.get_config_var``.
#
# Reviewer P1 on PR #455: the eager loop was the residual hop that
# kept thin-client ``import octomil`` reaching the pandas-tainted
# legacy package.
#
# Fix: every submodule alias is lazy now. The names below register
# the SDK ergonomic mapping (``octomil.secagg`` →
# ``octomil.python.octomil.secagg``), but the actual import only
# fires when a caller writes ``from octomil.secagge import X`` or
# accesses ``octomil.secagg``. ``__getattr__`` resolves both.
_LAZY_SUBMODULES = {
    # name → fully-qualified module path inside the inner package.
    # Was previously called ``_EAGER_SUBMODULES`` plus the original
    # lazy set; merged into one table so all aliasing is deferred.
    #
    # Note: ``auth`` is intentionally absent — the outer
    # ``octomil/auth.py`` is the canonical ``octomil.auth`` module
    # (it defines ``AuthConfig`` / ``DeviceTokenAuth`` /
    # ``OrgApiKeyAuth`` re-exported from the top of this file).
    # Aliasing it to the inner package would shadow the canonical
    # one. Inner ``octomil.python.octomil.auth`` is reachable
    # explicitly via that path.
    "api_client": "octomil.python.octomil.api_client",
    "control_plane": "octomil.python.octomil.control_plane",
    "edge": "octomil.python.octomil.edge",
    "federation": "octomil.python.octomil.federation",
    "filters": "octomil.python.octomil.filters",
    "gradient_cache": "octomil.python.octomil.gradient_cache",
    "inference": "octomil.python.octomil.inference",
    "registry": "octomil.python.octomil.registry",
    "resilience": "octomil.python.octomil.resilience",
    "secagg": "octomil.python.octomil.secagg",
    "data_loader": "octomil.python.octomil.data_loader",
    "feature_alignment": "octomil.python.octomil.feature_alignment",
    "feature_alignment.aligner": "octomil.python.octomil.feature_alignment.aligner",
    "federated_client": "octomil.python.octomil.federated_client",
}
# Kept for tests / docs that referenced the historical name.
# Empty: no submodule is eagerly imported any more.
_EAGER_SUBMODULES: list[str] = []


def _resolve_lazy_submodule(name: str):
    """Import + alias one of the deferred-load submodules on demand.

    Called from ``__getattr__`` and from the import-hook below when
    a caller asks for one of the alias names. After this runs once,
    the submodule is registered under ``octomil.<name>`` in
    ``sys.modules`` so subsequent ``from octomil.federated_client
    import …`` lookups go directly through Python's normal import
    machinery without re-entering ``__getattr__``.
    """
    fq = _LAZY_SUBMODULES[name]
    if fq not in _sys.modules:
        _importlib.import_module(fq)
    mod = _sys.modules[fq]
    _sys.modules[f"octomil.{name}"] = mod
    parts = name.split(".")
    parent = _sys.modules[__name__]
    for part in parts[:-1]:
        parent = getattr(parent, part, parent)
    setattr(parent, parts[-1], mod)
    return mod


# ---------------------------------------------------------------------------
# Import hook: make ``import octomil.secagg`` work without eager imports
# ---------------------------------------------------------------------------
#
# Reviewer P1 on PR #455 (post-9d8452c): module-level
# ``__getattr__`` only handles attribute access (``from octomil
# import secagg``); it does NOT make ``import octomil.secagg`` work
# because Python's import machinery looks for an actual submodule
# at that path. Pre-PR-C the eager loop above registered each alias
# in ``sys.modules`` upfront, which kept ``import octomil.secagg``
# working but also forced the pandas-tainted inner ``__init__`` to
# run on every ``import octomil``.
#
# The fix that satisfies both reviewer asks: install a
# ``MetaPathFinder`` that intercepts ``octomil.<alias>`` import
# requests, maps them to ``octomil.python.octomil.<alias>``, and
# returns the inner module — without running anything at top-level.
# Because the inner package's ``__init__`` is itself lazy now, the
# import is genuinely free for thin clients that never touch FL
# surfaces.


class _OctomilAliasLoader:
    """Loader that aliases ``octomil.<name>`` to the inner module
    object stored at ``octomil.python.octomil.<name>``.

    Reviewer P1 on PR #455 (post-7873ae7): the previous shape
    created a fresh spec on every ``find_spec`` call, which let
    Python create *two* module objects for the same alias when
    different import shapes hit the hook (``import octomil.secagg``
    vs ``importlib.import_module('octomil.secagg')``). Class
    identity then forked: ``m1.OctomilClientError is not
    m2.OctomilClientError``, so an exception raised under one
    object couldn't be caught under the other.

    Fix: ``create_module`` returns the canonical inner module
    object directly. Python binds *that exact object* into
    ``sys.modules[fullname]``, so every later import shape gets
    back the same object — there's no second module identity to
    create.
    """

    def __init__(self, inner_fq: str) -> None:
        self._inner_fq = inner_fq

    def create_module(self, spec):
        # Force-import the inner module if it isn't loaded yet.
        # This is the ONLY place we touch the inner package, so
        # there's exactly one canonical module object.
        if self._inner_fq not in _sys.modules:
            _importlib.import_module(self._inner_fq)
        return _sys.modules[self._inner_fq]

    def exec_module(self, module) -> None:
        # Module is the already-executed inner module; nothing
        # to do. Python will bind it under the alias name via
        # ``sys.modules[spec.name] = module`` for us.
        return None


class _OctomilSubmoduleAliasFinder:
    """``sys.meta_path`` finder that maps ``octomil.<alias>`` ↔
    ``octomil.python.octomil.<alias>`` without forcing the inner
    package to eager-import anything.

    Only registers a hit for names in ``_LAZY_SUBMODULES``; every
    other ``octomil.*`` name falls through to Python's normal
    finders so this hook is invisible for ``octomil.execution``,
    ``octomil.runtime``, etc.

    Returns a spec backed by ``_OctomilAliasLoader`` whose
    ``create_module`` returns the SAME inner module object every
    time — no duplicate identities, regardless of how the alias
    is imported.
    """

    @staticmethod
    def find_spec(fullname: str, path=None, target=None):
        if not fullname.startswith("octomil."):
            return None
        suffix = fullname[len("octomil.") :]
        if suffix not in _LAZY_SUBMODULES:
            return None
        # Already aliased — let Python's normal lookup return the
        # cached object directly (no re-execution).
        if fullname in _sys.modules:
            return None
        inner_fq = _LAZY_SUBMODULES[suffix]
        loader = _OctomilAliasLoader(inner_fq)
        # ``origin`` left None: the alias has no own source file.
        return _importlib.util.spec_from_loader(  # type: ignore[union-attr]
            fullname,
            loader=loader,  # type: ignore[arg-type]
            origin=None,
        )


import importlib.util as _importlib_util  # noqa: E402

# Reference held so ``_importlib.util`` resolves under attribute
# access; otherwise type-checkers complain that
# ``_importlib.util`` is unknown.
_importlib.util = _importlib_util  # type: ignore[attr-defined]

# Install the finder on the front of the meta path so it runs
# before the default file finders. Idempotent: re-importing
# ``octomil`` (e.g. through reload in tests) won't stack hooks.
_FINDER_CLASS = _OctomilSubmoduleAliasFinder
if not any(isinstance(f, _FINDER_CLASS) for f in _sys.meta_path):
    _sys.meta_path.insert(0, _FINDER_CLASS())


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
    """Initialise Octomil telemetry.

    Call this once at program start to enable automatic inference
    telemetry.  Configuration falls back to environment variables
    ``OCTOMIL_API_KEY``, ``OCTOMIL_ORG_ID``, and ``OCTOMIL_API_BASE``.

    Raises
    ------
    OctomilError
        If no API key is provided and ``OCTOMIL_API_KEY`` is not set.
    """
    global _config, _reporter  # noqa: PLW0603

    resolved_key = api_key if api_key else _os.environ.get("OCTOMIL_API_KEY", "")
    resolved_org = org_id if org_id else _os.environ.get("OCTOMIL_ORG_ID", "default")
    resolved_base = api_base if api_base else _os.environ.get("OCTOMIL_API_BASE", "https://api.octomil.com/api/v1")

    if not resolved_key:
        raise OctomilError(
            code=OctomilErrorCode.INVALID_API_KEY,
            message="Octomil API key required. Pass api_key= or set OCTOMIL_API_KEY.",
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
            raise OctomilError(
                code=OctomilErrorCode.INVALID_API_KEY,
                message=f"Invalid Octomil API key (HTTP {resp.status_code}). Check your OCTOMIL_API_KEY.",
            )

    _reporter = TelemetryReporter(
        api_key=resolved_key,
        api_base=resolved_base,
        org_id=resolved_org,
    )
    _logger.info("Octomil telemetry initialised (org=%s)", resolved_org)


def get_reporter() -> _Optional[TelemetryReporter]:
    """Return the global ``TelemetryReporter``, or ``None`` if :func:`init` has not been called."""
    return _reporter


__all__ = [
    "__version__",
    "OctomilClient",
    "AuthConfig",
    "OrgApiKeyAuth",
    "DeviceTokenAuth",
    "CapabilitiesClient",
    "CapabilityProfile",
    "ChatClient",
    "ChatCompletion",
    "ChatChunk",
    "TelemetryClient",
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
    "FacadeEmbeddings",
    "LegacyOctomil",
    "OctomilNotInitializedError",
    "OctomilClientError",
    "Federation",
    "FederatedClient",
    "ModelRegistry",
    "RolloutsAPI",
    "ExperimentsAPI",
    "FederatedAnalyticsClient",
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
    "RouterConfig",
    "SmartRouter",
    "OctomilControl",
    "DeviceRegistration",
    "HeartbeatResponse",
    "OctomilError",
    "OctomilErrorCode",
    "PublishableKeyAuth",
    "BootstrapTokenAuth",
    "AnonymousAuth",
    "DeviceAuthConfig",
    "configure",
    "get_device_context",
    "DeviceContext",
    "RegistrationState",
    "TokenState",
    "MonitoringConfig",
    "RouteMetadata",
    "RouteExecution",
    "RouteModel",
    "RouteModelRequested",
    "RouteModelResolved",
    "RouteArtifact",
    "ArtifactCache",
    "PlannerInfo",
    "FallbackInfo",
    "RouteReason",
]
