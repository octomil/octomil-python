"""Inner SDK package — heavy FL / analytics / training surface.

PR C reviewer P1: pre-existing eager imports here triggered pandas /
pyarrow / torch on every ``import octomil`` because the outer
``octomil/__init__.py`` (and ``octomil.client``) reach into this
package for runtime types. That cascade crashed Ren'Py / sandboxed
CPython / some PyInstaller builds via pandas's
``sysconfig.get_config_var`` quirk.

Fix: every export here is lazy. The ``__all__`` table tells callers
which names are reachable; module-level ``__getattr__`` resolves
each name on first access by importing the specific submodule that
defines it. ``import octomil.python.octomil`` itself runs no heavy
code; only ``octomil.python.octomil.FederatedClient`` (or any other
attribute access) imports ``federated_client`` (which imports
pandas) at that moment.

A submodule entry maps each public name to the dotted path that
defines it. ``LegacyOctomil`` (alias for ``Octomil`` in
``edge.py``) follows the same shape via the outer-module lazy
table.
"""

from __future__ import annotations

# Public names this package re-exports. Same surface as before; the
# difference is that retrieving any one of these triggers exactly
# one inner import instead of every submodule.
__all__ = [
    "Octomil",
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
    "DataKind",
    "DeltaFilter",
    "FilterRegistry",
    "FilterResult",
    "DeviceAuthClient",
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


# name → (submodule, attribute). When the attribute name on the
# submodule matches the public name, the second element is None.
_LAZY_NAME_MAP: dict[str, tuple[str, str | None]] = {
    "OctomilClientError": ("api_client", None),
    "DeviceAuthClient": ("auth", None),
    "ExperimentsAPI": ("control_plane", None),
    "FederatedAnalyticsAPI": ("control_plane", None),
    "FederatedAnalyticsClient": ("control_plane", None),
    "RolloutsAPI": ("control_plane", None),
    "Octomil": ("edge", None),
    "FederatedClient": ("federated_client", None),
    "apply_filters": ("federated_client", None),
    "compute_state_dict_delta": ("federated_client", None),
    "Federation": ("federation", None),
    "DataKind": ("filters", None),
    "DeltaFilter": ("filters", None),
    "FilterRegistry": ("filters", None),
    "FilterResult": ("filters", None),
    "ModelRegistry": ("registry", None),
    "HKDF_INFO_PAIRWISE_MASK": ("secagg", None),
    "HKDF_INFO_SELF_MASK": ("secagg", None),
    "HKDF_INFO_SHARE_ENCRYPTION": ("secagg", None),
    "SECAGG_PLUS_MOD_RANGE": ("secagg", None),
    "ECKeyPair": ("secagg", None),
    "SecAggClient": ("secagg", None),
    "SecAggConfig": ("secagg", None),
    "SecAggPlusClient": ("secagg", None),
    "SecAggPlusConfig": ("secagg", None),
}


def __getattr__(name: str):  # noqa: D401  (module-level dunder)
    """Lazy export resolver for the inner package.

    Pre-PR-C this module's top-level ran ``from .edge import
    Octomil`` and ``from .federated_client import …`` — both of
    which import pandas + pyarrow at module load. Thin TTS
    callers that only ``import octomil`` for the speech facade
    paid the full FL / analytics import cost (and crashed on
    Ren'Py / sandboxed CPython where pandas's
    ``sysconfig.get_config_var`` is patched).

    The resolver imports exactly one submodule per requested
    attribute, caches the resulting object on this module, and
    returns it. Subsequent accesses skip ``__getattr__`` entirely
    via Python's normal attribute lookup.
    """
    target = _LAZY_NAME_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'octomil.python.octomil' has no attribute {name!r}")
    submodule, attr = target
    from importlib import import_module

    mod = import_module(f".{submodule}", __name__)
    value = getattr(mod, attr or name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """``dir(octomil.python.octomil)`` should still surface the
    public names so REPL completion works."""
    return sorted(set(globals().keys()) | set(__all__))
