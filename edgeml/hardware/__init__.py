"""Hardware detection subsystem for edgeml.

Internal module — used by the model optimizer during ``edgeml serve``
and ``edgeml pull``.  Not part of the public API.
"""

from __future__ import annotations

from ._types import HardwareProfile
from ._unified import detect_hardware

__all__ = [
    "HardwareProfile",
    "detect_hardware",
]
