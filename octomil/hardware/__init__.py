"""Hardware detection subsystem for octomil.

Internal module — used by the model optimizer during ``octomil serve``
and ``octomil pull``.  Not part of the public API.
"""

from __future__ import annotations

from ._types import HardwareProfile
from ._unified import detect_hardware

__all__ = [
    "HardwareProfile",
    "detect_hardware",
]
