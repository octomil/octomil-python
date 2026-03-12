"""ExecuTorch engine — Meta's on-device runtime (experimental)."""

from octomil.runtime.engines.experimental.executorch.engine import ExecuTorchEngine

TIER = "experimental"

__all__ = ["ExecuTorchEngine", "TIER"]
