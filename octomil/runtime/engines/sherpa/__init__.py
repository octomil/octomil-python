"""sherpa-onnx engine — on-device speech synthesis (and ASR, future)."""

from octomil.runtime.engines.sherpa.engine import (
    SherpaTtsEngine,
    is_sherpa_tts_model,
)

TIER = "supported"

__all__ = ["SherpaTtsEngine", "TIER", "is_sherpa_tts_model"]
