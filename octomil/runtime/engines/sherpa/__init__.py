"""sherpa-onnx engine — on-device speech synthesis (and ASR, future)."""

from octomil.runtime.engines.sherpa.engine import (
    _KOKORO_VOICES,
    SherpaTtsEngine,
    catalog_for_model,
    is_sherpa_tts_model,
    is_sherpa_tts_runtime_available,
)

TIER = "supported"

__all__ = [
    "SherpaTtsEngine",
    "TIER",
    "catalog_for_model",
    "is_sherpa_tts_model",
    "is_sherpa_tts_runtime_available",
    "_KOKORO_VOICES",
]
