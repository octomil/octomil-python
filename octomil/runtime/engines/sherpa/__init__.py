"""sherpa-onnx engine — on-device speech synthesis (and ASR, future)."""

from octomil.runtime.engines.sherpa.engine import (
    _KOKORO_VOICES,
    SherpaTtsEngine,
    is_sherpa_tts_model,
    is_sherpa_tts_model_staged,
)

TIER = "supported"

__all__ = [
    "SherpaTtsEngine",
    "TIER",
    "is_sherpa_tts_model",
    "is_sherpa_tts_model_staged",
    "_KOKORO_VOICES",
]
