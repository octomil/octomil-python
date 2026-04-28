"""sherpa-onnx engine — on-device speech synthesis (and ASR, future)."""

from octomil.runtime.engines.sherpa.engine import (
    _KOKORO_VOICES,
    ResolvedVoiceCatalog,
    SherpaTtsEngine,
    catalog_for_model,
    fallback_catalog_for_artifact,
    is_sherpa_tts_model,
    is_sherpa_tts_runtime_available,
    resolve_voice_catalog,
)

TIER = "supported"

__all__ = [
    "SherpaTtsEngine",
    "TIER",
    "ResolvedVoiceCatalog",
    "catalog_for_model",
    "fallback_catalog_for_artifact",
    "is_sherpa_tts_model",
    "is_sherpa_tts_runtime_available",
    "resolve_voice_catalog",
    "_KOKORO_VOICES",
]
