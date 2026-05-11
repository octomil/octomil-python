"""Auto-generated from octomil-contracts runtime_capability.json. Do not edit.

Source of truth for capability strings used in BOTH directions of the runtime ABI:
  (a) advertised via oct_runtime_capabilities().supported_capabilities[]
  (b) requested via oct_session_config_t.capability
"""

from enum import Enum


class RuntimeCapability(str, Enum):
    AUDIO_DIARIZATION = "audio.diarization"
    AUDIO_REALTIME_SESSION = "audio.realtime.session"
    AUDIO_SPEAKER_EMBEDDING = "audio.speaker.embedding"
    AUDIO_STT_BATCH = "audio.stt.batch"
    AUDIO_STT_STREAM = "audio.stt.stream"
    AUDIO_TRANSCRIPTION = "audio.transcription"
    AUDIO_TTS_BATCH = "audio.tts.batch"
    AUDIO_TTS_STREAM = "audio.tts.stream"
    AUDIO_VAD = "audio.vad"
    CACHE_INTROSPECT = "cache.introspect"
    CHAT_COMPLETION = "chat.completion"
    CHAT_STREAM = "chat.stream"
    EMBEDDINGS_IMAGE = "embeddings.image"
    EMBEDDINGS_TEXT = "embeddings.text"
    INDEX_VECTOR_QUERY = "index.vector.query"
