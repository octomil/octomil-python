"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class ModelCapability(str, Enum):
    CHAT = "chat"
    """Interactive conversational generation. Opens in chat UI."""
    TRANSCRIPTION = "transcription"
    """Audio-to-text speech recognition (e.g. Whisper). Used by voice pipeline."""
    TEXT_COMPLETION = "text_completion"
    """General text continuation or infilling. Not necessarily conversational."""
    KEYBOARD_PREDICTION = "keyboard_prediction"
    """Next-token/word prediction for keyboard suggestion chips."""
    EMBEDDING = "embedding"
    """Vector encoding for similarity search and retrieval."""
    CLASSIFICATION = "classification"
    """Categorization of text or images into labels."""
