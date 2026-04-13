"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class CloudProvider(str, Enum):
    OCTOMIL = "octomil"
    """Octomil-hosted inference (local or cloud). OpenAI-compatible protocol."""
    OPENAI = "openai"
    """OpenAI cloud inference. OpenAI-compatible protocol."""
    ANTHROPIC = "anthropic"
    """Anthropic cloud inference (Claude). Anthropic Messages API protocol."""
    GROQ = "groq"
    """Groq cloud inference. OpenAI-compatible protocol."""
    TOGETHER = "together"
    """Together AI cloud inference. OpenAI-compatible protocol."""
    MOONSHOT = "moonshot"
    """Moonshot AI cloud inference. OpenAI-compatible protocol."""
    MINIMAX = "minimax"
    """MiniMax cloud inference. OpenAI-compatible protocol."""
    DEEPSEEK = "deepseek"
    """DeepSeek cloud inference. OpenAI-compatible protocol."""
    UNKNOWN = "unknown"
    """Catch-all for unrecognized provider strings. SDKs MUST map unrecognized values here."""
