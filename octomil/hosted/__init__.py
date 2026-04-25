"""Hosted Octomil SDK client.

Hosted clients call the Octomil cloud control plane (api.octomil.com)
instead of executing on the local device. This package is intentionally
separate from ``octomil.OctomilClient`` (the local-runtime facade) so
hosted workloads do not drag the runtime planner / engine registry into
the import path, and so the constructor stays a simple
``HostedClient(api_key, base_url)``.

For hosted text/chat/embeddings keep using the existing OpenAI-compatible
HTTP layer; this surface is expanded as more hosted endpoints land.
"""

from .audio import HostedAudio
from .client import HostedClient
from .speech import HostedSpeech, SpeechResponse

__all__ = [
    "HostedClient",
    "HostedAudio",
    "HostedSpeech",
    "SpeechResponse",
]
