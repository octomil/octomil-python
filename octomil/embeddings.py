"""Cloud embeddings via POST /api/v1/embeddings.

Calls the Octomil embeddings endpoint and returns dense vectors
suitable for semantic search, clustering, and RAG pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingUsage:
    """Token usage statistics from the embeddings endpoint."""

    prompt_tokens: int
    total_tokens: int


@dataclass
class EmbeddingResult:
    """Result returned by :func:`embed`."""

    embeddings: list[list[float]]
    model: str
    usage: EmbeddingUsage


def embed(
    server_url: str,
    api_key: str,
    model_id: str,
    input: Union[str, list[str]],
    timeout: float = 30.0,
) -> EmbeddingResult:
    """Generate embeddings via the Octomil cloud endpoint.

    Args:
        server_url: Base URL of the Octomil API (e.g. ``https://api.octomil.com/api/v1``).
        api_key: Bearer token for authentication.
        model_id: Embedding model identifier (e.g. ``"nomic-embed-text"``).
        input: A single string or list of strings to embed.
        timeout: HTTP timeout in seconds.

    Returns:
        :class:`EmbeddingResult` with dense vectors, model name, and usage.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses.
        ValueError: If *server_url* or *api_key* is empty.
    """
    if not server_url:
        raise ValueError("server_url is required for embed()")
    if not api_key:
        raise ValueError("api_key is required for embed()")

    url = f"{server_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model_id": model_id,
        "input": input,
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    embeddings = [item["embedding"] for item in data["data"]]
    usage_raw = data.get("usage", {})

    return EmbeddingResult(
        embeddings=embeddings,
        model=data.get("model", model_id),
        usage=EmbeddingUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        ),
    )
