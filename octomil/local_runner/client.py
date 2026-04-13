"""HTTP client for the local runner.

Uses httpx (already a core dependency) to talk to the local runner
server over ``127.0.0.1``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LocalRunnerClient:
    """Async HTTP client for the invisible local runner server."""

    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Responses
    # ------------------------------------------------------------------

    async def create_response(
        self,
        *,
        model: str,
        input: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call /v1/chat/completions on the local runner.

        Wraps the prompt into a chat completion request for the
        OpenAI-compatible endpoint exposed by the inner serve app.
        """
        import httpx

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": input}],
            "stream": stream,
        }
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            payload["max_tokens"] = kwargs["max_tokens"]
        if "max_output_tokens" in kwargs and kwargs["max_output_tokens"] is not None:
            payload["max_tokens"] = kwargs["max_output_tokens"]

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def create_embedding(
        self,
        *,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call /v1/embeddings on the local runner."""
        import httpx

        payload: dict[str, Any] = {
            "model": model,
            "input": input,
        }
        payload.update(kwargs)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/v1/embeddings",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Audio transcription
    # ------------------------------------------------------------------

    async def create_transcription(
        self,
        *,
        model: str,
        file_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call /v1/audio/transcriptions on the local runner."""
        from pathlib import Path

        import httpx

        audio_path = Path(file_path)
        filename = audio_path.name

        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"file": (filename, audio_path.read_bytes())}
            data: dict[str, str] = {}
            if model:
                data["model"] = model
            for k, v in kwargs.items():
                if v is not None:
                    data[k] = str(v)

            resp = await client.post(
                f"{self._base_url}/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._token}"},
                files=files,
                data=data,
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Health / Shutdown
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """Check if the runner is healthy."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def shutdown(self) -> bool:
        """Request graceful shutdown. Returns True on success."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self._base_url}/shutdown",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            return False
