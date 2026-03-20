"""Shared OpenAI-compatible chat completions client.

Used by both CloudModelRuntime (runtime layer) and
CloudInferenceBackend (serve layer) to avoid duplicating
HTTP transport, SSE parsing, retry logic, and error handling.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

from octomil.errors import OctomilError, OctomilErrorCode

logger = logging.getLogger(__name__)

# Retryable status codes
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class CloudClient:
    """OpenAI-compatible chat completions client (shared transport)."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(connect=60.0, read=120.0, write=30.0, pool=30.0),
        )

    @property
    def model(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion. Returns parsed JSON response.

        Retries once on 429/5xx.
        """
        body = self._build_body(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            stream=False,
        )

        response = await self._post_with_retry("/chat/completions", body)
        return response.json()

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat completion. Yields parsed delta dicts from SSE.

        No retry after first byte received.
        """
        body = self._build_body(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            stream=True,
        )

        response: Optional[httpx.Response] = None
        try:
            response = await self._post_stream("/chat/completions", body)
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]  # strip "data: " prefix
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed SSE chunk: %.100s", data)
                    continue
                yield chunk
        finally:
            if response is not None:
                await response.aclose()

    async def list_models(self) -> list[str]:
        """GET /models, return model IDs."""
        response = await self._client.get("/models")
        _raise_for_status(response)
        data = response.json()
        return [m["id"] for m in data.get("data", [])]

    async def close(self) -> None:
        await self._client.aclose()

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: Optional[list[dict[str, Any]]],
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
        return body

    async def _post_with_retry(
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_retries: int = 1,
    ) -> httpx.Response:
        """POST with retry on retryable status codes."""
        last_response: Optional[httpx.Response] = None
        for attempt in range(1 + max_retries):
            response = await self._client.post(path, json=body)
            if response.status_code not in _RETRYABLE_STATUSES:
                _raise_for_status(response)
                return response
            last_response = response
            if attempt < max_retries:
                retry_after = _parse_retry_after(response)
                if retry_after > 0:
                    import asyncio

                    await asyncio.sleep(min(retry_after, 10.0))
                else:
                    import asyncio

                    await asyncio.sleep(1.0 * (attempt + 1))
                logger.info(
                    "Retrying cloud request (attempt %d/%d, status %d)",
                    attempt + 2,
                    1 + max_retries,
                    response.status_code,
                )

        # Exhausted retries
        assert last_response is not None
        _raise_for_status(last_response)
        return last_response  # unreachable but satisfies type checker

    async def _post_stream(self, path: str, body: dict[str, Any]) -> httpx.Response:
        """POST for streaming. Returns the response for line-by-line iteration.

        Retries once if the connection fails before any bytes arrive.
        """
        try:
            response = await self._client.send(
                self._client.build_request("POST", path, json=body),
                stream=True,
            )
            _raise_for_status(response)
            return response
        except (httpx.ConnectError, httpx.TimeoutException):
            # One retry on connection-level failure (no bytes sent yet)
            logger.info("Retrying cloud stream request after connection failure")
            response = await self._client.send(
                self._client.build_request("POST", path, json=body),
                stream=True,
            )
            _raise_for_status(response)
            return response


def _raise_for_status(response: httpx.Response) -> None:
    """Raise OctomilError for HTTP error responses."""
    if response.is_success:
        return

    status = response.status_code
    try:
        error_body = response.json()
        error_msg = error_body.get("error", {}).get("message", response.text)
    except Exception:
        error_msg = response.text

    if status == 401:
        raise OctomilError(
            code=OctomilErrorCode.AUTHENTICATION_FAILED,
            message=f"Cloud API authentication failed: {error_msg}",
        )
    if status == 429:
        raise OctomilError(
            code=OctomilErrorCode.RATE_LIMITED,
            message=f"Cloud API rate limited: {error_msg}",
        )
    raise OctomilError(
        code=OctomilErrorCode.SERVER_ERROR,
        message=f"Cloud API error ({status}): {error_msg}",
    )


def _parse_retry_after(response: httpx.Response) -> float:
    """Parse Retry-After header, return seconds (0 if absent/invalid)."""
    val = response.headers.get("retry-after")
    if val is None:
        return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0
