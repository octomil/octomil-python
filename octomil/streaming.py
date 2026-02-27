"""Cloud streaming inference via Server-Sent Events (SSE).

Consumes ``POST /api/v1/inference/stream`` and yields ``StreamToken``
objects as they arrive.  Designed for cloud-routed inference — local
on-device streaming uses the separate ``inference`` module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, Optional, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class StreamToken:
    """A single token received from the streaming inference endpoint."""

    token: str
    done: bool
    provider: Optional[str] = None
    latency_ms: Optional[float] = None
    session_id: Optional[str] = None


def stream_inference(
    server_url: str,
    api_key: str,
    model_id: str,
    input_data: Union[str, list[dict[str, str]]],
    parameters: Optional[dict[str, Any]] = None,
    timeout: float = 120.0,
) -> Iterator[StreamToken]:
    """Stream tokens from the cloud inference endpoint (sync).

    Args:
        server_url: Base URL of the Octomil API (e.g. ``https://api.octomil.com/api/v1``).
        api_key: Bearer token for authentication.
        model_id: Model identifier (e.g. ``"phi-4-mini"``).
        input_data: Either a plain string prompt or a list of chat messages
            (``[{"role": "user", "content": "..."}]``).
        parameters: Optional generation parameters (temperature, max_tokens, etc.).
        timeout: HTTP timeout in seconds for the streaming connection.

    Yields:
        :class:`StreamToken` for each SSE ``data:`` event.
    """
    url = f"{server_url.rstrip('/')}/inference/stream"
    payload = _build_payload(model_id, input_data, parameters)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            yield from _parse_sse_lines(response.iter_lines())


async def stream_inference_async(
    server_url: str,
    api_key: str,
    model_id: str,
    input_data: Union[str, list[dict[str, str]]],
    parameters: Optional[dict[str, Any]] = None,
    timeout: float = 120.0,
) -> AsyncIterator[StreamToken]:
    """Stream tokens from the cloud inference endpoint (async).

    Same interface as :func:`stream_inference` but uses ``httpx.AsyncClient``.
    """
    url = f"{server_url.rstrip('/')}/inference/stream"
    payload = _build_payload(model_id, input_data, parameters)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", url, json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                token = _parse_sse_line(line)
                if token is not None:
                    yield token


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _build_payload(
    model_id: str,
    input_data: Union[str, list[dict[str, str]]],
    parameters: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build the JSON request body."""
    payload: dict[str, Any] = {"model_id": model_id}
    if isinstance(input_data, str):
        payload["input_data"] = input_data
    else:
        payload["messages"] = input_data
    if parameters:
        payload["parameters"] = parameters
    return payload


def _parse_sse_lines(lines: Iterator[str]) -> Iterator[StreamToken]:
    """Parse SSE ``data:`` lines from a sync line iterator."""
    for line in lines:
        token = _parse_sse_line(line)
        if token is not None:
            yield token


def _parse_sse_line(line: str) -> Optional[StreamToken]:
    """Parse a single SSE line into a :class:`StreamToken`, or ``None``."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data_str = line[len("data:") :].strip()
    if not data_str:
        return None
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        logger.warning("Failed to parse SSE data: %s", data_str)
        return None
    return StreamToken(
        token=data.get("token", ""),
        done=data.get("done", False),
        provider=data.get("provider"),
        latency_ms=data.get("latency_ms"),
        session_id=data.get("session_id"),
    )
