from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger(__name__)


class OctomilClientError(RuntimeError):
    pass


# Status codes that are safe to retry (server-side transient errors).
_RETRYABLE_STATUS_CODES = {502, 503, 504, 429}


class _ApiClient:
    def __init__(
        self,
        auth_token_provider: Callable[[], str],
        api_base: str,
        timeout: float = 20.0,
        download_timeout: float = 120.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ):
        self.auth_token_provider = auth_token_provider
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.download_timeout = download_timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client: Optional[httpx.Client] = None

    def _get_client(self, timeout: Optional[float] = None) -> httpx.Client:
        """Return a shared httpx.Client, creating one if needed."""
        effective_timeout = timeout or self.timeout
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=effective_timeout)
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client is not None and not self._client.is_closed:
            self._client.close()
            self._client = None

    def _headers(self) -> dict[str, str]:
        token = self.auth_token_provider()
        if not token:
            raise OctomilClientError("auth_token_provider returned an empty token")
        return {"Authorization": f"Bearer {token}"}

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Execute an HTTP request with exponential backoff retry.

        Retries on connection errors and retryable HTTP status codes
        (502, 503, 504, 429).  Non-retryable errors (4xx except 429)
        are raised immediately.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                client = self._get_client(timeout)
                res = client.request(method, url, **kwargs)

                if res.status_code < 400:
                    return res

                # Non-retryable client error -- fail immediately.
                if res.status_code < 500 and res.status_code not in _RETRYABLE_STATUS_CODES:
                    raise OctomilClientError(res.text)

                # Retryable server error -- backoff and retry.
                if res.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries - 1:
                    wait = self.backoff_base * (2**attempt)
                    logger.warning(
                        "Retryable HTTP %d on %s %s (attempt %d/%d, waiting %.1fs)",
                        res.status_code,
                        method,
                        url,
                        attempt + 1,
                        self.max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                # Final attempt or non-retryable 5xx -- raise.
                raise OctomilClientError(res.text)

            except OctomilClientError:
                raise
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    wait = self.backoff_base * (2**attempt)
                    logger.warning(
                        "Connection error on %s %s: %s (attempt %d/%d, waiting %.1fs)",
                        method,
                        url,
                        exc,
                        attempt + 1,
                        self.max_retries,
                        wait,
                    )
                    # Reset the client on connection errors so the next attempt
                    # starts with a fresh socket.
                    self.close()
                    time.sleep(wait)
                else:
                    raise OctomilClientError(
                        f"Request failed after {self.max_retries} attempts: {exc}"
                    ) from exc

        # Should not reach here, but just in case.
        raise OctomilClientError(
            f"Request failed after {self.max_retries} attempts"
            + (f": {last_exc}" if last_exc else "")
        )

    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        res = self._request_with_retry(
            "GET",
            f"{self.api_base}{path}",
            params=params,
            headers=self._headers(),
        )
        return res.json()

    def post(self, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        res = self._request_with_retry(
            "POST",
            f"{self.api_base}{path}",
            json=payload or {},
            headers=self._headers(),
        )
        return res.json()

    def put(self, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        res = self._request_with_retry(
            "PUT",
            f"{self.api_base}{path}",
            json=payload or {},
            headers=self._headers(),
        )
        return res.json() if res.text else {}

    def patch(self, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        res = self._request_with_retry(
            "PATCH",
            f"{self.api_base}{path}",
            json=payload or {},
            headers=self._headers(),
        )
        return res.json() if res.text else {}

    def delete(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        res = self._request_with_retry(
            "DELETE",
            f"{self.api_base}{path}",
            params=params,
            headers=self._headers(),
        )
        return res.json() if res.text else {}

    def get_bytes(self, path: str, params: Optional[dict[str, Any]] = None) -> bytes:
        res = self._request_with_retry(
            "GET",
            f"{self.api_base}{path}",
            timeout=self.download_timeout,
            params=params,
            headers=self._headers(),
        )
        return res.content

    def post_bytes(self, path: str, data: bytes) -> Any:
        """POST raw bytes (``application/octet-stream``)."""
        headers = {**self._headers(), "Content-Type": "application/octet-stream"}
        res = self._request_with_retry(
            "POST",
            f"{self.api_base}{path}",
            content=data,
            headers=headers,
        )
        return res.json()

    def report_inference_event(self, payload: dict[str, Any]) -> Any:
        """Report a streaming inference event to ``POST /inference/events``."""
        return self.post("/inference/events", payload)

    # ------------------------------------------------------------------
    # SecAgg endpoints
    # ------------------------------------------------------------------

    def secagg_get_session(self, round_id: str, device_id: str) -> Any:
        """Fetch the SecAgg session config for a round."""
        return self.get(
            f"/training/rounds/{round_id}/secagg/session",
            params={"device_id": device_id},
        )

    def secagg_submit_shares(self, round_id: str, device_id: str, shares_data: bytes) -> Any:
        """Upload this client's Shamir key-shares for the round."""
        return self.post_bytes(
            f"/training/rounds/{round_id}/secagg/shares?device_id={device_id}",
            shares_data,
        )

    def secagg_submit_masked_update(self, round_id: str, device_id: str, masked_data: bytes) -> Any:
        """Upload the masked model update."""
        return self.post_bytes(
            f"/training/rounds/{round_id}/secagg/masked?device_id={device_id}",
            masked_data,
        )

    def secagg_submit_unmask_share(
        self, round_id: str, device_id: str, peer_id: str, share_data: bytes
    ) -> Any:
        """Reveal a Shamir share for a dropped-out peer."""
        return self.post_bytes(
            f"/training/rounds/{round_id}/secagg/unmask"
            f"?device_id={device_id}&peer_id={peer_id}",
            share_data,
        )
