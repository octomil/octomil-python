from __future__ import annotations

from typing import Any, Callable, Optional

import httpx


class EdgeMLClientError(RuntimeError):
    pass


class _ApiClient:
    def __init__(
        self,
        auth_token_provider: Callable[[], str],
        api_base: str,
        timeout: float = 20.0,
    ):
        self.auth_token_provider = auth_token_provider
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        token = self.auth_token_provider()
        if not token:
            raise EdgeMLClientError("auth_token_provider returned an empty token")
        return {"Authorization": f"Bearer {token}"}

    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(f"{self.api_base}{path}", params=params, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json()

    def post(self, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(f"{self.api_base}{path}", json=payload or {}, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json()

    def put(self, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.put(f"{self.api_base}{path}", json=payload or {}, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json() if res.text else {}

    def patch(self, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.patch(f"{self.api_base}{path}", json=payload or {}, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json() if res.text else {}

    def delete(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.delete(f"{self.api_base}{path}", params=params, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json() if res.text else {}

    def get_bytes(self, path: str, params: Optional[dict[str, Any]] = None) -> bytes:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(f"{self.api_base}{path}", params=params, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.content

    def report_inference_event(self, payload: dict[str, Any]) -> Any:
        """Report a streaming inference event to ``POST /inference/events``."""
        return self.post("/inference/events", payload)

