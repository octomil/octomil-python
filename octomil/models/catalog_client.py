"""Unified v2 catalog client — fetches the manifest from a single endpoint.

Provides ``CatalogClientV2`` which fetches the complete manifest from
``GET /api/v2/catalog/manifest``.

Infrastructure classes (``CachedData``, ``_ServerFetcher``) are preserved
and enhanced: ``_ServerFetcher`` now supports query params via ``fetch()``.

Pattern follows ``PolicyClient`` in ``octomil.routing``:
- ETag-based conditional requests for bandwidth efficiency
- Disk cache in ``~/.cache/octomil/`` for offline use
- TTL-based expiration with graceful degradation
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ._embedded_catalog import EMBEDDED_MANIFEST

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.environ.get("OCTOMIL_CACHE_DIR", Path.home() / ".cache" / "octomil"))


# ---------------------------------------------------------------------------
# Cached data wrapper
# ---------------------------------------------------------------------------


class CachedData:
    """Container for server-fetched data with TTL and ETag tracking."""

    def __init__(
        self,
        data: Any,
        *,
        fetched_at: float = 0.0,
        etag: str = "",
        ttl_seconds: int = 3600,
    ) -> None:
        self.data = data
        self.fetched_at = fetched_at
        self.etag = etag
        self.ttl_seconds = ttl_seconds

    @property
    def is_expired(self) -> bool:
        if self.fetched_at == 0.0:
            return True
        return (time.time() - self.fetched_at) > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "fetched_at": self.fetched_at,
            "etag": self.etag,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CachedData:
        return cls(
            data=d.get("data"),
            fetched_at=d.get("fetched_at", 0.0),
            etag=d.get("etag", ""),
            ttl_seconds=d.get("ttl_seconds", 3600),
        )


# ---------------------------------------------------------------------------
# Generic fetcher — reused by CatalogClientV2 and device_config
# ---------------------------------------------------------------------------


class _ServerFetcher:
    """Generic fetch + cache helper for a single API endpoint.

    Supports two calling conventions:
    - ``get()`` — no params, used by existing consumers (device_config, etc.)
    - ``fetch(params=...)`` — with optional query params for the v2 manifest
    """

    def __init__(
        self,
        endpoint: str,
        cache_filename: str,
        default_data: Any,
        *,
        api_base: str = "",
        api_key: str = "",
    ) -> None:
        self._endpoint = endpoint
        self._cache_filename = cache_filename
        self._default_data = default_data
        self._api_base = api_base.rstrip("/") if api_base else ""
        self._api_key = api_key
        self._cached: Optional[CachedData] = None

    @property
    def api_base(self) -> str:
        if self._api_base:
            return self._api_base
        return os.environ.get("OCTOMIL_API_BASE", "https://api.octomil.com/api/v1")

    def get(self) -> Any:
        """Return cached data, refreshing from server if expired (no params)."""
        return self.fetch()

    def fetch(self, params: Optional[dict[str, str]] = None) -> Any:
        """Return cached data, refreshing from server if expired.

        Parameters
        ----------
        params:
            Optional query parameters appended to the request URL.
            Note: caching is keyed by endpoint only (not params), so
            callers should use a consistent param set per fetcher instance.
        """
        if self._cached is not None and not self._cached.is_expired:
            return self._cached.data

        # Try loading from disk cache
        if self._cached is None:
            self._cached = self._load_from_disk()

        # If still valid after loading from disk, return it
        if self._cached is not None and not self._cached.is_expired:
            return self._cached.data

        # Try fetching from server
        fetched = self._fetch_from_server(params=params)
        if fetched is not None and fetched.data:
            self._cached = fetched
            self._save_to_disk(fetched)
            return self._cached.data

        # Use disk cache even if expired
        if self._cached is not None:
            logger.debug(
                "Using expired cached %s (server unreachable)",
                self._cache_filename,
            )
            return self._cached.data

        # Fall back to minimal embedded default
        logger.debug(
            "Using default embedded %s (no cache, server unreachable)",
            self._cache_filename,
        )
        self._cached = CachedData(
            data=self._default_data,
            fetched_at=0.0,
            ttl_seconds=0,
        )
        return self._cached.data

    def invalidate(self) -> None:
        """Force re-fetch on next access by clearing in-memory cache."""
        self._cached = None

    def _fetch_from_server(self, params: Optional[dict[str, str]] = None) -> Optional[CachedData]:
        """Fetch from the API endpoint with ETag-based conditional requests."""
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not available — cannot fetch %s", self._endpoint)
            return None

        headers: dict[str, str] = {}
        api_key = self._api_key or os.environ.get("OCTOMIL_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if self._cached and self._cached.etag:
            headers["If-None-Match"] = self._cached.etag

        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(
                    f"{self.api_base}/{self._endpoint.lstrip('/')}",
                    headers=headers,
                    params=params,
                )

            if resp.status_code == 304:
                # Not modified — refresh TTL on existing cache
                if self._cached:
                    self._cached.fetched_at = time.time()
                    self._save_to_disk(self._cached)
                return self._cached

            if resp.status_code != 200:
                logger.debug("Fetch %s returned HTTP %d", self._endpoint, resp.status_code)
                return None

            data = resp.json()

            # Extract TTL from Cache-Control header if present
            cc = resp.headers.get("cache-control", "")
            ttl = 3600
            if "max-age=" in cc:
                try:
                    ttl = int(cc.split("max-age=")[1].split(",")[0].strip())
                except (ValueError, IndexError):
                    pass

            etag = resp.headers.get("etag", "")

            return CachedData(
                data=data,
                fetched_at=time.time(),
                etag=etag,
                ttl_seconds=ttl,
            )

        except Exception:
            logger.debug("Failed to fetch %s from server", self._endpoint, exc_info=True)
            return None

    def _load_from_disk(self) -> Optional[CachedData]:
        """Load cached data from disk."""
        path = _CACHE_DIR / self._cache_filename
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return CachedData.from_dict(raw)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def _save_to_disk(self, cached: CachedData) -> None:
        """Persist data to disk cache."""
        path = _CACHE_DIR / self._cache_filename
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(cached.to_dict(), indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.debug("Failed to write cache to %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Platform auto-detection
# ---------------------------------------------------------------------------


def _detect_platform() -> str:
    """Detect the current platform for manifest filtering.

    Returns one of: ``macos``, ``linux``, ``windows``, ``ios``, ``android``.
    """
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "linux":
        return "linux"
    if sys.platform == "win32":
        return "windows"
    return sys.platform


# ---------------------------------------------------------------------------
# V2 catalog client — single endpoint replaces 5 v1 clients
# ---------------------------------------------------------------------------


class CatalogClientV2:
    """Client for the v2 unified catalog manifest endpoint.

    Fetches the complete model manifest from ``GET /api/v2/catalog/manifest``.
    The manifest contains all models, packages, engine mappings, and aliases
    in a single response, replacing the 5 separate v1 endpoints.

    Falls back to :data:`EMBEDDED_MANIFEST` when the server is unreachable
    and no disk cache exists.
    """

    _MANIFEST_ENDPOINT = "/api/v2/catalog/manifest"
    _CACHE_FILENAME = "catalog_manifest_v2.json"

    def __init__(self, base_url: str | None = None) -> None:
        api_base = base_url.rstrip("/") if base_url else ""
        # Override the default api_base to use v2 path prefix.
        # _ServerFetcher.api_base defaults to .../api/v1 but our endpoint
        # includes the full path already, so we set the base to the root.
        if api_base:
            fetcher_base = api_base
        else:
            fetcher_base = os.environ.get("OCTOMIL_API_BASE", "https://api.octomil.com").rstrip("/")
            # Strip trailing /api/v1 if present — the endpoint has its own path.
            if fetcher_base.endswith("/api/v1"):
                fetcher_base = fetcher_base[: -len("/api/v1")]

        self._fetcher = _ServerFetcher(
            endpoint=self._MANIFEST_ENDPOINT,
            cache_filename=self._CACHE_FILENAME,
            default_data=EMBEDDED_MANIFEST,
            api_base=fetcher_base,
        )

    def get_manifest(self, platform: str | None = None) -> dict:
        """Get the full catalog manifest, optionally filtered by platform.

        Parameters
        ----------
        platform:
            Filter packages to a specific platform (e.g. ``"macos"``).
            If ``None``, auto-detects from the current system.
            Pass ``"all"`` to skip filtering.
        """
        resolved_platform = platform if platform else _detect_platform()
        params: dict[str, str] | None = None
        if resolved_platform and resolved_platform != "all":
            params = {"platform": resolved_platform}
        return self._fetcher.fetch(params=params)

    def get_models(self, platform: str | None = None) -> list[dict]:
        """Get a flat models list extracted from the nested manifest.

        Returns a list of dicts with ``id``, ``family``, ``parameter_count``,
        ``packages`` etc. — derived from the canonical nested format.
        """
        from .catalog import _iter_manifest_models

        manifest = self.get_manifest(platform=platform)
        return _iter_manifest_models(manifest)

    def invalidate_cache(self) -> None:
        """Force re-fetch on next access."""
        self._fetcher.invalidate()
