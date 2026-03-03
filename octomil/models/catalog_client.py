"""Server-side model catalog, aliases, engine priority, and registry client.

Fetches model routing intelligence from the Octomil API and caches locally.
Falls back to minimal embedded defaults when the server is unreachable.

Pattern follows ``PolicyClient`` in ``octomil.routing``:
- ETag-based conditional requests for bandwidth efficiency
- Disk cache in ``~/.cache/octomil/`` for offline use
- TTL-based expiration with graceful degradation
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

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
# Generic fetcher — reused by all endpoint-specific clients
# ---------------------------------------------------------------------------


class _ServerFetcher:
    """Generic fetch + cache helper for a single API endpoint."""

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
        """Return cached data, refreshing from server if expired."""
        if self._cached is not None and not self._cached.is_expired:
            return self._cached.data

        # Try loading from disk cache
        if self._cached is None:
            self._cached = self._load_from_disk()

        # If still valid after loading from disk, return it
        if self._cached is not None and not self._cached.is_expired:
            return self._cached.data

        # Try fetching from server
        fetched = self._fetch_from_server()
        if fetched is not None:
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

    def _fetch_from_server(self) -> Optional[CachedData]:
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
# Endpoint-specific clients
# ---------------------------------------------------------------------------


class CatalogClient:
    """Fetches model catalog from ``GET /api/v1/models/catalog``.

    The server returns a dict of model entries keyed by family name.
    Falls back to a minimal catalog with generic model IDs.
    """

    # Minimal fallback — just enough to not crash. No engine-specific
    # variants, no sizes, no quantization info, no proprietary mappings.
    _FALLBACK_CATALOG: dict[str, Any] = {
        "gemma-1b": {
            "publisher": "Google",
            "params": "1B",
            "default_quant": "4bit",
            "variants": {},
            "engines": [],
        },
        "llama-1b": {
            "publisher": "Meta",
            "params": "1B",
            "default_quant": "4bit",
            "variants": {},
            "engines": [],
        },
        "phi-mini": {
            "publisher": "Microsoft",
            "params": "3.8B",
            "default_quant": "4bit",
            "variants": {},
            "engines": [],
        },
    }

    _FALLBACK_ALIASES: dict[str, str] = {}

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
    ) -> None:
        self._catalog_fetcher = _ServerFetcher(
            endpoint="models/catalog",
            cache_filename="model_catalog.json",
            default_data=self._FALLBACK_CATALOG,
            api_base=api_base,
            api_key=api_key,
        )
        self._aliases_fetcher = _ServerFetcher(
            endpoint="models/aliases",
            cache_filename="model_aliases.json",
            default_data=self._FALLBACK_ALIASES,
            api_base=api_base,
            api_key=api_key,
        )

    def get_catalog(self) -> dict[str, Any]:
        """Return the model catalog dict (server-fetched or fallback)."""
        return self._catalog_fetcher.get()

    def get_aliases(self) -> dict[str, str]:
        """Return the model alias mapping (server-fetched or fallback)."""
        return self._aliases_fetcher.get()


class EnginePriorityClient:
    """Fetches engine priority from ``GET /api/v1/models/engine-priority``.

    Falls back to a single generic ``["auto"]`` priority.
    """

    _FALLBACK_PRIORITY: list[str] = ["auto"]

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
    ) -> None:
        self._fetcher = _ServerFetcher(
            endpoint="models/engine-priority",
            cache_filename="engine_priority.json",
            default_data=self._FALLBACK_PRIORITY,
            api_base=api_base,
            api_key=api_key,
        )

    def get_priority(self) -> list[str]:
        """Return engine priority list (server-fetched or fallback)."""
        return self._fetcher.get()


class ModelFamiliesClient:
    """Fetches model families from ``GET /api/v1/models/families``.

    Falls back to an empty dict.
    """

    _FALLBACK_FAMILIES: dict[str, Any] = {}

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
    ) -> None:
        self._fetcher = _ServerFetcher(
            endpoint="models/families",
            cache_filename="model_families.json",
            default_data=self._FALLBACK_FAMILIES,
            api_base=api_base,
            api_key=api_key,
        )

    def get_families(self) -> dict[str, Any]:
        """Return model families dict (server-fetched or fallback)."""
        return self._fetcher.get()


class SourceAliasesClient:
    """Fetches source-level model aliases from ``GET /api/v1/models/aliases``.

    These are the ``hf``/``ollama``/``kaggle`` source mappings used by
    ``octomil.sources.resolver``. Falls back to an empty dict.
    """

    _FALLBACK_ALIASES: dict[str, dict[str, str]] = {}

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
    ) -> None:
        self._fetcher = _ServerFetcher(
            endpoint="models/source-aliases",
            cache_filename="source_aliases.json",
            default_data=self._FALLBACK_ALIASES,
            api_base=api_base,
            api_key=api_key,
        )

    def get_aliases(self) -> dict[str, dict[str, str]]:
        """Return source-level alias mapping (server-fetched or fallback)."""
        return self._fetcher.get()


class SdkConfigClient:
    """Consolidated client — fetches all model routing data in a single call.

    Calls ``GET /api/v1/models/sdk-config`` which returns catalog, aliases,
    engine priority, families, and source aliases in one response.  This
    replaces 5 separate HTTP round-trips during SDK startup.

    The public API mirrors the individual clients for backward compat.
    """

    _FALLBACK_CONFIG: dict[str, Any] = {
        "catalog": CatalogClient._FALLBACK_CATALOG,
        "aliases": CatalogClient._FALLBACK_ALIASES,
        "engine_priority": EnginePriorityClient._FALLBACK_PRIORITY,
        "families": ModelFamiliesClient._FALLBACK_FAMILIES,
        "source_aliases": SourceAliasesClient._FALLBACK_ALIASES,
    }

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
    ) -> None:
        self._fetcher = _ServerFetcher(
            endpoint="models/sdk-config",
            cache_filename="sdk_config.json",
            default_data=self._FALLBACK_CONFIG,
            api_base=api_base,
            api_key=api_key,
        )

    def _data(self) -> dict[str, Any]:
        return self._fetcher.get()

    def get_catalog(self) -> dict[str, Any]:
        """Return the model catalog dict."""
        return self._data().get("catalog", {})

    def get_aliases(self) -> dict[str, str]:
        """Return the model alias mapping."""
        return self._data().get("aliases", {})

    def get_priority(self) -> list[str]:
        """Return engine priority list."""
        return self._data().get("engine_priority", ["auto"])

    def get_families(self) -> dict[str, Any]:
        """Return model families dict."""
        return self._data().get("families", {})

    def get_source_aliases(self) -> dict[str, dict[str, str]]:
        """Return source-level alias mapping."""
        return self._data().get("source_aliases", {})
