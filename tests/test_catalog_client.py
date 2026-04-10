"""Tests for the v2 catalog client and infrastructure (catalog_client.py).

Covers:
- CachedData TTL and serialization
- _ServerFetcher fallback behavior, params support, invalidate
- CatalogClientV2 manifest fetching
- _detect_platform
- Disk cache persistence and ETag-based conditional requests
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from octomil.models.catalog_client import (
    CachedData,
    CatalogClientV2,
    _detect_platform,
    _ServerFetcher,
)

# =====================================================================
# CachedData tests
# =====================================================================


class TestCachedData:
    """Tests for the CachedData container."""

    def test_expired_when_never_fetched(self) -> None:
        """Data with fetched_at=0 should be expired."""
        cd = CachedData(data={}, fetched_at=0.0, ttl_seconds=3600)
        assert cd.is_expired is True

    def test_not_expired_within_ttl(self) -> None:
        """Data fetched recently should not be expired."""
        cd = CachedData(data={}, fetched_at=time.time(), ttl_seconds=3600)
        assert cd.is_expired is False

    def test_expired_after_ttl(self) -> None:
        """Data fetched beyond TTL should be expired."""
        cd = CachedData(data={}, fetched_at=time.time() - 7200, ttl_seconds=3600)
        assert cd.is_expired is True

    def test_to_dict_roundtrip(self) -> None:
        """to_dict/from_dict should roundtrip correctly."""
        original = CachedData(
            data={"key": "value"},
            fetched_at=1234567890.0,
            etag='"abc123"',
            ttl_seconds=1800,
        )
        d = original.to_dict()
        restored = CachedData.from_dict(d)
        assert restored.data == original.data
        assert restored.fetched_at == original.fetched_at
        assert restored.etag == original.etag
        assert restored.ttl_seconds == original.ttl_seconds


# =====================================================================
# _ServerFetcher tests
# =====================================================================


class TestServerFetcher:
    """Tests for the generic fetch + cache helper."""

    def test_returns_default_when_no_server_no_cache(self) -> None:
        """When server is unreachable and no disk cache, return default."""
        fetcher = _ServerFetcher(
            endpoint="test/endpoint",
            cache_filename="test_cache.json",
            default_data={"fallback": True},
            api_base="https://api.example.com",
        )

        # Patch httpx to fail
        with patch.dict("sys.modules", {"httpx": None}):
            result = fetcher.get()

        assert result == {"fallback": True}

    def test_disk_cache_persistence(self, tmp_path: Path) -> None:
        """Data should be saved to and loaded from disk cache."""
        with patch("octomil.models.catalog_client._CACHE_DIR", tmp_path):
            fetcher = _ServerFetcher(
                endpoint="test/endpoint",
                cache_filename="test_persist.json",
                default_data={},
                api_base="https://api.example.com",
            )

            # Manually save data
            cached = CachedData(
                data={"persisted": True},
                fetched_at=time.time(),
                etag='"persist-etag"',
                ttl_seconds=3600,
            )
            fetcher._save_to_disk(cached)

            # Verify file exists
            cache_file = tmp_path / "test_persist.json"
            assert cache_file.exists()

            # Load it back
            loaded = fetcher._load_from_disk()
            assert loaded is not None
            assert loaded.data == {"persisted": True}
            assert loaded.etag == '"persist-etag"'

    def test_in_memory_cache_avoids_refetch(self) -> None:
        """Second call should use in-memory cache, not re-fetch."""
        fetcher = _ServerFetcher(
            endpoint="test/endpoint",
            cache_filename="test_inmem.json",
            default_data={"default": True},
            api_base="https://api.example.com",
        )

        # Manually populate in-memory cache
        fetcher._cached = CachedData(
            data={"cached": True},
            fetched_at=time.time(),
            ttl_seconds=3600,
        )

        result = fetcher.get()
        assert result == {"cached": True}

    def test_fetch_with_params_returns_data(self) -> None:
        """fetch() with params should behave like get() for cached data."""
        fetcher = _ServerFetcher(
            endpoint="test/endpoint",
            cache_filename="test_params.json",
            default_data={"default": True},
            api_base="https://api.example.com",
        )

        fetcher._cached = CachedData(
            data={"cached": True},
            fetched_at=time.time(),
            ttl_seconds=3600,
        )

        result = fetcher.fetch(params={"platform": "macos"})
        assert result == {"cached": True}

    def test_invalidate_clears_cache(self) -> None:
        """invalidate() should force re-fetch on next access."""
        fetcher = _ServerFetcher(
            endpoint="test/endpoint",
            cache_filename="test_invalidate.json",
            default_data={"default": True},
            api_base="https://api.example.com",
        )

        fetcher._cached = CachedData(
            data={"cached": True},
            fetched_at=time.time(),
            ttl_seconds=3600,
        )

        fetcher.invalidate()
        assert fetcher._cached is None

        # Next get() should fall through to default (no server, no disk cache)
        with patch.dict("sys.modules", {"httpx": None}):
            result = fetcher.get()
        assert result == {"default": True}


# =====================================================================
# CatalogClientV2 tests
# =====================================================================


class TestCatalogClientV2:
    """Tests for the v2 catalog client."""

    def test_get_manifest_returns_empty_when_no_server(self, tmp_path: Path) -> None:
        """When server unreachable and no cache, should return empty dict."""
        client = CatalogClientV2(base_url="https://api.example.com")

        # Redirect the disk cache to an empty tmp dir so a real
        # ~/.cache/octomil/catalog_manifest_v2.json does not leak into
        # this test and cause a non-empty result.
        with (
            patch("octomil.models.catalog_client._CACHE_DIR", tmp_path),
            patch.dict("sys.modules", {"httpx": None}),
        ):
            manifest = client.get_manifest(platform="all")

        assert manifest == {}

    def test_invalidate_cache(self) -> None:
        """invalidate_cache() should clear the fetcher cache."""
        client = CatalogClientV2(base_url="https://api.example.com")
        # Prime the cache
        with patch.dict("sys.modules", {"httpx": None}):
            client.get_manifest(platform="all")

        client.invalidate_cache()
        assert client._fetcher._cached is None

    def test_base_url_strips_api_v1_suffix(self) -> None:
        """CatalogClientV2 should strip /api/v1 from OCTOMIL_API_BASE."""
        with patch.dict(os.environ, {"OCTOMIL_API_BASE": "https://api.octomil.com/api/v1"}):
            client = CatalogClientV2()
            assert "/api/v1/api/v1" not in client._fetcher.api_base


# =====================================================================
# Platform detection tests
# =====================================================================


class TestDetectPlatform:
    """Tests for _detect_platform."""

    def test_macos(self) -> None:
        with patch("octomil.models.catalog_client.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert _detect_platform() == "macos"

    def test_linux(self) -> None:
        with patch("octomil.models.catalog_client.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert _detect_platform() == "linux"

    def test_windows(self) -> None:
        with patch("octomil.models.catalog_client.sys") as mock_sys:
            mock_sys.platform = "win32"
            assert _detect_platform() == "windows"
