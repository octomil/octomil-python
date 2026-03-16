"""Tests for PublishableKeyAuth in octomil.auth."""

from __future__ import annotations

import pytest

from octomil._generated.auth_type import AuthType
from octomil._generated.scope import Scope
from octomil.auth import PUBLISHABLE_KEY_SCOPES, PublishableKeyAuth
from octomil.errors import OctomilError, OctomilErrorCode


class TestPublishableKeyAuth:
    def test_valid_key_accepted(self) -> None:
        auth = PublishableKeyAuth(api_key="oct_pub_live_abc123")
        assert auth.api_key == "oct_pub_live_abc123"
        assert auth.auth_type == AuthType.PUBLISHABLE_KEY

    def test_invalid_prefix_rejected(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            PublishableKeyAuth(api_key="edg_not_a_pub_key")
        assert exc_info.value.code == OctomilErrorCode.INVALID_API_KEY
        assert "oct_pub_" in str(exc_info.value)

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(OctomilError):
            PublishableKeyAuth(api_key="")

    def test_scopes_are_restricted(self) -> None:
        auth = PublishableKeyAuth(api_key="oct_pub_test_key")
        expected = frozenset(
            {
                Scope.DEVICES_REGISTER,
                Scope.DEVICES_HEARTBEAT,
                Scope.TELEMETRY_WRITE,
                Scope.MODELS_READ,
            }
        )
        assert auth.scopes == expected

    def test_scopes_exclude_write_operations(self) -> None:
        auth = PublishableKeyAuth(api_key="oct_pub_test_key")
        assert Scope.MODELS_WRITE not in auth.scopes
        assert Scope.ROLLOUTS_WRITE not in auth.scopes
        assert Scope.BENCHMARKS_WRITE not in auth.scopes

    def test_headers(self) -> None:
        auth = PublishableKeyAuth(api_key="oct_pub_mykey")
        headers = auth.headers()
        assert headers["Authorization"] == "Bearer oct_pub_mykey"
        assert headers["X-Octomil-Auth-Type"] == "publishable_key"

    def test_frozen(self) -> None:
        auth = PublishableKeyAuth(api_key="oct_pub_frozen")
        with pytest.raises(AttributeError):
            auth.api_key = "changed"  # type: ignore[misc]

    def test_custom_api_base(self) -> None:
        auth = PublishableKeyAuth(api_key="oct_pub_custom", api_base="https://custom.api/v1")
        assert auth.api_base == "https://custom.api/v1"

    def test_module_level_scopes_constant(self) -> None:
        assert PUBLISHABLE_KEY_SCOPES == frozenset(
            {
                Scope.DEVICES_REGISTER,
                Scope.DEVICES_HEARTBEAT,
                Scope.TELEMETRY_WRITE,
                Scope.MODELS_READ,
            }
        )
