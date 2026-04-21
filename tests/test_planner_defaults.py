"""Tests for planner routing defaults."""

from __future__ import annotations

import os
from unittest.mock import patch

from octomil.planner_defaults import (
    default_routing_policy,
    is_cloud_blocked,
    resolve_planner_enabled,
)

# ===========================================================================
# Fixtures
# ===========================================================================


class FakeOrgApiKeyAuth:
    """Stub for OrgApiKeyAuth."""

    def __init__(self, api_key: str = "edg_test_123", org_id: str = "org_abc") -> None:
        self.api_key = api_key
        self.org_id = org_id


class FakePublishableKeyAuth:
    """Stub for PublishableKeyAuth."""

    def __init__(self, api_key: str = "oct_pub_test_abc123") -> None:
        self.api_key = api_key


class FakeNoCredAuth:
    """Stub for auth with no credentials."""

    pass


# ===========================================================================
# resolve_planner_enabled
# ===========================================================================


class TestResolvePlannerEnabled:
    def test_default_on_with_org_api_key(self):
        result = resolve_planner_enabled(auth=FakeOrgApiKeyAuth())
        assert result is True

    def test_default_on_with_publishable_key(self):
        result = resolve_planner_enabled(auth=FakePublishableKeyAuth())
        assert result is True

    def test_default_off_with_no_auth(self):
        result = resolve_planner_enabled(auth=None)
        assert result is False

    def test_default_off_with_empty_api_key(self):
        result = resolve_planner_enabled(auth=FakeOrgApiKeyAuth(api_key=""))
        assert result is False

    def test_default_off_with_no_credentials(self):
        result = resolve_planner_enabled(auth=FakeNoCredAuth())
        assert result is False

    def test_explicit_false_overrides_credentials(self):
        result = resolve_planner_enabled(explicit_override=False, auth=FakeOrgApiKeyAuth())
        assert result is False

    def test_explicit_true_overrides_no_credentials(self):
        result = resolve_planner_enabled(explicit_override=True, auth=None)
        assert result is True

    def test_env_var_disables_planner(self):
        with patch.dict(os.environ, {"OCTOMIL_DISABLE_PLANNER": "1"}):
            result = resolve_planner_enabled(auth=FakeOrgApiKeyAuth())
            assert result is False

    def test_env_var_true_disables_planner(self):
        with patch.dict(os.environ, {"OCTOMIL_DISABLE_PLANNER": "true"}):
            result = resolve_planner_enabled(auth=FakeOrgApiKeyAuth())
            assert result is False

    def test_env_var_overrides_explicit_true(self):
        with patch.dict(os.environ, {"OCTOMIL_DISABLE_PLANNER": "1"}):
            result = resolve_planner_enabled(explicit_override=True, auth=FakeOrgApiKeyAuth())
            assert result is False

    def test_env_var_zero_does_not_disable(self):
        with patch.dict(os.environ, {"OCTOMIL_DISABLE_PLANNER": "0"}):
            result = resolve_planner_enabled(auth=FakeOrgApiKeyAuth())
            assert result is True


# ===========================================================================
# is_cloud_blocked
# ===========================================================================


class TestIsCloudBlocked:
    def test_private_blocks_cloud(self):
        assert is_cloud_blocked("private") is True

    def test_local_only_blocks_cloud(self):
        assert is_cloud_blocked("local_only") is True

    def test_cloud_first_does_not_block(self):
        assert is_cloud_blocked("cloud_first") is False

    def test_local_first_does_not_block(self):
        assert is_cloud_blocked("local_first") is False

    def test_auto_does_not_block(self):
        assert is_cloud_blocked("auto") is False

    def test_none_does_not_block(self):
        assert is_cloud_blocked(None) is False


# ===========================================================================
# default_routing_policy
# ===========================================================================


class TestDefaultRoutingPolicy:
    def test_auto_when_planner_enabled(self):
        assert default_routing_policy(planner_enabled=True) == "auto"

    def test_local_first_when_planner_disabled(self):
        assert default_routing_policy(planner_enabled=False) == "local_first"
