"""
Enterprise API client for EdgeML org/team management.

Wraps the onboarding, org, settings, and API key endpoints behind
a clean interface suitable for CLI and script usage::

    from edgeml.enterprise import EnterpriseClient

    client = EnterpriseClient(api_key="edg_...")
    client.create_org("Acme Corp", region="us")
    client.set_compliance("hipaa")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_API_BASE = "https://api.edgeml.io/api/v1"

# ---------------------------------------------------------------------------
# Compliance presets — map names to OrgSettings fields
# ---------------------------------------------------------------------------

COMPLIANCE_PRESETS: dict[str, dict[str, Any]] = {
    "hipaa": {
        "hipaa_mode": True,
        "audit_retention_days": 2190,  # 6 years
        "require_mfa_for_admin": True,
        "mfa_required": True,
        "require_admin_approval": True,
        "session_duration_hours": 8,
        "reauth_interval_minutes": 30,
        "policy_profile": "regulated",
    },
    "gdpr": {
        "audit_retention_days": 1825,  # 5 years
        "require_mfa_for_admin": True,
        "policy_profile": "balanced",
    },
    "pci": {
        "require_mfa_for_admin": True,
        "mfa_required": True,
        "session_duration_hours": 8,
        "reauth_interval_minutes": 30,
        "require_admin_approval": True,
        "policy_profile": "regulated",
    },
    "soc2": {
        "audit_retention_days": 365,
        "require_mfa_for_admin": True,
        "require_admin_approval": True,
        "require_model_approval": True,
        "auto_rollback_enabled": True,
        "policy_profile": "balanced",
    },
}


# ---------------------------------------------------------------------------
# Config file helpers (~/.edgeml/config.json)
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    return Path.home() / ".edgeml" / "config.json"


def load_config() -> dict[str, Any]:
    """Load the local EdgeML config from ``~/.edgeml/config.json``."""
    path = _config_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_config(config: dict[str, Any]) -> None:
    """Persist config to ``~/.edgeml/config.json``."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    # Restrict permissions — config may contain org metadata
    path.chmod(0o600)


def get_org_id() -> str:
    """Read org_id from config or EDGEML_ORG_ID env var."""
    env_id = os.environ.get("EDGEML_ORG_ID", "")
    if env_id:
        return env_id
    cfg = load_config()
    return cfg.get("org_id", "")


# ---------------------------------------------------------------------------
# Enterprise API client
# ---------------------------------------------------------------------------


class EnterpriseClientError(RuntimeError):
    """Raised when an enterprise API call fails."""


class EnterpriseClient:
    """Client for enterprise org/team management.

    Talks to the fed-learning server's onboarding, org management,
    settings, and API key endpoints.

    Args:
        api_key: Bearer token for authentication.
        api_base: Base URL for the EdgeML API (default: ``https://api.edgeml.io/api/v1``).
    """

    def __init__(
        self,
        api_key: str,
        api_base: str | None = None,
    ) -> None:
        base = (
            api_base
            or os.environ.get("EDGEML_API_URL")
            or os.environ.get("EDGEML_API_BASE")
            or _DEFAULT_API_BASE
        )
        self._http = httpx.Client(
            base_url=base.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20.0,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    # -- helpers ----------------------------------------------------------

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise EnterpriseClientError(
                f"API error {resp.status_code}: {detail}"
            )

    # -- Org creation (onboarding) ----------------------------------------

    def create_org(
        self,
        name: str,
        org_id: str | None = None,
        region: str = "us",
        workspace_type: str = "enterprise",
    ) -> dict[str, Any]:
        """Create an organization via the onboarding workspace endpoint.

        If *org_id* is ``None`` the server auto-generates one.

        Returns:
            Onboarding status response dict.
        """
        # The onboarding workspace endpoint is PUT /onboarding/{org_id}/workspace
        # We need an org_id — if not provided, use the name as a slug
        effective_org_id = org_id or name.lower().replace(" ", "-")[:36]

        payload = {
            "workspace_type": workspace_type,
            "workspace_name": name,
            "region": region,
        }
        resp = self._http.put(
            f"/onboarding/{effective_org_id}/workspace",
            json=payload,
        )
        self._raise_for_status(resp)
        return resp.json()

    # -- Policy profile ---------------------------------------------------

    def set_policy_profile(
        self, org_id: str, profile: str
    ) -> dict[str, Any]:
        """Apply a policy profile (startup, balanced, regulated).

        Returns:
            Onboarding status response dict.
        """
        resp = self._http.put(
            f"/onboarding/{org_id}/policy",
            json={"policy_profile": profile},
        )
        self._raise_for_status(resp)
        return resp.json()

    # -- Compliance presets -----------------------------------------------

    def set_compliance(
        self, org_id: str, preset: str
    ) -> dict[str, Any]:
        """Apply a named compliance preset to org settings.

        Valid presets: hipaa, gdpr, pci, soc2.

        This first applies the underlying policy profile, then
        pushes the detailed settings via the settings endpoint.

        Returns:
            Updated settings response dict.
        """
        if preset not in COMPLIANCE_PRESETS:
            raise EnterpriseClientError(
                f"Unknown compliance preset '{preset}'. "
                f"Valid options: {', '.join(sorted(COMPLIANCE_PRESETS))}"
            )

        settings = dict(COMPLIANCE_PRESETS[preset])
        policy_profile = settings.pop("policy_profile", None)

        # Apply the policy profile first
        if policy_profile:
            self.set_policy_profile(org_id, policy_profile)

        # Push the detailed settings overrides
        return self.update_settings(org_id, **settings)

    # -- Team / members ---------------------------------------------------

    def invite_member(
        self,
        org_id: str,
        email: str,
        role: str = "member",
        name: str | None = None,
    ) -> dict[str, Any]:
        """Invite a team member to the organization.

        Returns:
            OrgMemberResponse dict with user_id, email, name, role.
        """
        payload: dict[str, Any] = {"email": email, "role": role}
        if name:
            payload["name"] = name
        resp = self._http.post(
            f"/orgs/{org_id}/members",
            json=payload,
        )
        self._raise_for_status(resp)
        return resp.json()

    def list_members(self, org_id: str) -> list[dict[str, Any]]:
        """List all members of an organization.

        Returns:
            List of OrgMemberResponse dicts.
        """
        resp = self._http.get(f"/orgs/{org_id}/members")
        self._raise_for_status(resp)
        return resp.json()

    # -- API keys ---------------------------------------------------------

    def create_api_key(
        self,
        org_id: str,
        name: str,
        scopes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new API key for the organization.

        Returns:
            CreateApiKeyResponse dict (includes ``api_key`` field with raw key).
        """
        payload: dict[str, Any] = {
            "org_id": org_id,
            "name": name,
            "scopes": scopes or {},
        }
        resp = self._http.post("/api-keys", json=payload)
        self._raise_for_status(resp)
        return resp.json()

    def list_api_keys(self, org_id: str) -> list[dict[str, Any]]:
        """List all API keys for the organization.

        Returns:
            List of ApiKeyResponse dicts.
        """
        resp = self._http.get("/api-keys", params={"org_id": org_id})
        self._raise_for_status(resp)
        return resp.json()

    def revoke_api_key(self, key_id: str) -> dict[str, Any]:
        """Revoke an API key.

        Returns:
            ApiKeyResponse dict with ``revoked_at`` populated.
        """
        resp = self._http.post(f"/api-keys/{key_id}/revoke")
        self._raise_for_status(resp)
        return resp.json()

    # -- Settings ---------------------------------------------------------

    def get_settings(self, org_id: str) -> dict[str, Any]:
        """Get organization settings.

        Returns:
            OrgSettingsResponse dict.
        """
        resp = self._http.get(f"/settings/{org_id}")
        self._raise_for_status(resp)
        return resp.json()

    def update_settings(
        self, org_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Update organization settings.

        Keyword args are forwarded as the JSON body.

        Returns:
            Updated OrgSettingsResponse dict.
        """
        resp = self._http.put(f"/settings/{org_id}", json=kwargs)
        self._raise_for_status(resp)
        return resp.json()

    # -- Onboarding status ------------------------------------------------

    def get_onboarding_status(self, org_id: str) -> dict[str, Any]:
        """Get the onboarding status for an organization.

        Returns:
            OnboardingStatusResponse dict.
        """
        resp = self._http.get(f"/onboarding/{org_id}")
        self._raise_for_status(resp)
        return resp.json()

    def complete_onboarding(self, org_id: str) -> dict[str, Any]:
        """Mark onboarding as complete.

        Returns:
            OnboardingStatusResponse dict.
        """
        resp = self._http.post(
            f"/onboarding/{org_id}/complete",
            json={"sdk_bootstrap_confirmed": True},
        )
        self._raise_for_status(resp)
        return resp.json()
