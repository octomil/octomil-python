"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class AuthMethod(str, Enum):
    PASSWORD = "password"
    """Email and password login."""
    PASSKEY = "passkey"
    """WebAuthn/FIDO2 passkey authentication."""
    OAUTH_GOOGLE = "oauth_google"
    """Google OAuth 2.0 (OpenID Connect). Fetches email, name, and profile picture."""
    OAUTH_APPLE = "oauth_apple"
    """Apple Sign In (OpenID Connect). Fetches email and name (name only on first auth)."""
    OAUTH_GITHUB = "oauth_github"
    """GitHub OAuth 2.0. Fetches email (from /user/emails), name, and avatar (from /user)."""
    SSO_SAML = "sso_saml"
    """Enterprise SSO via SAML/SCIM identity federation."""
    DEV_LOGIN = "dev_login"
    """Development-only bypass login. Disabled in production."""
