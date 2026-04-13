"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class OauthProvider(str, Enum):
    GOOGLE = "google"
    """Google OAuth 2.0 (OpenID Connect). Scopes — openid, email, profile."""
    APPLE = "apple"
    """Apple Sign In (OpenID Connect). Scopes — openid, email, name."""
    GITHUB = "github"
    """GitHub OAuth 2.0. Scopes — user:email, read:user."""
    MICROSOFT = "microsoft"
    """Microsoft Entra ID (Azure AD) OAuth."""
    OKTA = "okta"
    """Okta workforce identity OAuth/OIDC."""
