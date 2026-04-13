"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class EmailTemplate(str, Enum):
    PASSWORD_RESET = "password_reset"
    """Password reset link email."""
    EMAIL_VERIFICATION = "email_verification"
    """Email address verification for new accounts."""
    WELCOME = "welcome"
    """Welcome email sent after account creation."""
    STATUS_SUBSCRIPTION_VERIFICATION = "status_subscription_verification"
    """Verify subscription to status page notifications."""
    INCIDENT_NOTIFICATION = "incident_notification"
    """Active incident alert sent to status subscribers."""
    INCIDENT_RESOLUTION = "incident_resolution"
    """Incident resolved notification sent to status subscribers."""
    TEAM_INVITATION = "team_invitation"
    """Invite a user to join an organization."""
    API_KEY_EXPIRY_WARNING = "api_key_expiry_warning"
    """Warning that an API key is approaching expiration."""
