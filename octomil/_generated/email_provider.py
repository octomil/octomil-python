"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class EmailProvider(str, Enum):
    RESEND = "resend"
    """Resend (resend.com). Current default provider."""
    SENDGRID = "sendgrid"
    """Twilio SendGrid."""
    SES = "ses"
    """Amazon Simple Email Service (SES)."""
    POSTMARK = "postmark"
    """Postmark by ActiveCampaign."""
    SMTP = "smtp"
    """Generic SMTP relay."""
