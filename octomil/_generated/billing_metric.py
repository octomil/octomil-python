"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class BillingMetric(str, Enum):
    HOSTED_REQUESTS_MONTHLY = "hosted_requests_monthly"
    """Hosted (cloud) inference requests served via the managed gateway. Replaces the legacy `cloud_inference_monthly` metric.
"""
    API_REQUESTS_MONTHLY = "api_requests_monthly"
    """Control-plane API requests (non-inference) authenticated against the org's API keys or session.
"""
    MANAGED_BILLABLE_MICROS_MONTHLY = "managed_billable_micros_monthly"
    """Managed-tier metered usage in micros (1e-6 of a billable unit) for consumption-priced workloads.
"""
