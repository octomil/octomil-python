"""Auto-generated from octomil-contracts. Do not edit.

Source of truth boundaries:
  Contract defines: plan codes, feature flags, quota limits, display pricing (marketing-grade).
  Stripe defines: purchasable price IDs, charge state, collection status, subscription lifecycle.
  Prices here (monthly_cents, annual_cents) are indicative display prices, NOT authoritative billing amounts."""

from enum import Enum
from typing import NamedTuple


class BillingPlan(str, Enum):
    FREE = "free"
    """Developer sandbox for prototyping and evaluation."""
    TEAM = "team"
    """Production tier for teams shipping on-device AI."""
    ENTERPRISE = "enterprise"
    """Custom deployment with compliance, VPC, and dedicated support."""


class PlanLimits(NamedTuple):
    max_devices: int | None
    max_models: int | None
    max_environments: int | None
    storage_gb: int | None
    requests_monthly: int | None
    training_rounds_monthly: int | None
    federated_rounds_monthly: int | None
    model_downloads_monthly: int | None
    model_conversions_monthly: int | None
    data_retention_days: int | None


class PlanFeatures(NamedTuple):
    sso: bool
    federated_learning: bool
    differential_privacy: bool
    secure_aggregation: bool
    hipaa_mode: bool
    advanced_monitoring: bool
    webhooks: bool
    experiments: bool
    rollouts: bool
    scim: bool
    siem_export: bool


class PlanPricing(NamedTuple):
    monthly_cents: int | None
    annual_cents: int | None
    overage_per_device_cents: int | None


class PlanSupportLevel(str, Enum):
    COMMUNITY = "community"
    EMAIL = "email"
    DEDICATED = "dedicated"


class PlanConfig(NamedTuple):
    display_name: str
    limits: PlanLimits
    features: PlanFeatures
    pricing: PlanPricing
    support: PlanSupportLevel


PLAN_CONFIG: dict[BillingPlan, PlanConfig] = {
    BillingPlan.FREE: PlanConfig(
        display_name="Developer",
        limits=PlanLimits(
            max_devices=25,
            max_models=3,
            max_environments=1,
            storage_gb=5,
            requests_monthly=100000,
            training_rounds_monthly=100,
            federated_rounds_monthly=1,
            model_downloads_monthly=2500,
            model_conversions_monthly=20,
            data_retention_days=7,
        ),
        features=PlanFeatures(
            sso=False,
            federated_learning=True,
            differential_privacy=False,
            secure_aggregation=False,
            hipaa_mode=False,
            advanced_monitoring=False,
            webhooks=False,
            experiments=True,
            rollouts=True,
            scim=False,
            siem_export=False,
        ),
        pricing=PlanPricing(
            monthly_cents=0,
            annual_cents=0,
            overage_per_device_cents=0,
        ),
        support=PlanSupportLevel.COMMUNITY,
    ),
    BillingPlan.TEAM: PlanConfig(
        display_name="Team",
        limits=PlanLimits(
            max_devices=1000,
            max_models=20,
            max_environments=3,
            storage_gb=100,
            requests_monthly=1000000,
            training_rounds_monthly=10000,
            federated_rounds_monthly=10,
            model_downloads_monthly=50000,
            model_conversions_monthly=500,
            data_retention_days=90,
        ),
        features=PlanFeatures(
            sso=True,
            federated_learning=True,
            differential_privacy=False,
            secure_aggregation=False,
            hipaa_mode=False,
            advanced_monitoring=True,
            webhooks=True,
            experiments=True,
            rollouts=True,
            scim=False,
            siem_export=False,
        ),
        pricing=PlanPricing(
            monthly_cents=120000,
            annual_cents=1152000,
            overage_per_device_cents=5,
        ),
        support=PlanSupportLevel.EMAIL,
    ),
    BillingPlan.ENTERPRISE: PlanConfig(
        display_name="Enterprise",
        limits=PlanLimits(
            max_devices=None,
            max_models=None,
            max_environments=None,
            storage_gb=10000,
            requests_monthly=100000000,
            training_rounds_monthly=None,
            federated_rounds_monthly=None,
            model_downloads_monthly=None,
            model_conversions_monthly=None,
            data_retention_days=None,
        ),
        features=PlanFeatures(
            sso=True,
            federated_learning=True,
            differential_privacy=True,
            secure_aggregation=True,
            hipaa_mode=True,
            advanced_monitoring=True,
            webhooks=True,
            experiments=True,
            rollouts=True,
            scim=True,
            siem_export=True,
        ),
        pricing=PlanPricing(
            monthly_cents=None,
            annual_cents=None,
            overage_per_device_cents=None,
        ),
        support=PlanSupportLevel.DEDICATED,
    ),
}


# Backward-compatible dict form (for migration convenience)
PLAN_LIMITS: dict[str, dict[str, int | None]] = {
    "free": {
        "max_devices": 25,
        "max_models": 3,
        "max_environments": 1,
        "storage_gb": 5,
        "requests_monthly": 100000,
        "training_rounds_monthly": 100,
        "federated_rounds_monthly": 1,
        "model_downloads_monthly": 2500,
        "model_conversions_monthly": 20,
        "data_retention_days": 7,
    },
    "team": {
        "max_devices": 1000,
        "max_models": 20,
        "max_environments": 3,
        "storage_gb": 100,
        "requests_monthly": 1000000,
        "training_rounds_monthly": 10000,
        "federated_rounds_monthly": 10,
        "model_downloads_monthly": 50000,
        "model_conversions_monthly": 500,
        "data_retention_days": 90,
    },
    "enterprise": {
        "max_devices": None,
        "max_models": None,
        "max_environments": None,
        "storage_gb": 10000,
        "requests_monthly": 100000000,
        "training_rounds_monthly": None,
        "federated_rounds_monthly": None,
        "model_downloads_monthly": None,
        "model_conversions_monthly": None,
        "data_retention_days": None,
    },
}

PLAN_FEATURES: dict[str, dict[str, bool]] = {
    "free": {
        "sso": False,
        "federated_learning": True,
        "differential_privacy": False,
        "secure_aggregation": False,
        "hipaa_mode": False,
        "advanced_monitoring": False,
        "webhooks": False,
        "experiments": True,
        "rollouts": True,
        "scim": False,
        "siem_export": False,
    },
    "team": {
        "sso": True,
        "federated_learning": True,
        "differential_privacy": False,
        "secure_aggregation": False,
        "hipaa_mode": False,
        "advanced_monitoring": True,
        "webhooks": True,
        "experiments": True,
        "rollouts": True,
        "scim": False,
        "siem_export": False,
    },
    "enterprise": {
        "sso": True,
        "federated_learning": True,
        "differential_privacy": True,
        "secure_aggregation": True,
        "hipaa_mode": True,
        "advanced_monitoring": True,
        "webhooks": True,
        "experiments": True,
        "rollouts": True,
        "scim": True,
        "siem_export": True,
    },
}
