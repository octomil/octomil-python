"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class CacheScope(str, Enum):
    REQUEST = "request"
    """Cache is keyed per-request. Each new request starts with an empty cache entry. Safest scope — zero cross-request data sharing.
"""
    SESSION = "session"
    """Cache is keyed per-session. State is shared across turns within a single oct_session_t lifetime but not across sessions or apps.
"""
    RUNTIME = "runtime"
    """Cache is keyed per-runtime instance. State is shared across sessions within the same oct_runtime_t lifetime. Requires cache_privacy_mode=policy_allowed.
"""
    APP = "app"
    """Cache is keyed per-app installation. State persists across runtime restarts for the same app identity. Broadest scope; requires cache_privacy_mode=policy_allowed and explicit operator opt-in.
"""
