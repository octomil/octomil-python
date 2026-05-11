"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class CachePrivacyMode(str, Enum):
    STRICT = "strict"
    """No cross-boundary data retention. The cache MUST NOT persist any token IDs, audio bytes, text fragments, embeddings, prompts, transcripts, file paths, PHI, or PII across the scope boundary. This is the default and the fail-closed value.
strict is INCOMPATIBLE with cache_scope=runtime or cache_scope=app (Codex R3 M8): cross-session/cross-app retention requires explicit operator policy and is enforced as a hard rejection in the CachePolicy JSON Schema (see schemas/core/cache_policy.json `if/then` clause). strict implies cache_scope ∈ {request, session}.
Tokenization and frontend phoneme/phrase caches whose entries contain no user-supplied data MAY opt into a broader scope, but only by switching to policy_allowed AND documenting the data categories at the capability layer. They still MUST satisfy the no-plaintext invariant for all payload fields.
"""
    POLICY_ALLOWED = "policy_allowed"
    """Cache MAY retain data across the scope boundary as permitted by the operator's privacy policy. Requires explicit opt-in in PolicyConfig. Operators MUST document which data categories are retained and for how long. Result caches (any cache whose entries are derived from user input, including KV state caches, response caches, and embedding caches keyed on user text) MUST NOT use policy_allowed without a documented legal basis.
"""
