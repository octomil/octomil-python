"""Tests for octomil.routing — thin policy-based routing client."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from octomil.routing import (
    TIER_ORDER,
    DeterministicResult,
    ModelInfo,
    PolicyClient,
    QueryRouter,
    RoutingPolicy,
    _DEFAULT_POLICY,
    _format_number,
    _safe_eval,
    assign_tiers,
    check_deterministic,
)


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_defaults(self):
        info = ModelInfo(name="test")
        assert info.tier == "balanced"
        assert info.param_b == 0.0
        assert info.loaded is True

    def test_tier_index(self):
        assert ModelInfo(name="a", tier="fast").tier_index == 0
        assert ModelInfo(name="b", tier="balanced").tier_index == 1
        assert ModelInfo(name="c", tier="quality").tier_index == 2

    def test_tier_index_unknown(self):
        """Unknown tier defaults to 1 (balanced)."""
        info = ModelInfo(name="x", tier="unknown")
        assert info.tier_index == 1

    def test_tier_order_constant(self):
        assert TIER_ORDER == ["fast", "balanced", "quality"]


# ---------------------------------------------------------------------------
# DeterministicResult dataclass
# ---------------------------------------------------------------------------


class TestDeterministicResult:
    def test_basic_fields(self):
        r = DeterministicResult(answer="4", method="arithmetic")
        assert r.answer == "4"
        assert r.method == "arithmetic"
        assert r.confidence == 1.0

    def test_custom_confidence(self):
        r = DeterministicResult(answer="42", method="test", confidence=0.9)
        assert r.confidence == 0.9


# ---------------------------------------------------------------------------
# check_deterministic
# ---------------------------------------------------------------------------


class TestCheckDeterministic:
    def test_simple_addition(self):
        result = check_deterministic("2+2")
        assert result is not None
        assert result.answer == "4"
        assert result.method == "arithmetic"

    def test_what_is_prefix(self):
        result = check_deterministic("what is 2+2?")
        assert result is not None
        assert result.answer == "4"

    def test_calculate_prefix(self):
        result = check_deterministic("calculate 15*3")
        assert result is not None
        assert result.answer == "45"

    def test_subtraction(self):
        result = check_deterministic("100 - 37")
        assert result is not None
        assert result.answer == "63"

    def test_division(self):
        result = check_deterministic("100/4")
        assert result is not None
        assert result.answer == "25"

    def test_exponent_caret(self):
        result = check_deterministic("2^10")
        assert result is not None
        assert result.answer == "1024"

    def test_sqrt(self):
        result = check_deterministic("sqrt(16)")
        assert result is not None
        assert result.answer == "4"

    def test_natural_language_returns_none(self):
        assert check_deterministic("tell me about Python") is None

    def test_empty_returns_none(self):
        assert check_deterministic("") is None

    def test_greeting_returns_none(self):
        assert check_deterministic("hello how are you?") is None

    def test_percentage(self):
        result = check_deterministic("15% of 200")
        assert result is not None
        assert result.answer == "30"


# ---------------------------------------------------------------------------
# _safe_eval
# ---------------------------------------------------------------------------


class TestSafeEval:
    def test_addition(self):
        assert _safe_eval("2+2") == 4.0

    def test_multiplication(self):
        assert _safe_eval("5*3") == 15.0

    def test_nested_parens(self):
        assert _safe_eval("(2+3)*4") == 20.0

    def test_invalid_expression(self):
        assert _safe_eval("hello") is None

    def test_syntax_error(self):
        assert _safe_eval("2++") is None

    def test_division_by_zero(self):
        assert _safe_eval("1/0") is None

    def test_exponent_too_large(self):
        assert _safe_eval("2**999999999") is None

    def test_sqrt_function(self):
        result = _safe_eval("sqrt(25)")
        assert result is not None
        assert abs(result - 5.0) < 1e-10

    def test_pi_constant(self):
        import math

        result = _safe_eval("pi")
        assert result is not None
        assert abs(result - math.pi) < 1e-10


# ---------------------------------------------------------------------------
# _format_number
# ---------------------------------------------------------------------------


class TestFormatNumber:
    def test_integer_result(self):
        assert _format_number(4.0) == "4"

    def test_float_result(self):
        result = _format_number(3.14159)
        assert "3.14159" in result

    def test_large_integer(self):
        assert _format_number(1000000.0) == "1000000"

    def test_inf(self):
        assert _format_number(float("inf")) == "inf"

    def test_negative_inf(self):
        assert _format_number(float("-inf")) == "-inf"


# ---------------------------------------------------------------------------
# RoutingPolicy
# ---------------------------------------------------------------------------


class TestRoutingPolicy:
    def test_from_dict_defaults(self):
        policy = RoutingPolicy.from_dict(_DEFAULT_POLICY)
        assert policy.version == 1
        assert policy.fast_max_words == 10
        assert policy.quality_min_words == 50
        assert policy.deterministic_enabled is True
        assert len(policy.complex_indicators) > 0

    def test_round_trip(self):
        policy = RoutingPolicy.from_dict(_DEFAULT_POLICY)
        policy.fetched_at = time.time()
        policy.etag = '"abc123"'
        restored = RoutingPolicy.from_dict(policy.to_dict())
        assert restored.version == policy.version
        assert restored.fast_max_words == policy.fast_max_words
        assert restored.etag == policy.etag

    def test_is_expired_default(self):
        policy = RoutingPolicy.from_dict(_DEFAULT_POLICY)
        assert policy.is_expired is True  # fetched_at = 0

    def test_is_expired_recent(self):
        policy = RoutingPolicy.from_dict(_DEFAULT_POLICY)
        policy.fetched_at = time.time()
        assert policy.is_expired is False

    def test_is_expired_old(self):
        policy = RoutingPolicy.from_dict(_DEFAULT_POLICY)
        policy.fetched_at = time.time() - 7200  # 2 hours ago, TTL is 1 hour
        assert policy.is_expired is True


# ---------------------------------------------------------------------------
# PolicyClient
# ---------------------------------------------------------------------------


class TestPolicyClient:
    def test_returns_default_policy_when_offline(self):
        client = PolicyClient(api_base="http://localhost:9999", api_key="test")
        policy = client.get_policy()
        assert policy.version == 1
        assert policy.fast_max_words == 10

    def test_caches_policy_to_disk(self, tmp_path: Path):
        with patch("octomil.routing._CACHE_DIR", tmp_path):
            client = PolicyClient(api_base="http://localhost:9999", api_key="test")
            # Manually set a policy and save it
            policy = RoutingPolicy.from_dict(_DEFAULT_POLICY)
            policy.fetched_at = time.time()
            client._save_to_disk(policy)

            # New client should load from disk
            client2 = PolicyClient(api_base="http://localhost:9999", api_key="test")
            loaded = client2._load_from_disk()
            assert loaded is not None
            assert loaded.version == 1

    def test_loads_expired_disk_cache_when_offline(self, tmp_path: Path):
        with patch("octomil.routing._CACHE_DIR", tmp_path):
            client = PolicyClient(api_base="http://localhost:9999", api_key="test")
            # Save an expired policy to disk
            policy = RoutingPolicy.from_dict(_DEFAULT_POLICY)
            policy.fetched_at = time.time() - 7200
            client._save_to_disk(policy)

            # Should use expired cache rather than default
            client2 = PolicyClient(api_base="http://localhost:9999", api_key="test")
            result = client2.get_policy()
            assert result.version == 1

    def test_fetches_from_server(self, tmp_path: Path):
        """Mock httpx to return a policy from the server."""
        server_policy = {
            "version": 2,
            "thresholds": {"fast_max_words": 5, "quality_min_words": 30},
            "complex_indicators": ["analyze"],
            "deterministic_enabled": False,
            "ttl_seconds": 1800,
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = server_policy
        mock_response.headers = {"cache-control": "max-age=900", "etag": '"v2"'}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("octomil.routing._CACHE_DIR", tmp_path):
            with patch("httpx.Client", return_value=mock_client):
                client = PolicyClient(
                    api_base="http://test.example.com/api/v1", api_key="key"
                )
                policy = client.get_policy()

        assert policy.version == 2
        assert policy.fast_max_words == 5
        assert policy.deterministic_enabled is False
        assert policy.ttl_seconds == 900  # from Cache-Control
        assert policy.etag == '"v2"'

    def test_304_not_modified(self, tmp_path: Path):
        """ETag-based conditional request returns 304."""
        mock_response = MagicMock()
        mock_response.status_code = 304
        mock_response.headers = {}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("octomil.routing._CACHE_DIR", tmp_path):
            with patch("httpx.Client", return_value=mock_client):
                client = PolicyClient(
                    api_base="http://test.example.com/api/v1", api_key="key"
                )
                # Pre-populate with a policy that has an etag
                old_policy = RoutingPolicy.from_dict(_DEFAULT_POLICY)
                old_policy.etag = '"v1"'
                old_policy.fetched_at = time.time() - 7200  # expired
                client._policy = old_policy

                policy = client.get_policy()

        # Should have refreshed fetched_at
        assert policy is not None
        assert policy.fetched_at > old_policy.fetched_at - 1


# ---------------------------------------------------------------------------
# QueryRouter — basic routing
# ---------------------------------------------------------------------------


_MODELS = {
    "smollm": ModelInfo(name="smollm", tier="fast", param_b=0.36),
    "phi-mini": ModelInfo(name="phi-mini", tier="balanced", param_b=3.8),
    "llama-3b": ModelInfo(name="llama-3b", tier="quality", param_b=3.0),
}


def _make_router(**kwargs) -> QueryRouter:
    """Create a router that uses the default embedded policy (no server)."""
    return QueryRouter(_MODELS, **kwargs)


class TestQueryRouter:
    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError, match="At least one model"):
            QueryRouter({})

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            QueryRouter(_MODELS, strategy="random")

    def test_short_query_routes_to_fast(self):
        router = _make_router()
        decision = router.route([{"role": "user", "content": "hi"}])
        assert decision.tier == "fast"
        assert decision.model_name == "smollm"

    def test_complex_query_routes_to_quality(self):
        router = _make_router()
        decision = router.route(
            [
                {
                    "role": "user",
                    "content": "implement a binary search tree with self-balancing",
                }
            ]
        )
        assert decision.tier == "quality"
        assert decision.model_name == "llama-3b"

    def test_medium_query_routes_to_balanced(self):
        router = _make_router()
        # 15 words, no complex indicators
        decision = router.route(
            [
                {
                    "role": "user",
                    "content": "tell me about the history of the Roman empire in brief summary please yes",
                }
            ]
        )
        assert decision.tier == "balanced"
        assert decision.model_name == "phi-mini"

    def test_deterministic_arithmetic(self):
        router = _make_router()
        decision = router.route([{"role": "user", "content": "2+2"}])
        assert decision.tier == "deterministic"
        assert decision.deterministic_result is not None
        assert decision.deterministic_result.answer == "4"

    def test_deterministic_disabled(self):
        router = _make_router(enable_deterministic=False)
        decision = router.route([{"role": "user", "content": "2+2"}])
        assert decision.tier != "deterministic"

    def test_fallback_chain(self):
        router = _make_router()
        decision = router.route([{"role": "user", "content": "hi"}])
        assert decision.model_name == "smollm"
        # Fallback chain should contain the other models
        assert len(decision.fallback_chain) == 2
        assert "smollm" not in decision.fallback_chain

    def test_get_fallback_higher_tier(self):
        router = _make_router()
        result = router.get_fallback("smollm")
        # Should suggest a higher-tier model
        assert result in ("phi-mini", "llama-3b")

    def test_get_fallback_no_models(self):
        router = QueryRouter({"only": ModelInfo(name="only", tier="balanced")})
        assert router.get_fallback("only") is None

    def test_route_uses_last_user_message(self):
        router = _make_router()
        decision = router.route(
            [
                {"role": "user", "content": "implement a complex algorithm"},
                {"role": "assistant", "content": "Sure."},
                {"role": "user", "content": "hi"},
            ]
        )
        # Should route based on "hi" (last user message)
        assert decision.tier == "fast"


# ---------------------------------------------------------------------------
# QueryRouter — decomposed routing
# ---------------------------------------------------------------------------


class TestQueryRouterDecomposed:
    def test_non_decomposed_returns_routing_decision(self):
        from octomil.routing import RoutingDecision

        router = _make_router()
        result = router.route_decomposed([{"role": "user", "content": "hello"}])
        assert isinstance(result, RoutingDecision)


# ---------------------------------------------------------------------------
# QueryRouter — model resolution & fallback
# ---------------------------------------------------------------------------


class TestModelResolution:
    def test_missing_tier_falls_back_upward(self):
        models = {
            "small": ModelInfo(name="small", tier="fast"),
            "large": ModelInfo(name="large", tier="quality"),
        }
        router = QueryRouter(models)
        # No balanced model — should fall back to quality
        assert router._resolve_model("balanced") == "large"

    def test_resolve_unknown_tier(self):
        router = _make_router()
        # Unknown tier falls back to first available from TIER_ORDER[0:]
        result = router._resolve_model("unknown")
        assert result in _MODELS


# ---------------------------------------------------------------------------
# assign_tiers
# ---------------------------------------------------------------------------


class TestAssignTiers:
    def test_empty(self):
        assert assign_tiers([]) == {}

    def test_single_model(self):
        result = assign_tiers(["model-a"])
        assert result["model-a"].tier == "balanced"

    def test_two_models(self):
        result = assign_tiers(["small", "large"])
        assert result["small"].tier == "fast"
        assert result["large"].tier == "quality"

    def test_three_models(self):
        result = assign_tiers(["s", "m", "l"])
        assert result["s"].tier == "fast"
        assert result["m"].tier == "balanced"
        assert result["l"].tier == "quality"

    def test_four_models(self):
        result = assign_tiers(["a", "b", "c", "d"])
        tiers = [result[n].tier for n in ["a", "b", "c", "d"]]
        assert "fast" in tiers
        assert "balanced" in tiers
        assert "quality" in tiers

    def test_six_models(self):
        names = ["a", "b", "c", "d", "e", "f"]
        result = assign_tiers(names)
        assert result["a"].tier == "fast"
        assert result["b"].tier == "fast"
        assert result["c"].tier == "balanced"
        assert result["d"].tier == "balanced"
        assert result["e"].tier == "quality"
        assert result["f"].tier == "quality"


# ---------------------------------------------------------------------------
# Backward compatibility — exports
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Verify that all previously-public names are still importable."""

    def test_dataclasses_importable(self):
        from octomil.routing import (
            DecomposedRoutingDecision,
            DeterministicResult,
            ModelInfo,
            RoutingDecision,
        )

        assert ModelInfo is not None
        assert RoutingDecision is not None
        assert DecomposedRoutingDecision is not None
        assert DeterministicResult is not None

    def test_queryrouter_importable(self):
        from octomil.routing import QueryRouter

        assert QueryRouter is not None

    def test_functions_importable(self):
        from octomil.routing import assign_tiers, check_deterministic

        assert assign_tiers is not None
        assert check_deterministic is not None

    def test_tier_order_importable(self):
        from octomil.routing import TIER_ORDER

        assert TIER_ORDER == ["fast", "balanced", "quality"]

    def test_init_exports(self):
        """Check that octomil.__init__ re-exports routing names."""
        import octomil

        assert hasattr(octomil, "ModelInfo")
        assert hasattr(octomil, "QueryRouter")
        assert hasattr(octomil, "RoutingDecision")
        assert hasattr(octomil, "DecomposedRoutingDecision")
        assert hasattr(octomil, "assign_tiers")
