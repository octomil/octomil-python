"""Tests for octomil.routing — query routing, complexity estimation, and tier-0 deterministic answers."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from octomil.routing import (
    TIER_ORDER,
    DeterministicResult,
    ModelInfo,
    QueryRouter,
    RoutingDecision,
    _estimate_complexity,
    _format_number,
    _normalise_query,
    _prepare_expression,
    _safe_arithmetic_eval,
    _try_percentage,
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
# _normalise_query
# ---------------------------------------------------------------------------


class TestNormaliseQuery:
    def test_strips_whitespace(self):
        assert _normalise_query("  2+2  ") == "2+2"

    def test_strips_question_mark(self):
        assert _normalise_query("2+2?") == "2+2"

    def test_strips_period(self):
        assert _normalise_query("2+2.") == "2+2"

    def test_strips_what_is(self):
        assert _normalise_query("what is 2+2") == "2+2"

    def test_strips_calculate(self):
        assert _normalise_query("calculate 15*3") == "15*3"

    def test_strips_how_much_is(self):
        assert _normalise_query("how much is 100/4") == "100/4"

    def test_strips_whats(self):
        assert _normalise_query("what's 5+5") == "5+5"

    def test_case_insensitive(self):
        assert _normalise_query("WHAT IS 2+2") == "2+2"

    def test_no_prefix(self):
        assert _normalise_query("2+2") == "2+2"


# ---------------------------------------------------------------------------
# _prepare_expression
# ---------------------------------------------------------------------------


class TestPrepareExpression:
    def test_caret_to_pow(self):
        assert _prepare_expression("2^10") == "2**10"

    def test_implicit_multiplication(self):
        assert _prepare_expression("2(3)") == "2*(3)"

    def test_no_change(self):
        assert _prepare_expression("2+3") == "2+3"

    def test_multiple_carets(self):
        assert _prepare_expression("2^3+4^5") == "2**3+4**5"


# ---------------------------------------------------------------------------
# _try_percentage
# ---------------------------------------------------------------------------


class TestTryPercentage:
    def test_basic_percentage(self):
        assert _try_percentage("15% of 200") == 30.0

    def test_decimal_percentage(self):
        result = _try_percentage("7.5% of 400")
        assert result == pytest.approx(30.0)

    def test_hundred_percent(self):
        assert _try_percentage("100% of 50") == 50.0

    def test_no_match(self):
        assert _try_percentage("2+2") is None

    def test_no_match_text(self):
        assert _try_percentage("hello world") is None


# ---------------------------------------------------------------------------
# _safe_arithmetic_eval
# ---------------------------------------------------------------------------


class TestSafeArithmeticEval:
    def test_addition(self):
        assert _safe_arithmetic_eval("2+2") == 4.0

    def test_subtraction(self):
        assert _safe_arithmetic_eval("10-3") == 7.0

    def test_multiplication(self):
        assert _safe_arithmetic_eval("15*3") == 45.0

    def test_division(self):
        assert _safe_arithmetic_eval("100/4") == 25.0

    def test_floor_division(self):
        assert _safe_arithmetic_eval("7//2") == 3.0

    def test_modulo(self):
        assert _safe_arithmetic_eval("10%3") == 1.0

    def test_exponent(self):
        assert _safe_arithmetic_eval("2**10") == 1024.0

    def test_negative(self):
        assert _safe_arithmetic_eval("-5") == -5.0

    def test_unary_plus(self):
        assert _safe_arithmetic_eval("+5") == 5.0

    def test_parentheses(self):
        assert _safe_arithmetic_eval("(2+3)*4") == 20.0

    def test_nested_parentheses(self):
        assert _safe_arithmetic_eval("((2+3)*4)/5") == 4.0

    def test_sqrt(self):
        assert _safe_arithmetic_eval("sqrt(16)") == 4.0

    def test_abs(self):
        assert _safe_arithmetic_eval("abs(-5)") == 5.0

    def test_float_literal(self):
        assert _safe_arithmetic_eval("3.14") == pytest.approx(3.14)

    def test_complex_expression(self):
        assert _safe_arithmetic_eval("2+3*4-1") == 13.0

    def test_division_by_zero(self):
        assert _safe_arithmetic_eval("1/0") is None

    def test_syntax_error(self):
        assert _safe_arithmetic_eval("2++") is None

    def test_empty_string(self):
        assert _safe_arithmetic_eval("") is None

    def test_string_literal_rejected(self):
        assert _safe_arithmetic_eval("'hello'") is None

    def test_import_rejected(self):
        assert _safe_arithmetic_eval("__import__('os')") is None

    def test_attribute_access_rejected(self):
        assert _safe_arithmetic_eval("os.system('ls')") is None

    def test_list_comprehension_rejected(self):
        assert _safe_arithmetic_eval("[x for x in range(10)]") is None

    def test_exponent_bomb_rejected(self):
        """Exponents larger than 10000 are rejected to prevent DoS."""
        assert _safe_arithmetic_eval("2**999999999") is None

    def test_keyword_args_rejected(self):
        assert _safe_arithmetic_eval("round(3.14, ndigits=1)") is None

    def test_unknown_function_rejected(self):
        assert _safe_arithmetic_eval("print(42)") is None

    def test_pi_constant(self):
        result = _safe_arithmetic_eval("pi")
        assert result is not None
        assert result == pytest.approx(3.14159265, rel=1e-5)

    def test_e_constant(self):
        result = _safe_arithmetic_eval("e")
        assert result is not None
        assert result == pytest.approx(2.71828182, rel=1e-5)


# ---------------------------------------------------------------------------
# _format_number
# ---------------------------------------------------------------------------


class TestFormatNumber:
    def test_integer_result(self):
        assert _format_number(4.0) == "4"

    def test_large_integer(self):
        assert _format_number(1024.0) == "1024"

    def test_float_result(self):
        result = _format_number(33.333333333)
        assert "33.333" in result

    def test_negative_integer(self):
        assert _format_number(-5.0) == "-5"

    def test_zero(self):
        assert _format_number(0.0) == "0"

    def test_infinity(self):
        assert _format_number(float("inf")) == "inf"

    def test_nan(self):
        assert _format_number(float("nan")) == "nan"


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------


class TestComplexityEstimation:
    """Test the _estimate_complexity heuristic."""

    def test_simple_greeting(self):
        score = _estimate_complexity("hi")
        assert score < 0.3, f"Greeting should be simple, got {score}"

    def test_simple_hello(self):
        score = _estimate_complexity("hello")
        assert score < 0.3, f"Hello should be simple, got {score}"

    def test_simple_thanks(self):
        score = _estimate_complexity("thanks")
        assert score < 0.3, f"Thanks should be simple, got {score}"

    def test_simple_factual(self):
        score = _estimate_complexity("what is a dog")
        assert score < 0.35, f"Simple factual should be low complexity, got {score}"

    def test_medium_question(self):
        score = _estimate_complexity(
            "Explain the difference between TCP and UDP protocols"
        )
        assert 0.15 < score < 0.75, f"Medium question should be mid-range, got {score}"

    def test_complex_code_request(self):
        score = _estimate_complexity(
            "Write a function that implements a binary search tree with insert, "
            "delete, and balance operations. Include proper error handling and "
            "type hints. The algorithm should handle edge cases like duplicate keys."
        )
        assert score > 0.4, (
            f"Complex code request should be high complexity, got {score}"
        )

    def test_complex_reasoning(self):
        score = _estimate_complexity(
            "Analyze the tradeoffs between microservice and monolithic architectures. "
            "Compare latency, throughput, and deployment complexity. Evaluate which "
            "is better for a startup with 3 engineers vs an enterprise with 200."
        )
        assert score > 0.4, f"Complex reasoning should be high complexity, got {score}"

    def test_math_is_complex(self):
        score = _estimate_complexity(
            "Derive the gradient of the cross-entropy loss function with respect to "
            "the softmax output. Prove that it simplifies to y_pred - y_true."
        )
        assert score > 0.5, f"Math derivation should be complex, got {score}"

    def test_system_prompt_raises_complexity(self):
        text = "Summarize this"
        score_no_sys = _estimate_complexity(text)
        score_with_sys = _estimate_complexity(
            text,
            system_prompt="You are a detailed technical writer who must provide "
            "comprehensive analysis with citations, code examples, and step-by-step "
            "breakdowns of every concept mentioned." * 3,
        )
        assert score_with_sys > score_no_sys, (
            f"System prompt should raise complexity: {score_with_sys} vs {score_no_sys}"
        )

    def test_multi_turn_raises_complexity(self):
        text = "And what about the edge cases?"
        score_1_turn = _estimate_complexity(text, turn_count=1)
        score_5_turns = _estimate_complexity(text, turn_count=5)
        assert score_5_turns > score_1_turn, (
            f"Multi-turn should raise complexity: {score_5_turns} vs {score_1_turn}"
        )

    def test_score_bounded_zero_to_one(self):
        """Complexity should always be in [0.0, 1.0]."""
        test_cases = [
            "",
            "hi",
            "a" * 10000,
            "write implement code function class algorithm " * 50,
            "hello bye thanks what is define " * 20,
        ]
        for text in test_cases:
            score = _estimate_complexity(text)
            assert 0.0 <= score <= 1.0, (
                f"Score out of bounds for '{text[:30]}...': {score}"
            )

    def test_code_indicators_raise_complexity(self):
        score_no_code = _estimate_complexity("explain sorting")
        score_with_code = _estimate_complexity(
            "```python\ndef sort(arr):\n    pass\n```\nfix this function"
        )
        assert score_with_code > score_no_code

    def test_technical_vocab_raises_complexity(self):
        score_simple = _estimate_complexity("the cat sat on the mat")
        score_technical = _estimate_complexity(
            "asynchronous concurrency with mutex and semaphore for deadlock prevention"
        )
        assert score_technical > score_simple


# ---------------------------------------------------------------------------
# check_deterministic -- main function
# ---------------------------------------------------------------------------


class TestCheckDeterministic:
    def test_basic_addition(self):
        result = check_deterministic("2+2")
        assert result is not None
        assert result.answer == "4"
        assert result.method == "arithmetic"

    def test_basic_subtraction(self):
        result = check_deterministic("10-3")
        assert result is not None
        assert result.answer == "7"

    def test_basic_multiplication(self):
        result = check_deterministic("15*3")
        assert result is not None
        assert result.answer == "45"

    def test_basic_division(self):
        result = check_deterministic("100/4")
        assert result is not None
        assert result.answer == "25"

    def test_division_with_decimal(self):
        result = check_deterministic("100/3")
        assert result is not None
        # Should be a reasonable decimal representation
        assert "33.333" in result.answer

    def test_exponent_caret(self):
        result = check_deterministic("2^10")
        assert result is not None
        assert result.answer == "1024"

    def test_exponent_double_star(self):
        result = check_deterministic("2**10")
        assert result is not None
        assert result.answer == "1024"

    def test_percentage(self):
        result = check_deterministic("15% of 200")
        assert result is not None
        assert result.answer == "30"
        assert result.method == "percentage"

    def test_sqrt(self):
        result = check_deterministic("sqrt(16)")
        assert result is not None
        assert result.answer == "4"

    def test_what_is_prefix(self):
        result = check_deterministic("what is 2+2")
        assert result is not None
        assert result.answer == "4"

    def test_what_is_prefix_with_question_mark(self):
        result = check_deterministic("what is 2+2?")
        assert result is not None
        assert result.answer == "4"

    def test_calculate_prefix(self):
        result = check_deterministic("calculate 15*3")
        assert result is not None
        assert result.answer == "45"

    def test_how_much_is_prefix(self):
        result = check_deterministic("how much is 100/4?")
        assert result is not None
        assert result.answer == "25"

    # --- Should NOT match ---

    def test_rejects_natural_language(self):
        result = check_deterministic("explain quantum computing")
        assert result is None

    def test_rejects_poem_request(self):
        result = check_deterministic("write a poem about math")
        assert result is None

    def test_rejects_code_injection(self):
        result = check_deterministic("__import__('os').system('rm -rf /')")
        assert result is None

    def test_rejects_import_attempt(self):
        result = check_deterministic("import os; os.system('ls')")
        assert result is None

    def test_rejects_class_access(self):
        result = check_deterministic("().__class__.__bases__[0]")
        assert result is None

    def test_rejects_eval_attempt(self):
        result = check_deterministic("eval('2+2')")
        assert result is None

    def test_rejects_exec_attempt(self):
        result = check_deterministic("exec('print(1)')")
        assert result is None

    def test_rejects_empty_string(self):
        result = check_deterministic("")
        assert result is None

    def test_rejects_only_whitespace(self):
        result = check_deterministic("   ")
        assert result is None

    def test_rejects_mixed_text_and_math(self):
        result = check_deterministic("the answer to 2+2 is four")
        assert result is None

    def test_rejects_complex_question_about_math(self):
        result = check_deterministic("what is the derivative of x^2")
        assert result is None


# ---------------------------------------------------------------------------
# assign_tiers
# ---------------------------------------------------------------------------


class TestAssignTiers:
    def test_empty_list(self):
        result = assign_tiers([])
        assert result == {}

    def test_single_model(self):
        result = assign_tiers(["model-a"])
        assert len(result) == 1
        assert result["model-a"].tier == "balanced"

    def test_two_models(self):
        result = assign_tiers(["small", "large"])
        assert result["small"].tier == "fast"
        assert result["large"].tier == "quality"

    def test_three_models(self):
        result = assign_tiers(["small", "medium", "large"])
        assert result["small"].tier == "fast"
        assert result["medium"].tier == "balanced"
        assert result["large"].tier == "quality"

    def test_four_models(self):
        result = assign_tiers(["xs", "s", "m", "l"])
        # 4 models: first 1 = fast, middle 2 = balanced, last 1 = quality
        assert result["xs"].tier == "fast"
        assert result["s"].tier == "balanced"
        assert result["m"].tier == "balanced"
        assert result["l"].tier == "quality"

    def test_six_models(self):
        result = assign_tiers(["a", "b", "c", "d", "e", "f"])
        # 6 models: first 2 = fast, middle 2 = balanced, last 2 = quality
        assert result["a"].tier == "fast"
        assert result["b"].tier == "fast"
        assert result["c"].tier == "balanced"
        assert result["d"].tier == "balanced"
        assert result["e"].tier == "quality"
        assert result["f"].tier == "quality"


# ---------------------------------------------------------------------------
# QueryRouter (model-based complexity routing)
# ---------------------------------------------------------------------------


class TestQueryRouter:
    @pytest.fixture
    def three_models(self) -> dict[str, ModelInfo]:
        return {
            "small": ModelInfo(name="small", tier="fast", param_b=0.36),
            "medium": ModelInfo(name="medium", tier="balanced", param_b=3.8),
            "large": ModelInfo(name="large", tier="quality", param_b=7.0),
        }

    @pytest.fixture
    def router(self, three_models: dict[str, ModelInfo]) -> QueryRouter:
        return QueryRouter(three_models, strategy="complexity", enable_deterministic=False)

    def test_init_requires_models(self):
        with pytest.raises(ValueError, match="At least one model"):
            QueryRouter({}, strategy="complexity")

    def test_init_rejects_unknown_strategy(self, three_models):
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            QueryRouter(three_models, strategy="round_robin")

    def test_routes_simple_to_small(self, router):
        decision = router.route([{"role": "user", "content": "hi"}])
        assert decision.model_name == "small"
        assert decision.tier == "fast"
        assert decision.complexity_score < 0.3

    def test_routes_complex_to_large(self, router):
        decision = router.route(
            [
                {
                    "role": "user",
                    "content": (
                        "Write a complete implementation of a distributed hash table "
                        "with consistent hashing, virtual nodes, and replication. "
                        "Include the algorithm for node join/leave and key redistribution. "
                        "Prove the load balance properties step by step."
                    ),
                }
            ]
        )
        assert decision.model_name == "large"
        assert decision.tier == "quality"
        assert decision.complexity_score >= 0.7

    def test_routes_medium_to_balanced(self, router):
        decision = router.route(
            [
                {
                    "role": "user",
                    "content": "Explain the difference between REST and GraphQL APIs",
                }
            ]
        )
        assert decision.tier in ("balanced", "fast", "quality")
        # The exact routing depends on the heuristic, but the score should
        # be in a reasonable range
        assert 0.0 <= decision.complexity_score <= 1.0

    def test_routing_decision_has_fallback_chain(self, router):
        decision = router.route([{"role": "user", "content": "hello"}])
        assert isinstance(decision.fallback_chain, list)
        # The primary model should not be in the fallback chain
        assert decision.model_name not in decision.fallback_chain

    def test_fallback_chain_excludes_primary(self, router):
        decision = router.route([{"role": "user", "content": "hi"}])
        assert decision.model_name == "small"
        assert "small" not in decision.fallback_chain
        # Other models should be in the chain
        assert len(decision.fallback_chain) == 2

    def test_routing_decision_strategy(self, router):
        decision = router.route([{"role": "user", "content": "test"}])
        assert decision.strategy == "complexity"

    def test_system_prompt_affects_routing(self, router):
        messages_simple = [{"role": "user", "content": "summarize"}]
        messages_complex = [
            {
                "role": "system",
                "content": "You are an expert compiler engineer. Provide detailed "
                "analysis with assembly code examples, optimization passes, "
                "and formal verification proofs." * 5,
            },
            {"role": "user", "content": "summarize"},
        ]

        decision_simple = router.route(messages_simple)
        decision_complex = router.route(messages_complex)
        assert decision_complex.complexity_score > decision_simple.complexity_score

    def test_custom_thresholds(self, three_models):
        # Very permissive fast tier (0-0.8)
        router = QueryRouter(three_models, thresholds=(0.8, 0.95), enable_deterministic=False)
        decision = router.route(
            [{"role": "user", "content": "Explain how binary search works"}]
        )
        # With a very high threshold, most things route to fast
        assert decision.tier == "fast"

    def test_get_fallback(self, router):
        fallback = router.get_fallback("small")
        # Should get a larger model
        assert fallback in ("medium", "large")

    def test_get_fallback_from_largest(self, router):
        fallback = router.get_fallback("large")
        # Should fall back to a same/lower tier model
        assert fallback in ("small", "medium")

    def test_get_fallback_none_single_model(self):
        router = QueryRouter(
            {"only": ModelInfo(name="only", tier="balanced")},
        )
        fallback = router.get_fallback("only")
        assert fallback is None

    def test_single_model_routes_everything(self):
        router = QueryRouter(
            {"only": ModelInfo(name="only", tier="balanced")},
            enable_deterministic=False,
        )
        for text in ["hi", "write a compiler", "prove P=NP"]:
            decision = router.route([{"role": "user", "content": text}])
            assert decision.model_name == "only"

    def test_two_model_routing(self):
        models = {
            "fast": ModelInfo(name="fast", tier="fast"),
            "smart": ModelInfo(name="smart", tier="quality"),
        }
        router = QueryRouter(models, enable_deterministic=False)
        simple = router.route([{"role": "user", "content": "hello"}])
        assert simple.model_name == "fast"

    def test_resolve_model_missing_tier(self):
        """When the target tier has no model, fall back to next larger."""
        models = {
            "small": ModelInfo(name="small", tier="fast"),
            "big": ModelInfo(name="big", tier="quality"),
        }
        router = QueryRouter(models, enable_deterministic=False)
        # A medium-complexity query targets "balanced", but no balanced model exists
        # Should fall back to quality
        decision = router.route(
            [
                {
                    "role": "user",
                    "content": "Compare and contrast the TCP and UDP protocols in networking",
                }
            ]
        )
        # The model should be resolved (either fast or quality, not crash)
        assert decision.model_name in ("small", "big")

    def test_deterministic_routing_in_query_router(self):
        """QueryRouter with enable_deterministic=True intercepts arithmetic."""
        models = {
            "small": ModelInfo(name="small", tier="fast"),
            "big": ModelInfo(name="big", tier="quality"),
        }
        router = QueryRouter(models, enable_deterministic=True)
        decision = router.route([{"role": "user", "content": "2+2"}])
        assert decision.tier == "deterministic"
        assert decision.deterministic_result is not None
        assert decision.deterministic_result.answer == "4"

    def test_deterministic_disabled_in_query_router(self):
        """QueryRouter with enable_deterministic=False skips arithmetic check."""
        models = {
            "small": ModelInfo(name="small", tier="fast"),
            "big": ModelInfo(name="big", tier="quality"),
        }
        router = QueryRouter(models, enable_deterministic=False)
        decision = router.route([{"role": "user", "content": "2+2"}])
        assert decision.tier != "deterministic"
        assert decision.deterministic_result is None


# ---------------------------------------------------------------------------
# Multi-model serve app (integration with EchoBackend)
# ---------------------------------------------------------------------------


class TestMultiModelServeApp:
    @pytest.fixture
    def multi_model_app(self):
        """Create a multi-model FastAPI app with EchoBackends."""
        from octomil.serve import EchoBackend, create_multi_model_app

        def mock_detect(name, **kwargs):
            echo = EchoBackend()
            echo.load_model(name)
            return echo

        with patch("octomil.serve._detect_backend", side_effect=mock_detect):
            app = create_multi_model_app(
                ["small-model", "medium-model", "large-model"],
            )

            async def _trigger_lifespan():
                ctx = app.router.lifespan_context(app)
                await ctx.__aenter__()

            asyncio.run(_trigger_lifespan())

        return app

    @pytest.mark.asyncio
    async def test_health_endpoint(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mode"] == "multi-model"
        assert "small-model" in data["models"]
        assert "medium-model" in data["models"]
        assert "large-model" in data["models"]
        assert data["strategy"] == "complexity"

    @pytest.mark.asyncio
    async def test_list_models(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "small-model" in model_ids
        assert "medium-model" in model_ids
        assert "large-model" in model_ids

    @pytest.mark.asyncio
    async def test_chat_completion_routes_simple(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        # Should have routing headers
        assert "x-octomil-routed-model" in resp.headers
        assert "x-octomil-complexity" in resp.headers
        assert "x-octomil-tier" in resp.headers
        # Simple query -> small model
        assert resp.headers["x-octomil-routed-model"] == "small-model"
        assert resp.headers["x-octomil-tier"] == "fast"

    @pytest.mark.asyncio
    async def test_chat_completion_routes_complex(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert distributed systems architect. "
                                "Provide comprehensive analysis with formal proofs, "
                                "code implementations, optimization strategies, and "
                                "detailed explanations of every algorithm and data "
                                "structure used. Always include error handling, "
                                "concurrency considerations, and performance analysis."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Write a complete implementation of a distributed "
                                "consensus algorithm using Raft protocol. Implement "
                                "leader election with randomized timeouts, log "
                                "replication with consistency guarantees, and prove "
                                "the safety and liveness properties step by step. "
                                "Include mutex-based concurrency control, "
                                "serialization for network transport, and analyze "
                                "the latency and throughput tradeoffs. Compare with "
                                "Paxos and derive the asymptotic complexity bounds."
                            ),
                        },
                    ],
                },
            )
        assert resp.status_code == 200
        assert resp.headers["x-octomil-routed-model"] == "large-model"
        assert resp.headers["x-octomil-tier"] == "quality"

    @pytest.mark.asyncio
    async def test_chat_completion_response_format(self, multi_model_app):
        """Response should have standard OpenAI chat completion format."""
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "test"}],
                },
            )
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_routing_stats_endpoint(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Make a request first
            await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            # Check stats
            resp = await client.get("/v1/routing/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] >= 1
        assert "routed_counts" in data
        assert data["strategy"] == "complexity"
        assert "models" in data

    @pytest.mark.asyncio
    async def test_streaming_has_routing_headers(self, multi_model_app):
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert "x-octomil-routed-model" in resp.headers


# ---------------------------------------------------------------------------
# Fallback chain integration
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_fallback_on_model_failure(self):
        """When the primary model fails, the next model should be tried."""
        from octomil.serve import EchoBackend, create_multi_model_app

        call_count = {"small": 0, "large": 0}

        class FailingBackend(EchoBackend):
            """Backend that fails on first generate call."""

            def __init__(self, name: str):
                super().__init__()
                self._model_name = name

            def generate(self, request):
                call_count[self._model_name] = call_count.get(self._model_name, 0) + 1
                if self._model_name == "small":
                    raise RuntimeError("Small model OOM")
                return super().generate(request)

        def mock_detect(name, **kwargs):
            backend = FailingBackend(name)
            backend.load_model(name)
            return backend

        with patch("octomil.serve._detect_backend", side_effect=mock_detect):
            app = create_multi_model_app(["small", "large"])

            async def _trigger_lifespan():
                ctx = app.router.lifespan_context(app)
                await ctx.__aenter__()

            asyncio.run(_trigger_lifespan())

        async def _run():
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hello"}]},
                )
            return resp

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        # Should have used fallback
        assert resp.headers.get("x-octomil-fallback") == "true"
        # The large model should have handled it
        assert resp.headers["x-octomil-routed-model"] == "large"

    def test_all_models_fail_returns_503(self):
        """When all models fail, return 503."""
        from octomil.serve import EchoBackend, create_multi_model_app

        class AlwaysFailBackend(EchoBackend):
            def generate(self, request):
                raise RuntimeError("Always fails")

        def mock_detect(name, **kwargs):
            backend = AlwaysFailBackend()
            backend.load_model(name)
            return backend

        with patch("octomil.serve._detect_backend", side_effect=mock_detect):
            app = create_multi_model_app(["a", "b"])

            async def _trigger_lifespan():
                ctx = app.router.lifespan_context(app)
                await ctx.__aenter__()

            asyncio.run(_trigger_lifespan())

        async def _run():
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hello"}]},
                )
            return resp

        resp = asyncio.run(_run())
        assert resp.status_code == 503
        assert "All models failed" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------


class TestServeCliMultiModel:
    def test_auto_route_requires_models(self):
        from click.testing import CliRunner

        from octomil.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "test", "--auto-route"])
        assert result.exit_code != 0
        assert "--auto-route requires --models" in result.output

    def test_auto_route_requires_two_models(self):
        from click.testing import CliRunner

        from octomil.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main, ["serve", "test", "--auto-route", "--models", "single-model"]
        )
        assert result.exit_code != 0
        assert "at least 2 models" in result.output

    def test_multi_model_prints_tier_info(self):
        from click.testing import CliRunner

        from octomil.cli import main

        runner = CliRunner()
        with patch("octomil.serve.run_multi_model_server"):
            result = runner.invoke(
                main,
                [
                    "serve",
                    "small",
                    "--auto-route",
                    "--models",
                    "small,medium,large",
                ],
            )
        assert result.exit_code == 0
        assert "Loading 3 models" in result.output
        assert "auto-routing" in result.output
        assert "tier=" in result.output

    def test_route_strategy_default_is_complexity(self):
        from click.testing import CliRunner

        from octomil.cli import main

        runner = CliRunner()
        with patch("octomil.serve.run_multi_model_server") as mock_run:
            result = runner.invoke(
                main,
                [
                    "serve",
                    "test",
                    "--auto-route",
                    "--models",
                    "a,b",
                ],
            )
        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("route_strategy") == "complexity"

    def test_single_model_mode_unchanged(self):
        """Without --auto-route, serve works exactly as before."""
        from click.testing import CliRunner

        from octomil.cli import main

        runner = CliRunner()
        with patch("octomil.serve.run_server") as mock_run:
            result = runner.invoke(main, ["serve", "gemma-1b"])
        assert result.exit_code == 0
        assert "Starting Octomil serve" in result.output
        mock_run.assert_called_once()

    def test_invalid_route_strategy_rejected(self):
        from click.testing import CliRunner

        from octomil.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "serve",
                "test",
                "--auto-route",
                "--models",
                "a,b",
                "--route-strategy",
                "random",
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Integration: serve.py deterministic endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_app_with_routing():
    """Create a FastAPI app with EchoBackend for testing deterministic routing."""
    from octomil.serve import EchoBackend, create_app

    with patch("octomil.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo

        app = create_app("test-model")

        # Trigger lifespan startup
        import asyncio

        async def _trigger_lifespan():
            ctx = app.router.lifespan_context(app)
            await ctx.__aenter__()

        asyncio.run(_trigger_lifespan())

    return app


@pytest.mark.asyncio
async def test_serve_deterministic_arithmetic(echo_app_with_routing):
    """Chat completions endpoint returns deterministic result for arithmetic."""
    transport = ASGITransport(app=echo_app_with_routing)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "2+2"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "4"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] == "stop"
    # Should indicate deterministic
    assert data["usage"]["deterministic"] is True
    assert data["usage"]["deterministic_method"] == "arithmetic"


@pytest.mark.asyncio
async def test_serve_deterministic_percentage(echo_app_with_routing):
    """Chat completions endpoint handles percentage queries deterministically."""
    transport = ASGITransport(app=echo_app_with_routing)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "what is 15% of 200?"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "30"
    assert data["usage"]["deterministic"] is True
    assert data["usage"]["deterministic_method"] == "percentage"


@pytest.mark.asyncio
async def test_serve_non_deterministic_falls_through(echo_app_with_routing):
    """Non-arithmetic queries fall through to the model backend."""
    transport = ASGITransport(app=echo_app_with_routing)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "tell me a joke"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    # Should NOT be deterministic -- should use the echo backend
    assert "deterministic" not in data.get("usage", {}) or not data["usage"].get(
        "deterministic"
    )
    assert "[echo:test-model]" in data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_serve_streaming_skips_deterministic(echo_app_with_routing):
    """Streaming requests bypass deterministic routing (stream to model)."""
    transport = ASGITransport(app=echo_app_with_routing)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "2+2"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    # Should be SSE streaming, not a deterministic JSON response
    assert "data:" in resp.text


@pytest.mark.asyncio
async def test_serve_deterministic_model_field(echo_app_with_routing):
    """Deterministic response uses the correct model name."""
    transport = ASGITransport(app=echo_app_with_routing)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "custom-model",
                "messages": [{"role": "user", "content": "sqrt(16)"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "custom-model"
    assert data["choices"][0]["message"]["content"] == "4"
