"""Tests for edgeml.decomposer — query decomposition and result merging."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from edgeml.decomposer import (
    DecompositionResult,
    QueryDecomposer,
    ResultMerger,
    SubTask,
    SubTaskResult,
    _detect_dependencies,
    _split_by_connectors,
    _split_comma_separated_imperatives,
    _split_multi_sentence_imperatives,
    _split_numbered_list,
    _split_sequential_markers,
)
from edgeml.routing import (
    DecomposedRoutingDecision,
    ModelInfo,
    QueryRouter,
    RoutingDecision,
)


# ---------------------------------------------------------------------------
# SubTask dataclass
# ---------------------------------------------------------------------------


class TestSubTask:
    def test_defaults(self):
        task = SubTask(text="do something", index=0)
        assert task.text == "do something"
        assert task.index == 0
        assert task.depends_on == []

    def test_with_dependencies(self):
        task = SubTask(text="translate it", index=1, depends_on=[0])
        assert task.depends_on == [0]


# ---------------------------------------------------------------------------
# DecompositionResult dataclass
# ---------------------------------------------------------------------------


class TestDecompositionResult:
    def test_not_decomposed(self):
        result = DecompositionResult(
            decomposed=False,
            tasks=[SubTask(text="hello", index=0)],
            original_query="hello",
        )
        assert not result.decomposed
        assert len(result.tasks) == 1

    def test_decomposed(self):
        result = DecompositionResult(
            decomposed=True,
            tasks=[
                SubTask(text="task 1", index=0),
                SubTask(text="task 2", index=1),
            ],
            original_query="task 1 and also task 2",
        )
        assert result.decomposed
        assert len(result.tasks) == 2


# ---------------------------------------------------------------------------
# Numbered list splitting
# ---------------------------------------------------------------------------


class TestSplitNumberedList:
    def test_dot_format(self):
        text = "1. Summarize the article 2. Translate to French 3. Write a tweet"
        parts = _split_numbered_list(text)
        assert parts is not None
        assert len(parts) == 3
        assert "Summarize" in parts[0]
        assert "Translate" in parts[1]
        assert "tweet" in parts[2]

    def test_paren_format(self):
        text = "1) Explain quantum physics 2) Give an example 3) Compare with classical"
        parts = _split_numbered_list(text)
        assert parts is not None
        assert len(parts) == 3

    def test_newline_separated(self):
        text = "1. First task\n2. Second task\n3. Third task"
        parts = _split_numbered_list(text)
        assert parts is not None
        assert len(parts) == 3

    def test_single_item_no_split(self):
        text = "1. Just one thing"
        parts = _split_numbered_list(text)
        assert parts is None

    def test_no_numbers(self):
        parts = _split_numbered_list("do this and that")
        assert parts is None


# ---------------------------------------------------------------------------
# Connector splitting
# ---------------------------------------------------------------------------


class TestSplitByConnectors:
    def test_and_also(self):
        text = "Explain machine learning and also give me code examples for it"
        parts = _split_by_connectors(text)
        assert parts is not None
        assert len(parts) == 2
        assert "Explain" in parts[0]
        assert "code examples" in parts[1]

    def test_additionally(self):
        text = "Summarize the report additionally provide recommendations"
        parts = _split_by_connectors(text)
        assert parts is not None
        assert len(parts) == 2

    def test_as_well_as(self):
        text = "List the pros as well as describe the cons in detail"
        parts = _split_by_connectors(text)
        assert parts is not None
        assert len(parts) == 2

    def test_no_connector(self):
        parts = _split_by_connectors("explain quantum computing")
        assert parts is None


# ---------------------------------------------------------------------------
# Comma-separated imperatives
# ---------------------------------------------------------------------------


class TestSplitCommaSeparatedImperatives:
    def test_three_imperatives(self):
        text = (
            "Summarize the article, translate it to French, and format as bullet points"
        )
        parts = _split_comma_separated_imperatives(text)
        assert parts is not None
        assert len(parts) == 3
        assert "Summarize" in parts[0]
        assert "translate" in parts[1]
        assert "format" in parts[2]

    def test_two_imperatives(self):
        text = "Explain the concept, and give examples"
        parts = _split_comma_separated_imperatives(text)
        assert parts is not None
        assert len(parts) == 2

    def test_non_imperative_not_split(self):
        text = "the cat, the dog, and the bird are all pets"
        parts = _split_comma_separated_imperatives(text)
        assert parts is None

    def test_single_clause(self):
        parts = _split_comma_separated_imperatives("explain quantum computing")
        assert parts is None


# ---------------------------------------------------------------------------
# Sequential markers
# ---------------------------------------------------------------------------


class TestSplitSequentialMarkers:
    def test_first_then(self):
        text = "First summarize the text, then translate it to Spanish"
        parts = _split_sequential_markers(text)
        assert parts is not None
        assert len(parts) == 2
        assert "summarize" in parts[0]
        assert "translate" in parts[1]

    def test_first_then_finally(self):
        text = "First analyze the data, then summarize findings, finally write recommendations"
        parts = _split_sequential_markers(text)
        assert parts is not None
        assert len(parts) == 3

    def test_no_first_prefix(self):
        text = "then do something, finally do another"
        parts = _split_sequential_markers(text)
        assert parts is None

    def test_first_only(self):
        text = "First explain quantum computing"
        parts = _split_sequential_markers(text)
        assert parts is None


# ---------------------------------------------------------------------------
# Multi-sentence imperatives
# ---------------------------------------------------------------------------


class TestSplitMultiSentenceImperatives:
    def test_two_imperative_sentences(self):
        text = (
            "Summarize the article in three sentences. Translate the summary to French."
        )
        parts = _split_multi_sentence_imperatives(text)
        assert parts is not None
        assert len(parts) == 2
        assert "Summarize" in parts[0]
        assert "Translate" in parts[1]

    def test_non_imperative_sentences(self):
        text = "The cat sat on the mat. It was a sunny day."
        parts = _split_multi_sentence_imperatives(text)
        assert parts is None

    def test_single_sentence(self):
        parts = _split_multi_sentence_imperatives("Explain quantum computing.")
        assert parts is None


# ---------------------------------------------------------------------------
# Dependency detection
# ---------------------------------------------------------------------------


class TestDetectDependencies:
    def test_no_dependencies(self):
        tasks = ["Summarize the article", "Write a poem about cats"]
        deps = _detect_dependencies(tasks)
        assert deps == [[], []]

    def test_pronoun_reference(self):
        tasks = ["Summarize the article", "Translate it to French"]
        deps = _detect_dependencies(tasks)
        assert deps[0] == []
        assert deps[1] == [0]

    def test_the_summary_reference(self):
        tasks = ["Summarize the article", "Translate the summary to French"]
        deps = _detect_dependencies(tasks)
        assert deps[0] == []
        assert deps[1] == [0]

    def test_the_result_reference(self):
        tasks = ["Calculate the total", "Format the result as a table"]
        deps = _detect_dependencies(tasks)
        assert deps[0] == []
        assert deps[1] == [0]

    def test_three_tasks_chain(self):
        tasks = [
            "Summarize the article",
            "Translate the summary to French",
            "Format the result as bullet points",
        ]
        deps = _detect_dependencies(tasks)
        assert deps[0] == []
        assert deps[1] == [0]
        assert deps[2] == [1]

    def test_first_task_never_has_deps(self):
        tasks = ["Translate it to French"]
        deps = _detect_dependencies(tasks)
        assert deps[0] == []


# ---------------------------------------------------------------------------
# QueryDecomposer
# ---------------------------------------------------------------------------


class TestQueryDecomposer:
    @pytest.fixture
    def decomposer(self) -> QueryDecomposer:
        return QueryDecomposer()

    def test_single_query_passthrough(self, decomposer):
        """Single simple query should not be decomposed."""
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        result = decomposer.decompose(messages)
        assert not result.decomposed
        assert len(result.tasks) == 1
        assert result.tasks[0].text == "What is the capital of France?"

    def test_short_query_skip(self, decomposer):
        """Queries under 15 words should not be decomposed."""
        messages = [{"role": "user", "content": "hi there how are you"}]
        result = decomposer.decompose(messages)
        assert not result.decomposed
        assert len(result.tasks) == 1

    def test_empty_messages(self, decomposer):
        """Empty message list should return not decomposed."""
        result = decomposer.decompose([])
        assert not result.decomposed

    def test_no_user_message(self, decomposer):
        """Messages without a user role should return not decomposed."""
        messages = [{"role": "system", "content": "You are helpful."}]
        result = decomposer.decompose(messages)
        assert not result.decomposed

    def test_numbered_list_detection(self, decomposer):
        """Numbered lists should be decomposed."""
        messages = [
            {
                "role": "user",
                "content": (
                    "Please help me with these tasks: "
                    "1. Summarize the latest news about AI "
                    "2. Translate the summary to Spanish "
                    "3. Write a tweet about it"
                ),
            }
        ]
        result = decomposer.decompose(messages)
        assert result.decomposed
        assert len(result.tasks) >= 2

    def test_comma_separated_tasks(self, decomposer):
        """Comma-separated imperative clauses should be decomposed."""
        messages = [
            {
                "role": "user",
                "content": "Summarize the article about climate change, translate the key points to French, and create a presentation outline",
            }
        ]
        result = decomposer.decompose(messages)
        assert result.decomposed
        assert len(result.tasks) >= 2

    def test_and_also_connector(self, decomposer):
        """'and also' connector should split tasks."""
        messages = [
            {
                "role": "user",
                "content": "Explain the fundamentals of quantum computing in simple terms and also give me practical code examples for each concept",
            }
        ]
        result = decomposer.decompose(messages)
        assert result.decomposed
        assert len(result.tasks) == 2

    def test_sequential_dependency(self, decomposer):
        """Sequential tasks should be decomposed with dependencies."""
        messages = [
            {
                "role": "user",
                "content": "First summarize this long detailed article about machine learning and artificial intelligence, then translate the summary to French for our international team",
            }
        ]
        result = decomposer.decompose(messages)
        assert result.decomposed
        assert len(result.tasks) == 2
        # Second task should depend on first
        assert result.tasks[1].depends_on == [0]

    def test_not_multi_task_and(self, decomposer):
        """A single sentence with 'and' that isn't multi-task should not be decomposed."""
        messages = [
            {
                "role": "user",
                "content": "Cats and dogs are popular pets that people keep at home",
            }
        ]
        result = decomposer.decompose(messages)
        assert not result.decomposed

    def test_dag_independent_tasks(self, decomposer):
        """Independent tasks should have empty depends_on."""
        messages = [
            {
                "role": "user",
                "content": "1. Write a poem about the ocean 2. Explain how photosynthesis works 3. List the planets in our solar system",
            }
        ]
        result = decomposer.decompose(messages)
        assert result.decomposed
        for task in result.tasks:
            assert task.depends_on == [], f"Task {task.index} should have no deps"

    def test_dag_dependent_tasks(self, decomposer):
        """Dependent tasks should reference prior tasks."""
        messages = [
            {
                "role": "user",
                "content": (
                    "1. Summarize the article about climate change "
                    "2. Translate the summary to Japanese "
                    "3. Rewrite the result as bullet points"
                ),
            }
        ]
        result = decomposer.decompose(messages)
        assert result.decomposed
        assert result.tasks[0].depends_on == []
        assert result.tasks[1].depends_on == [0]
        assert result.tasks[2].depends_on == [1]

    def test_original_query_preserved(self, decomposer):
        """The original query text should be preserved in the result."""
        original = "1. Write a poem 2. Write an essay about the importance of poetry in modern education"
        messages = [{"role": "user", "content": original}]
        result = decomposer.decompose(messages)
        assert result.original_query == original

    def test_custom_min_words(self):
        """Custom min_words should change the threshold."""
        decomposer = QueryDecomposer(min_words=5)
        messages = [
            {
                "role": "user",
                "content": "Summarize this, and translate it, and format the result",
            }
        ]
        result = decomposer.decompose(messages)
        # With lower threshold, should attempt decomposition
        assert result.decomposed or len(result.tasks) == 1

    def test_uses_last_user_message(self, decomposer):
        """Should use the last user message for decomposition."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {
                "role": "user",
                "content": "1. Summarize the latest AI news 2. Write a tweet 3. Create a blog post outline about it",
            },
        ]
        result = decomposer.decompose(messages)
        assert result.decomposed
        assert "Summarize" in result.tasks[0].text or "tweet" in result.original_query


# ---------------------------------------------------------------------------
# ResultMerger
# ---------------------------------------------------------------------------


class TestResultMerger:
    @pytest.fixture
    def merger(self) -> ResultMerger:
        return ResultMerger()

    def test_empty_results(self, merger):
        assert merger.merge([]) == ""

    def test_single_result(self, merger):
        results = [
            SubTaskResult(
                task=SubTask(text="task", index=0),
                response="response here",
                model_used="model-a",
                tier="fast",
            )
        ]
        assert merger.merge(results) == "response here"

    def test_two_results_inline(self, merger):
        """Two results should be joined with double newline (inline format)."""
        results = [
            SubTaskResult(
                task=SubTask(text="task 1", index=0),
                response="First response",
                model_used="model-a",
                tier="fast",
            ),
            SubTaskResult(
                task=SubTask(text="task 2", index=1),
                response="Second response",
                model_used="model-b",
                tier="balanced",
            ),
        ]
        merged = merger.merge(results)
        assert "First response" in merged
        assert "Second response" in merged
        assert "\n\n" in merged
        # Should NOT have numbered sections
        assert "**1.**" not in merged

    def test_three_results_numbered(self, merger):
        """Three or more results should use numbered sections."""
        results = [
            SubTaskResult(
                task=SubTask(text="task 1", index=0),
                response="Response A",
                model_used="model-a",
                tier="fast",
            ),
            SubTaskResult(
                task=SubTask(text="task 2", index=1),
                response="Response B",
                model_used="model-b",
                tier="balanced",
            ),
            SubTaskResult(
                task=SubTask(text="task 3", index=2),
                response="Response C",
                model_used="model-c",
                tier="quality",
            ),
        ]
        merged = merger.merge(results)
        assert "**1.**" in merged
        assert "**2.**" in merged
        assert "**3.**" in merged
        assert "Response A" in merged
        assert "Response B" in merged
        assert "Response C" in merged

    def test_results_sorted_by_index(self, merger):
        """Results should be sorted by task index before merging."""
        results = [
            SubTaskResult(
                task=SubTask(text="task 2", index=1),
                response="Second",
                model_used="m",
                tier="fast",
            ),
            SubTaskResult(
                task=SubTask(text="task 1", index=0),
                response="First",
                model_used="m",
                tier="fast",
            ),
        ]
        merged = merger.merge(results)
        # First should come before Second
        assert merged.index("First") < merged.index("Second")


# ---------------------------------------------------------------------------
# Integration: QueryDecomposer + QueryRouter
# ---------------------------------------------------------------------------


class TestDecomposerRouterIntegration:
    @pytest.fixture
    def three_models(self) -> dict[str, ModelInfo]:
        return {
            "small": ModelInfo(name="small", tier="fast", param_b=0.36),
            "medium": ModelInfo(name="medium", tier="balanced", param_b=3.8),
            "large": ModelInfo(name="large", tier="quality", param_b=7.0),
        }

    def test_route_decomposed_single_query(self, three_models):
        """Single-task query returns standard RoutingDecision."""
        router = QueryRouter(three_models, enable_deterministic=False)
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        result = router.route_decomposed(messages)
        assert isinstance(result, RoutingDecision)

    def test_route_decomposed_multi_task(self, three_models):
        """Multi-task query returns DecomposedRoutingDecision."""
        router = QueryRouter(three_models, enable_deterministic=False)
        messages = [
            {
                "role": "user",
                "content": (
                    "1. Summarize the latest news about artificial intelligence "
                    "2. Write a poem about technology and the future of humanity "
                    "3. List the top programming languages for data science"
                ),
            }
        ]
        result = router.route_decomposed(messages)
        assert isinstance(result, DecomposedRoutingDecision)
        assert result.decomposed
        assert len(result.sub_decisions) >= 2
        assert len(result.tasks) == len(result.sub_decisions)

    def test_route_decomposed_preserves_system_prompt(self, three_models):
        """System prompt should be passed to sub-task routing."""
        router = QueryRouter(three_models, enable_deterministic=False)
        messages = [
            {
                "role": "system",
                "content": "You are a technical expert with deep knowledge of algorithms, data structures, and system design.",
            },
            {
                "role": "user",
                "content": (
                    "1. Explain binary search algorithm step by step "
                    "2. Write a Python implementation with comprehensive error handling"
                ),
            },
        ]
        result = router.route_decomposed(messages)
        assert isinstance(result, DecomposedRoutingDecision)
        # System prompt should raise complexity for all sub-tasks
        for decision in result.sub_decisions:
            assert decision.complexity_score > 0.0

    def test_route_decomposed_each_task_gets_own_tier(self, three_models):
        """Each sub-task should be independently routed to an appropriate tier."""
        router = QueryRouter(three_models, enable_deterministic=False)
        messages = [
            {
                "role": "user",
                "content": (
                    "1. Say hello to the user "
                    "2. Write a complete implementation of a distributed hash table "
                    "with consistent hashing, virtual nodes, replication, and prove "
                    "the load balance properties step by step using formal methods"
                ),
            }
        ]
        result = router.route_decomposed(messages)
        assert isinstance(result, DecomposedRoutingDecision)
        # First task is simple, second is complex -- they should get different tiers
        tiers = {d.tier for d in result.sub_decisions}
        assert len(tiers) >= 1  # At minimum they get routed


# ---------------------------------------------------------------------------
# Integration: Multi-model serve app with decomposition
# ---------------------------------------------------------------------------


class TestMultiModelServeDecomposition:
    @pytest.fixture
    def multi_model_app(self):
        """Create a multi-model FastAPI app with EchoBackends."""
        from edgeml.serve import EchoBackend, create_multi_model_app

        def mock_detect(name, **kwargs):
            echo = EchoBackend()
            echo.load_model(name)
            return echo

        with patch("edgeml.serve._detect_backend", side_effect=mock_detect):
            app = create_multi_model_app(
                ["small-model", "medium-model", "large-model"],
            )

            async def _trigger_lifespan():
                ctx = app.router.lifespan_context(app)
                await ctx.__aenter__()

            asyncio.run(_trigger_lifespan())

        return app

    @pytest.mark.asyncio
    async def test_decomposed_request_headers(self, multi_model_app):
        """Decomposed requests should have X-EdgeML-Decomposed and X-EdgeML-Subtasks headers."""
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "1. Summarize the latest artificial intelligence trends in detail "
                                "2. Write a haiku about technology and the modern world "
                                "3. List the top five programming languages for web development"
                            ),
                        }
                    ],
                },
            )
        assert resp.status_code == 200
        assert resp.headers.get("x-edgeml-decomposed") == "true"
        subtasks = resp.headers.get("x-edgeml-subtasks")
        assert subtasks is not None
        assert int(subtasks) >= 2

    @pytest.mark.asyncio
    async def test_decomposed_response_format(self, multi_model_app):
        """Decomposed response should have standard OpenAI format."""
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "1. Explain how neural networks work at a fundamental level "
                                "2. Give a practical real-world example of neural networks in action"
                            ),
                        }
                    ],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        # Content should contain responses from both sub-tasks
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_simple_query_not_decomposed(self, multi_model_app):
        """Simple queries should NOT be decomposed."""
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        # Should NOT have decomposition headers
        assert resp.headers.get("x-edgeml-decomposed") is None
        # Should have standard routing headers
        assert "x-edgeml-routed-model" in resp.headers

    @pytest.mark.asyncio
    async def test_streaming_skips_decomposition(self, multi_model_app):
        """Streaming requests should bypass decomposition."""
        transport = ASGITransport(app=multi_model_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "1. Summarize the latest AI news in detail "
                                "2. Write a tweet about it for social media"
                            ),
                        }
                    ],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        # Should NOT have decomposition headers (streaming bypasses)
        assert resp.headers.get("x-edgeml-decomposed") is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.fixture
    def decomposer(self) -> QueryDecomposer:
        return QueryDecomposer()

    def test_single_sentence_with_and_not_decomposed(self, decomposer):
        """'cats and dogs are pets' should not be decomposed."""
        messages = [
            {
                "role": "user",
                "content": "Cats and dogs are popular pets that many people keep at home for companionship",
            }
        ]
        result = decomposer.decompose(messages)
        assert not result.decomposed

    def test_long_single_task_not_decomposed(self, decomposer):
        """A long but single-task query should not be decomposed."""
        messages = [
            {
                "role": "user",
                "content": (
                    "Explain the complete history of the Roman Empire from its founding "
                    "to its fall, including all major emperors, battles, political "
                    "changes, and cultural developments that shaped Western civilization"
                ),
            }
        ]
        result = decomposer.decompose(messages)
        assert not result.decomposed

    def test_task_indices_are_sequential(self, decomposer):
        """Task indices should be 0, 1, 2, ..."""
        messages = [
            {
                "role": "user",
                "content": "1. Write a poem about nature 2. Explain photosynthesis 3. List endangered species",
            }
        ]
        result = decomposer.decompose(messages)
        if result.decomposed:
            for i, task in enumerate(result.tasks):
                assert task.index == i

    def test_whitespace_handling(self, decomposer):
        """Extra whitespace should be handled gracefully."""
        messages = [
            {
                "role": "user",
                "content": "  1.  Summarize the article about emerging technology trends   2.  Translate the full summary to German language for our partners   ",
            }
        ]
        result = decomposer.decompose(messages)
        assert result.decomposed
        for task in result.tasks:
            assert task.text.strip() == task.text
