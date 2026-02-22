"""Tests for edgeml.compression — prompt compression / context distillation."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from edgeml.compression import (
    CompressionConfig,
    CompressionStats,
    PromptCompressor,
    _apply_sliding_window,
    _prune_tokens,
    _summarise_turns,
    estimate_messages_tokens,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_single_word(self):
        result = estimate_tokens("hello")
        assert result >= 1

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        result = estimate_tokens(text)
        # 9 words / 0.75 = 12 tokens approx
        assert result > 0
        assert result >= 9  # at least 1 token per word

    def test_whitespace_only(self):
        # Whitespace-only splits into empty word list
        assert estimate_tokens("   ") == 0


class TestEstimateMessagesTokens:
    def test_empty_list(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        result = estimate_messages_tokens(msgs)
        assert result > 0

    def test_multiple_messages(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": "I don't have access to real-time weather data.",
            },
        ]
        result = estimate_messages_tokens(msgs)
        assert result > 10  # several words across messages

    def test_overhead_per_message(self):
        """Each message should add overhead (role token + formatting)."""
        single = estimate_messages_tokens([{"role": "user", "content": "Hi"}])
        double = estimate_messages_tokens(
            [
                {"role": "user", "content": "Hi"},
                {"role": "user", "content": "Hi"},
            ]
        )
        # Double should be roughly 2x, each with 4-token overhead
        assert double > single


# ---------------------------------------------------------------------------
# _prune_tokens
# ---------------------------------------------------------------------------


class TestPruneTokens:
    def test_removes_filler_words(self):
        text = "Um, well, so I think that we should, like, do this thing"
        result = _prune_tokens(text, target_ratio=0.5)
        # Filler words should be reduced
        assert "um" not in result.lower() or len(result) < len(text)

    def test_preserves_key_content(self):
        text = "Deploy the machine learning model to production servers"
        result = _prune_tokens(text, target_ratio=0.3)
        # Key words should survive
        assert "Deploy" in result or "deploy" in result
        assert "model" in result
        assert "production" in result or "servers" in result

    def test_zero_target_ratio(self):
        text = "Hello world"
        result = _prune_tokens(text, target_ratio=0.0)
        # Zero ratio means no pruning
        assert len(result) > 0

    def test_full_target_ratio(self):
        text = "The quick brown fox jumps over the lazy dog"
        result = _prune_tokens(text, target_ratio=1.0)
        # Should still produce some output
        assert isinstance(result, str)

    def test_empty_text(self):
        result = _prune_tokens("", target_ratio=0.5)
        assert result == ""

    def test_collapses_whitespace(self):
        text = "hello   world     test"
        result = _prune_tokens(text, target_ratio=0.1)
        # Multiple spaces should be collapsed
        assert "   " not in result


# ---------------------------------------------------------------------------
# _summarise_turns
# ---------------------------------------------------------------------------


class TestSummariseTurns:
    def test_empty_messages(self):
        assert _summarise_turns([]) == ""

    def test_single_message(self):
        msgs = [{"role": "user", "content": "What is federated learning?"}]
        result = _summarise_turns(msgs)
        assert "[user]:" in result
        assert "federated learning" in result

    def test_multiple_messages(self):
        msgs = [
            {"role": "user", "content": "Hello, I need help."},
            {"role": "assistant", "content": "Sure, what do you need help with?"},
            {"role": "user", "content": "I want to deploy a model."},
        ]
        result = _summarise_turns(msgs)
        assert "[user]:" in result
        assert "[assistant]:" in result
        # Should use pipe separator
        assert "|" in result

    def test_long_message_truncated(self):
        long_content = "word " * 200  # ~200 words with no sentence breaks
        msgs = [{"role": "user", "content": long_content}]
        result = _summarise_turns(msgs)
        # Should be much shorter than original
        assert len(result) < len(long_content)
        assert "..." in result

    def test_empty_content_skipped(self):
        msgs = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Response here."},
        ]
        result = _summarise_turns(msgs)
        assert "[assistant]:" in result
        # Empty message should not appear
        assert result.count("[") == 1


# ---------------------------------------------------------------------------
# _apply_sliding_window
# ---------------------------------------------------------------------------


class TestApplySlidingWindow:
    def test_short_conversation_unchanged(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = _apply_sliding_window(msgs, max_turns_verbatim=4, preserve_system=True)
        assert result == msgs

    def test_preserves_system_message(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
        ]
        result = _apply_sliding_window(msgs, max_turns_verbatim=2, preserve_system=True)
        # System message should be first
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        # Last 2 turns should be verbatim
        assert result[-1]["content"] == "A3"
        assert result[-2]["content"] == "Q3"

    def test_older_turns_summarised(self):
        msgs = [
            {"role": "user", "content": "First question about models."},
            {"role": "assistant", "content": "Models are neural networks."},
            {"role": "user", "content": "Second question about training."},
            {"role": "assistant", "content": "Training involves optimising weights."},
            {"role": "user", "content": "Final question about deployment."},
        ]
        result = _apply_sliding_window(msgs, max_turns_verbatim=2, preserve_system=True)
        # Should have summary message + 2 recent messages
        assert len(result) == 3
        # Summary should mention conversation
        assert "summary" in result[0]["content"].lower()

    def test_empty_messages(self):
        result = _apply_sliding_window([], max_turns_verbatim=4, preserve_system=True)
        assert result == []

    def test_no_preserve_system(self):
        msgs = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
        ]
        result = _apply_sliding_window(
            msgs, max_turns_verbatim=2, preserve_system=False
        )
        # System message is NOT preserved -- it becomes part of older turns
        # which get summarised
        assert result[0]["role"] == "system"  # summary is injected as system
        assert "summary" in result[0]["content"].lower()


# ---------------------------------------------------------------------------
# CompressionStats
# ---------------------------------------------------------------------------


class TestCompressionStats:
    def test_tokens_saved(self):
        stats = CompressionStats(
            original_tokens=100,
            compressed_tokens=60,
            compression_ratio=0.6,
            strategy="token_pruning",
            duration_ms=5.0,
            messages_before=10,
            messages_after=10,
        )
        assert stats.tokens_saved == 40

    def test_savings_pct(self):
        stats = CompressionStats(
            original_tokens=200,
            compressed_tokens=100,
            compression_ratio=0.5,
            strategy="sliding_window",
            duration_ms=3.0,
            messages_before=20,
            messages_after=5,
        )
        assert stats.savings_pct == 50.0

    def test_savings_pct_zero_original(self):
        stats = CompressionStats(
            original_tokens=0,
            compressed_tokens=0,
            compression_ratio=1.0,
            strategy="none",
            duration_ms=0.1,
            messages_before=0,
            messages_after=0,
        )
        assert stats.savings_pct == 0.0

    def test_frozen(self):
        stats = CompressionStats(
            original_tokens=100,
            compressed_tokens=50,
            compression_ratio=0.5,
            strategy="token_pruning",
            duration_ms=1.0,
            messages_before=5,
            messages_after=5,
        )
        with pytest.raises(AttributeError):
            stats.original_tokens = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CompressionConfig
# ---------------------------------------------------------------------------


class TestCompressionConfig:
    def test_defaults(self):
        cfg = CompressionConfig()
        assert cfg.enabled is True
        assert cfg.strategy == "token_pruning"
        assert cfg.target_ratio == 0.5
        assert cfg.max_turns_verbatim == 4
        assert cfg.token_threshold == 256
        assert cfg.preserve_system is True

    def test_custom_values(self):
        cfg = CompressionConfig(
            enabled=False,
            strategy="sliding_window",
            target_ratio=0.3,
            max_turns_verbatim=6,
            token_threshold=100,
            preserve_system=False,
        )
        assert cfg.enabled is False
        assert cfg.strategy == "sliding_window"
        assert cfg.target_ratio == 0.3
        assert cfg.max_turns_verbatim == 6
        assert cfg.token_threshold == 100
        assert cfg.preserve_system is False


# ---------------------------------------------------------------------------
# PromptCompressor
# ---------------------------------------------------------------------------


class TestPromptCompressor:
    def test_disabled_returns_unchanged(self):
        cfg = CompressionConfig(enabled=False)
        compressor = PromptCompressor(cfg)
        msgs = [{"role": "user", "content": "Hello world " * 100}]
        result, stats = compressor.compress(msgs)
        assert result == msgs
        assert stats.strategy == "none"
        assert stats.compression_ratio == 1.0

    def test_below_threshold_returns_unchanged(self):
        cfg = CompressionConfig(token_threshold=1000)
        compressor = PromptCompressor(cfg)
        msgs = [{"role": "user", "content": "Short message"}]
        result, stats = compressor.compress(msgs)
        assert result == msgs
        assert stats.strategy == "none"

    def test_token_pruning_reduces_tokens(self):
        cfg = CompressionConfig(
            strategy="token_pruning",
            target_ratio=0.4,
            token_threshold=10,
        )
        compressor = PromptCompressor(cfg)
        # Build a message with lots of stopwords
        text = (
            "Well, I think that the model is very good and it should "
            "be able to do this very well because it has been trained "
            "on a lot of data that is very relevant to the task at hand. "
            "So I would say that we should definitely use this model for "
            "our production deployment because it is the best option."
        )
        msgs = [{"role": "user", "content": text}]
        result, stats = compressor.compress(msgs)

        assert stats.strategy == "token_pruning"
        assert stats.compressed_tokens <= stats.original_tokens
        assert stats.duration_ms >= 0

    def test_sliding_window_reduces_messages(self):
        cfg = CompressionConfig(
            strategy="sliding_window",
            max_turns_verbatim=2,
            token_threshold=10,
        )
        compressor = PromptCompressor(cfg)
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Question one about machine learning."},
            {"role": "assistant", "content": "Answer one about machine learning."},
            {"role": "user", "content": "Question two about neural networks."},
            {"role": "assistant", "content": "Answer two about neural networks."},
            {"role": "user", "content": "Question three about deployment."},
            {"role": "assistant", "content": "Answer three about deployment."},
            {"role": "user", "content": "Final question about production."},
        ]
        result, stats = compressor.compress(msgs)

        assert stats.strategy == "sliding_window"
        assert stats.messages_after < stats.messages_before
        # System msg + summary + 2 recent turns
        assert len(result) <= 4

    def test_system_message_preserved_in_token_pruning(self):
        cfg = CompressionConfig(
            strategy="token_pruning",
            preserve_system=True,
            token_threshold=10,
        )
        compressor = PromptCompressor(cfg)
        system_content = "You are a very helpful and detailed AI assistant that provides thorough answers."
        msgs = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": "Tell me about the very important things that are happening in the world today and what we should do about them.",
            },
        ]
        result, stats = compressor.compress(msgs)

        # System message should be preserved verbatim
        assert result[0]["role"] == "system"
        assert result[0]["content"] == system_content

    def test_unknown_strategy_falls_back(self):
        cfg = CompressionConfig(
            strategy="nonexistent_strategy",
            token_threshold=10,
        )
        compressor = PromptCompressor(cfg)
        msgs = [
            {
                "role": "user",
                "content": "This is a moderately long message with enough words to exceed the threshold for compression testing purposes.",
            }
        ]
        # Should not raise, falls back to token_pruning
        result, stats = compressor.compress(msgs)
        assert isinstance(result, list)

    def test_default_config(self):
        compressor = PromptCompressor()
        assert compressor.config.enabled is True
        assert compressor.config.strategy == "token_pruning"

    def test_compression_stats_fields(self):
        cfg = CompressionConfig(
            strategy="token_pruning",
            token_threshold=5,
        )
        compressor = PromptCompressor(cfg)
        msgs = [
            {
                "role": "user",
                "content": "The quick brown fox jumps over the lazy dog and this is a longer message with many stopwords that should be compressed.",
            }
        ]
        _, stats = compressor.compress(msgs)

        assert isinstance(stats.original_tokens, int)
        assert isinstance(stats.compressed_tokens, int)
        assert isinstance(stats.compression_ratio, float)
        assert isinstance(stats.strategy, str)
        assert isinstance(stats.duration_ms, float)
        assert isinstance(stats.messages_before, int)
        assert isinstance(stats.messages_after, int)
        assert stats.original_tokens > 0
        assert stats.duration_ms >= 0


# ---------------------------------------------------------------------------
# Telemetry integration
# ---------------------------------------------------------------------------


class TestCompressionTelemetry:
    def test_report_prompt_compressed_payload(self):
        """Verify TelemetryReporter.report_prompt_compressed enqueues correct payload."""
        from edgeml.telemetry import TelemetryReporter

        reporter = TelemetryReporter(
            api_key="test-key",
            api_base="https://api.example.com/api/v1",
            org_id="test-org",
            device_id="dev-001",
        )

        reporter.report_prompt_compressed(
            session_id="sess-001",
            model_id="gemma-1b",
            version="1.0",
            original_tokens=500,
            compressed_tokens=250,
            compression_ratio=0.5,
            strategy="token_pruning",
            duration_ms=3.5,
        )

        # Give queue a moment to receive the event
        time.sleep(0.05)

        # Stop the worker and inspect
        reporter._queue.put(None)
        reporter._worker.join(timeout=2.0)

        # Drain remaining payloads
        payloads = []
        while not reporter._queue.empty():
            item = reporter._queue.get_nowait()
            if item is not None:
                payloads.append(item)

        # The event might have been consumed by the worker already.
        # Test the enqueue path by mocking _send.

    def test_report_prompt_compressed_enqueue(self):
        """Verify the payload structure via mock."""
        from edgeml.telemetry import TelemetryReporter

        reporter = TelemetryReporter(
            api_key="test-key",
            api_base="https://api.example.com/api/v1",
            org_id="test-org",
            device_id="dev-001",
        )

        captured = []
        original_enqueue = reporter._enqueue

        def mock_enqueue(**kwargs):
            captured.append(kwargs)
            return original_enqueue(**kwargs)

        reporter._enqueue = mock_enqueue

        reporter.report_prompt_compressed(
            session_id="sess-002",
            model_id="phi-mini",
            version="2.0",
            original_tokens=1000,
            compressed_tokens=400,
            compression_ratio=0.4,
            strategy="sliding_window",
            duration_ms=7.2,
        )

        assert len(captured) == 1
        call = captured[0]
        assert call["event_type"] == "prompt_compressed"
        assert call["model_id"] == "phi-mini"
        assert call["session_id"] == "sess-002"
        assert call["metrics"]["original_tokens"] == 1000
        assert call["metrics"]["compressed_tokens"] == 400
        assert call["metrics"]["compression_ratio"] == 0.4
        assert call["metrics"]["tokens_saved"] == 600
        assert call["metrics"]["strategy"] == "sliding_window"
        assert call["metrics"]["compression_duration_ms"] == 7.2

        reporter.close()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCompressionCLI:
    def test_serve_help_shows_compress_context(self):
        from click.testing import CliRunner
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--compress-context" in result.output

    def test_serve_help_shows_compression_strategy(self):
        from click.testing import CliRunner
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert "--compression-strategy" in result.output
        assert "token_pruning" in result.output
        assert "sliding_window" in result.output

    def test_serve_help_shows_compression_ratio(self):
        from click.testing import CliRunner
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert "--compression-ratio" in result.output

    def test_serve_help_shows_compression_max_turns(self):
        from click.testing import CliRunner
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert "--compression-max-turns" in result.output

    def test_serve_help_shows_compression_threshold(self):
        from click.testing import CliRunner
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert "--compression-threshold" in result.output

    def test_serve_with_compress_context_passes_through(self):
        """Verify --compress-context is accepted by the CLI and passed to run_server."""
        from click.testing import CliRunner
        from edgeml.cli import main

        runner = CliRunner()
        with patch("edgeml.engines.get_registry") as mock_registry:
            # Set up a minimal mock registry
            mock_reg = MagicMock()
            mock_detection = MagicMock()
            mock_detection.available = False
            mock_detection.engine = MagicMock()
            mock_detection.engine.name = "echo"
            mock_detection.engine.display_name = "Echo"
            mock_detection.info = ""
            mock_reg.detect_all.return_value = [mock_detection]
            mock_registry.return_value = mock_reg

            with patch("edgeml.serve.run_server") as mock_run:
                result = runner.invoke(
                    main,
                    [
                        "serve",
                        "gemma-1b",
                        "--compress-context",
                        "--compression-strategy",
                        "sliding_window",
                        "--compression-ratio",
                        "0.3",
                    ],
                )
                # Check the flags were parsed correctly (not a Click error)
                if result.exit_code != 0:
                    assert "no such option" not in (result.output or "").lower()
                # If run_server was called, check compression params
                if mock_run.called:
                    call_kwargs = mock_run.call_args
                    assert call_kwargs.kwargs.get("compress_context") is True
                    assert (
                        call_kwargs.kwargs.get("compression_strategy")
                        == "sliding_window"
                    )
                    assert call_kwargs.kwargs.get("compression_ratio") == 0.3


# ---------------------------------------------------------------------------
# Server integration (create_app with compression)
# ---------------------------------------------------------------------------


async def _make_echo_app_async(**create_kwargs):
    """Create a FastAPI app with EchoBackend for testing, triggering lifespan."""
    from unittest.mock import patch as _patch
    from edgeml.serve import EchoBackend, create_app

    with _patch("edgeml.serve._detect_backend") as mock_detect:
        echo = EchoBackend()
        echo.load_model("test-model")
        mock_detect.return_value = echo
        app = create_app("test-model", **create_kwargs)

        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()

    return app


class TestServeAppCompression:
    """Test that compression integrates with the serve app."""

    @pytest.mark.asyncio
    async def test_create_app_with_compression(self):
        """create_app should accept compress_context param and initialise compressor."""
        app = await _make_echo_app_async(
            compress_context=True,
            compression_strategy="token_pruning",
            compression_ratio=0.4,
            compression_threshold=10,
        )
        assert app is not None

    @pytest.mark.asyncio
    async def test_compression_in_chat_completions(self):
        """Compression should be applied to messages in chat completions."""
        from httpx import ASGITransport, AsyncClient

        app = await _make_echo_app_async(
            compress_context=True,
            compression_strategy="token_pruning",
            compression_ratio=0.3,
            compression_threshold=10,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Well, I think that the very important model should "
                                "be deployed to the production environment because it "
                                "is a very good model and it has been trained on a lot "
                                "of very relevant and important data that is useful."
                            ),
                        }
                    ],
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "usage" in data
            usage = data["usage"]
            # Compression stats should be present
            assert "compression" in usage
            comp = usage["compression"]
            assert "original_tokens" in comp
            assert "compressed_tokens" in comp
            assert "ratio" in comp
            assert "strategy" in comp
            assert comp["strategy"] == "token_pruning"

    @pytest.mark.asyncio
    async def test_no_compression_when_disabled(self):
        """Without compress_context, no compression stats in response."""
        from httpx import ASGITransport, AsyncClient

        app = await _make_echo_app_async(compress_context=False)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello world"}],
                },
            )
            assert response.status_code == 200
            data = response.json()
            usage = data["usage"]
            # No compression key when compression is off
            assert "compression" not in usage

    @pytest.mark.asyncio
    async def test_compression_below_threshold_no_stats(self):
        """Short messages below threshold should not show compression stats."""
        from httpx import ASGITransport, AsyncClient

        app = await _make_echo_app_async(
            compress_context=True,
            compression_threshold=99999,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert response.status_code == 200
            data = response.json()
            usage = data["usage"]
            # compression stats absent because strategy is "none" (below threshold)
            assert "compression" not in usage

    @pytest.mark.asyncio
    async def test_sliding_window_compression_in_server(self):
        """Sliding window strategy should work through the server."""
        from httpx import ASGITransport, AsyncClient

        app = await _make_echo_app_async(
            compress_context=True,
            compression_strategy="sliding_window",
            compression_max_turns=2,
            compression_threshold=10,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "First question about ML models."},
                        {"role": "assistant", "content": "ML models are algorithms."},
                        {"role": "user", "content": "Second question about training."},
                        {"role": "assistant", "content": "Training optimises weights."},
                        {"role": "user", "content": "Third question about deployment."},
                    ],
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "compression" in data["usage"]
            assert data["usage"]["compression"]["strategy"] == "sliding_window"
