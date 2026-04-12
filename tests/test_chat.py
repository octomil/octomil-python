"""Tests for octomil chat command and REPL."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from octomil.chat import run_chat_repl
from octomil.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sse_chunk(content: str) -> dict[str, Any]:
    """Build a minimal OpenAI-compatible SSE chunk."""
    return {
        "choices": [
            {
                "delta": {"content": content},
                "index": 0,
            }
        ]
    }


def _make_input_fn(inputs: Sequence[str | None]):
    """Return a callable that yields inputs in order, then returns None."""
    it = iter(inputs)

    def _read() -> str | None:
        try:
            return next(it)
        except StopIteration:
            return None

    return _read


# ---------------------------------------------------------------------------
# run_chat_repl
# ---------------------------------------------------------------------------


class TestRunChatRepl:
    @patch("octomil.chat.stream_chat_via_responses")
    def test_exit_command(self, mock_stream: MagicMock) -> None:
        """Typing /exit breaks out of the REPL."""
        input_fn = _make_input_fn(["/exit"])
        responses = MagicMock()
        run_chat_repl(
            "test-model",
            responses,
            _input_fn=input_fn,
        )
        mock_stream.assert_not_called()

    @patch("octomil.chat.stream_chat_via_responses")
    def test_eof_exits(self, mock_stream: MagicMock) -> None:
        """None (EOF) breaks out of the REPL."""
        input_fn = _make_input_fn([])  # immediately returns None
        responses = MagicMock()
        run_chat_repl(
            "test-model",
            responses,
            _input_fn=input_fn,
        )
        mock_stream.assert_not_called()

    @patch("octomil.chat.stream_chat_via_responses")
    def test_empty_input_skipped(self, mock_stream: MagicMock) -> None:
        """Blank lines do not send a request."""
        input_fn = _make_input_fn(["", "   ", "/exit"])
        responses = MagicMock()
        run_chat_repl(
            "test-model",
            responses,
            _input_fn=input_fn,
        )
        mock_stream.assert_not_called()

    @patch("octomil.chat.stream_chat_via_responses")
    def test_clear_resets_messages(self, mock_stream: MagicMock) -> None:
        """The /clear command keeps only system messages."""
        # Track messages at each call since the list is mutable
        captured_messages: list[list[dict[str, str]]] = []

        def _capture_stream(responses, model, messages, **kwargs):
            # Snapshot the messages at call time
            captured_messages.append([m.copy() for m in messages])
            return iter([_make_sse_chunk("response")])

        mock_stream.side_effect = _capture_stream

        inputs = ["hello", "/clear", "world", "/exit"]
        input_fn = _make_input_fn(inputs)

        responses = MagicMock()
        run_chat_repl(
            "test-model",
            responses,
            system_prompt="be helpful",
            _input_fn=input_fn,
        )

        # After /clear, the second call should only have system + "world"
        assert len(captured_messages) == 2
        second_call_messages = captured_messages[1]
        roles = [m["role"] for m in second_call_messages]
        assert roles == ["system", "user"]
        assert second_call_messages[1]["content"] == "world"

    @patch("octomil.chat.stream_chat_via_responses")
    def test_streams_and_accumulates_response(self, mock_stream: MagicMock) -> None:
        """Response tokens are accumulated and appended as assistant message."""
        captured_messages: list[list[dict[str, str]]] = []

        def _capture_stream(responses, model, messages, **kwargs):
            captured_messages.append([m.copy() for m in messages])
            return iter([_make_sse_chunk("Hello"), _make_sse_chunk(" there")])

        mock_stream.side_effect = _capture_stream

        input_fn = _make_input_fn(["hi", "/exit"])
        responses = MagicMock()
        run_chat_repl(
            "test-model",
            responses,
            _input_fn=input_fn,
        )

        assert len(captured_messages) == 1
        assert captured_messages[0][-1]["role"] == "user"
        assert captured_messages[0][-1]["content"] == "hi"

    @patch("octomil.chat.stream_chat_via_responses")
    def test_multi_turn_history(self, mock_stream: MagicMock) -> None:
        """Messages accumulate across turns."""
        captured_messages: list[list[dict[str, str]]] = []

        def _capture_stream(responses, model, messages, **kwargs):
            captured_messages.append([m.copy() for m in messages])
            return iter([_make_sse_chunk(f"reply-{len(captured_messages)}")])

        mock_stream.side_effect = _capture_stream

        input_fn = _make_input_fn(["a", "b", "/exit"])
        responses = MagicMock()
        run_chat_repl(
            "test-model",
            responses,
            _input_fn=input_fn,
        )

        assert len(captured_messages) == 2
        # Second call should have full history
        roles = [m["role"] for m in captured_messages[1]]
        assert roles == ["user", "assistant", "user"]

    @patch("octomil.chat.stream_chat_via_responses")
    def test_system_prompt_preserved_after_clear(self, mock_stream: MagicMock) -> None:
        """After /clear, system prompt remains."""
        captured_messages: list[list[dict[str, str]]] = []

        def _capture_stream(responses, model, messages, **kwargs):
            captured_messages.append([m.copy() for m in messages])
            return iter([_make_sse_chunk("ok")])

        mock_stream.side_effect = _capture_stream

        input_fn = _make_input_fn(["/clear", "question", "/exit"])
        responses = MagicMock()
        run_chat_repl(
            "test-model",
            responses,
            system_prompt="you are a bot",
            _input_fn=input_fn,
        )

        assert len(captured_messages) == 1
        assert captured_messages[0][0] == {"role": "system", "content": "you are a bot"}

    @patch("octomil.chat.stream_chat_via_responses")
    def test_metrics_displayed(self, mock_stream: MagicMock) -> None:
        """Token count and timing info are printed after each response."""
        mock_stream.return_value = iter(
            [
                _make_sse_chunk("a"),
                _make_sse_chunk("b"),
                _make_sse_chunk("c"),
            ]
        )

        input_fn = _make_input_fn(["hi", "/exit"])
        responses = MagicMock()
        run_chat_repl(
            "test-model",
            responses,
            _input_fn=input_fn,
        )
        # Verify stream_chat_via_responses was called with the correct model
        assert mock_stream.call_count == 1
        assert mock_stream.call_args_list[0][0][1] == "test-model"

    @patch("octomil.chat.stream_chat_via_responses", side_effect=RuntimeError("connection failed"))
    def test_connection_error_handled(self, mock_stream: MagicMock) -> None:
        """Connection errors are caught; the REPL continues."""
        input_fn = _make_input_fn(["hello", "/exit"])
        responses = MagicMock()
        # Should not raise
        run_chat_repl(
            "test-model",
            responses,
            _input_fn=input_fn,
        )
        mock_stream.assert_called_once()

    @patch("octomil.chat.stream_chat_via_responses", side_effect=FileNotFoundError("missing cached model"))
    def test_model_load_error_handled(self, mock_stream: MagicMock) -> None:
        """Model load errors are caught without a traceback."""
        input_fn = _make_input_fn(["hello", "/exit"])
        responses = MagicMock()
        run_chat_repl(
            "test-model",
            responses,
            _input_fn=input_fn,
        )
        mock_stream.assert_called_once()

    @patch("octomil.chat.stream_chat_via_responses")
    def test_temperature_and_max_tokens_passed(self, mock_stream: MagicMock) -> None:
        """Custom temperature and max_tokens are forwarded to stream_chat_via_responses."""
        mock_stream.return_value = iter([_make_sse_chunk("ok")])

        input_fn = _make_input_fn(["test", "/exit"])
        responses = MagicMock()
        run_chat_repl(
            "test-model",
            responses,
            temperature=0.3,
            max_tokens=512,
            _input_fn=input_fn,
        )

        _, kwargs = mock_stream.call_args
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 512


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestChatCLI:
    def test_chat_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "--help"])
        assert result.exit_code == 0
        assert "Chat with a model locally" in result.output

    def test_chat_help_shows_examples(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "--help"])
        assert "octomil chat" in result.output
        assert "--system" in result.output

    def test_chat_help_shows_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "--help"])
        assert "--select" in result.output
        assert "--policy" in result.output
        assert "--temperature" in result.output
        assert "--max-tokens" in result.output

    @patch("octomil.chat.run_chat_repl")
    def test_chat_uses_direct_local_default(
        self,
        mock_repl: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["chat"])
        assert result.exit_code == 0
        assert "Chat — gemma3-1b" in result.output
        assert "shared execution kernel" in result.output
        mock_repl.assert_called_once()
        assert mock_repl.call_args.args[0] == "gemma3-1b"

    @patch("octomil.chat.run_chat_repl")
    @patch("octomil.agents.launcher._select_model_tui", return_value="llama-8b")
    def test_chat_select_uses_picker(
        self,
        mock_select: MagicMock,
        mock_repl: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "--select"])
        assert result.exit_code == 0
        mock_select.assert_called_once()
        assert "Chat — llama-8b" in result.output
        mock_repl.assert_called_once()

    @patch("octomil.chat.run_chat_repl")
    def test_chat_uses_explicit_model(
        self,
        mock_repl: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "llama-8b"])
        assert result.exit_code == 0
        assert "Chat — llama-8b" in result.output
        assert mock_repl.call_args.args[0] == "llama-8b"

    @patch("octomil.chat.run_chat_repl")
    def test_chat_forwards_policy_and_app(
        self,
        mock_repl: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "test-model", "--policy", "cloud_first", "--app", "demo"])
        assert result.exit_code == 0
        assert mock_repl.call_args.kwargs["policy"] == "cloud_first"
        assert mock_repl.call_args.kwargs["app"] == "demo"

    @patch("octomil.chat.run_chat_repl")
    def test_chat_accepts_legacy_port_without_starting_server(
        self,
        mock_repl: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "test-model", "--port", "9090"])
        assert result.exit_code == 0
        assert "--port is ignored" in result.output
        mock_repl.assert_called_once()

    @patch("octomil.chat.run_chat_repl", side_effect=KeyboardInterrupt)
    def test_chat_propagates_keyboard_interrupt(
        self,
        mock_repl: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "test-model"])
        assert result.exit_code != 0
