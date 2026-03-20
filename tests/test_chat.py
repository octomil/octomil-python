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
        assert "--port" in result.output
        assert "--temperature" in result.output
        assert "--max-tokens" in result.output

    @patch("octomil.chat.run_chat_repl")
    @patch("octomil.agents.launcher.start_serve_background")
    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    @patch("octomil.agents.launcher._auto_select_model", return_value="qwen-coder-3b")
    def test_chat_auto_selects_model(
        self,
        mock_auto: MagicMock,
        mock_running: MagicMock,
        mock_serve: MagicMock,
        mock_repl: MagicMock,
    ) -> None:
        mock_proc = MagicMock()
        mock_serve.return_value = mock_proc
        runner = CliRunner()
        result = runner.invoke(main, ["chat"])
        assert result.exit_code == 0
        mock_auto.assert_called_once()
        mock_serve.assert_called_once_with("qwen-coder-3b", port=8080)
        mock_proc.terminate.assert_called_once()

    @patch("octomil.chat.run_chat_repl")
    @patch("octomil.agents.launcher.is_serve_running", return_value=True)
    def test_chat_uses_existing_server(
        self,
        mock_running: MagicMock,
        mock_repl: MagicMock,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "llama-8b"])
        assert result.exit_code == 0
        assert "Using existing server" in result.output
        mock_repl.assert_called_once()

    @patch("octomil.chat.run_chat_repl")
    @patch("octomil.agents.launcher.start_serve_background")
    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    def test_chat_starts_server_with_explicit_model(
        self,
        mock_running: MagicMock,
        mock_serve: MagicMock,
        mock_repl: MagicMock,
    ) -> None:
        mock_serve.return_value = MagicMock()
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "llama-8b", "--port", "9090"])
        assert result.exit_code == 0
        mock_serve.assert_called_once_with("llama-8b", port=9090)

    @patch("octomil.chat.run_chat_repl")
    @patch("octomil.agents.launcher.start_serve_background")
    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    def test_chat_terminates_server_on_exit(
        self,
        mock_running: MagicMock,
        mock_serve: MagicMock,
        mock_repl: MagicMock,
    ) -> None:
        mock_proc = MagicMock()
        mock_serve.return_value = mock_proc
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "test-model"])
        assert result.exit_code == 0
        mock_proc.terminate.assert_called_once()

    @patch("octomil.chat.run_chat_repl", side_effect=KeyboardInterrupt)
    @patch("octomil.agents.launcher.start_serve_background")
    @patch("octomil.agents.launcher.is_serve_running", return_value=False)
    def test_chat_terminates_server_on_keyboard_interrupt(
        self,
        mock_running: MagicMock,
        mock_serve: MagicMock,
        mock_repl: MagicMock,
    ) -> None:
        mock_proc = MagicMock()
        mock_serve.return_value = mock_proc
        runner = CliRunner()
        runner.invoke(main, ["chat", "test-model"])
        # KeyboardInterrupt causes non-zero exit
        mock_proc.terminate.assert_called_once()
