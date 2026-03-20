"""Tests for octomil.serve.thinking — reasoning token post-processing."""

from octomil.serve.thinking import ThinkingStreamParser, strip_thinking

# ---------------------------------------------------------------------------
# strip_thinking (stateless, non-streaming)
# ---------------------------------------------------------------------------


class TestStripThinking:
    def test_no_thinking_block(self):
        content, reasoning = strip_thinking("Hello, how can I help you?")
        assert content == "Hello, how can I help you?"
        assert reasoning is None

    def test_thinking_block_with_content(self):
        text = "<think>Let me reason about this.\nStep 1: analyze.</think>The answer is 42."
        content, reasoning = strip_thinking(text)
        assert content == "The answer is 42."
        assert reasoning == "Let me reason about this.\nStep 1: analyze."

    def test_thinking_block_with_whitespace_after(self):
        text = "<think>reasoning</think>\n\nHere is the answer."
        content, reasoning = strip_thinking(text)
        assert content == "Here is the answer."
        assert reasoning == "reasoning"

    def test_empty_thinking_block(self):
        text = "<think></think>Content here."
        content, reasoning = strip_thinking(text)
        assert content == "Content here."
        assert reasoning is None  # empty thinking -> None

    def test_unclosed_think_tag(self):
        text = "<think>I am still thinking about this..."
        content, reasoning = strip_thinking(text)
        assert content == ""
        assert reasoning == "I am still thinking about this..."

    def test_unclosed_think_tag_with_leading_whitespace(self):
        text = "  <think>thinking"
        content, reasoning = strip_thinking(text)
        assert content == ""
        assert reasoning == "thinking"

    def test_empty_string(self):
        content, reasoning = strip_thinking("")
        assert content == ""
        assert reasoning is None

    def test_only_think_tags(self):
        text = "<think></think>"
        content, reasoning = strip_thinking(text)
        assert content == ""
        assert reasoning is None

    def test_multiline_thinking(self):
        text = "<think>\nStep 1: Parse\nStep 2: Analyze\nStep 3: Respond\n</think>\nFinal answer."
        content, reasoning = strip_thinking(text)
        assert "Final answer." in content
        assert "Step 1: Parse" in reasoning
        assert "Step 3: Respond" in reasoning


# ---------------------------------------------------------------------------
# ThinkingStreamParser (stateful, streaming)
# ---------------------------------------------------------------------------


class TestThinkingStreamParser:
    def test_non_thinking_model(self):
        """Model that doesn't use <think> — all output is content."""
        parser = ThinkingStreamParser()
        results = parser.feed("Hello world")
        assert results == [("content", "Hello world")]

    def test_simple_thinking_then_content(self):
        """Complete thinking block followed by content."""
        parser = ThinkingStreamParser()
        all_results = []
        all_results.extend(parser.feed("<think>"))
        all_results.extend(parser.feed("reasoning"))
        all_results.extend(parser.feed("</think>"))
        all_results.extend(parser.feed("answer"))
        all_results.extend(parser.flush())

        reasoning_parts = [t for f, t in all_results if f == "reasoning_content"]
        content_parts = [t for f, t in all_results if f == "content"]
        assert "".join(reasoning_parts) == "reasoning"
        assert "".join(content_parts) == "answer"

    def test_partial_opening_tag(self):
        """Opening tag split across chunks."""
        parser = ThinkingStreamParser()
        all_results = []
        all_results.extend(parser.feed("<thi"))
        all_results.extend(parser.feed("nk>"))
        all_results.extend(parser.feed("reasoning text"))
        all_results.extend(parser.feed("</think>"))
        all_results.extend(parser.feed("content"))
        all_results.extend(parser.flush())

        reasoning_parts = [t for f, t in all_results if f == "reasoning_content"]
        content_parts = [t for f, t in all_results if f == "content"]
        assert "".join(reasoning_parts) == "reasoning text"
        assert "".join(content_parts) == "content"

    def test_partial_closing_tag(self):
        """Closing tag split across chunks."""
        parser = ThinkingStreamParser()
        all_results = []
        all_results.extend(parser.feed("<think>"))
        all_results.extend(parser.feed("reason"))
        all_results.extend(parser.feed("</thi"))
        all_results.extend(parser.feed("nk>"))
        all_results.extend(parser.feed("answer"))
        all_results.extend(parser.flush())

        reasoning_parts = [t for f, t in all_results if f == "reasoning_content"]
        content_parts = [t for f, t in all_results if f == "content"]
        assert "".join(reasoning_parts) == "reason"
        assert "".join(content_parts) == "answer"

    def test_unknown_to_content_no_think(self):
        """First token is normal text — should go to content directly."""
        parser = ThinkingStreamParser()
        results = parser.feed("Normal text")
        assert results == [("content", "Normal text")]

    def test_leading_whitespace_then_think(self):
        """Whitespace before <think> should be handled."""
        parser = ThinkingStreamParser()
        all_results = []
        all_results.extend(parser.feed("\n"))
        all_results.extend(parser.feed("<think>"))
        all_results.extend(parser.feed("reasoning"))
        all_results.extend(parser.feed("</think>"))
        all_results.extend(parser.feed("content"))
        all_results.extend(parser.flush())

        reasoning_parts = [t for f, t in all_results if f == "reasoning_content"]
        content_parts = [t for f, t in all_results if f == "content"]
        assert "".join(reasoning_parts) == "reasoning"
        assert "".join(content_parts) == "content"

    def test_unclosed_think_flush(self):
        """Unclosed <think> — flush should emit as reasoning."""
        parser = ThinkingStreamParser()
        all_results = []
        all_results.extend(parser.feed("<think>"))
        all_results.extend(parser.feed("still thinking"))
        all_results.extend(parser.flush())

        reasoning_parts = [t for f, t in all_results if f == "reasoning_content"]
        assert "".join(reasoning_parts) == "still thinking"

    def test_empty_feed(self):
        """Empty token should produce no results."""
        parser = ThinkingStreamParser()
        results = parser.feed("")
        assert results == []

    def test_flush_with_no_data(self):
        """Flush on empty parser should produce no results."""
        parser = ThinkingStreamParser()
        results = parser.flush()
        assert results == []

    def test_think_tag_in_single_chunk(self):
        """Entire <think>...</think> in one chunk."""
        parser = ThinkingStreamParser()
        all_results = []
        all_results.extend(parser.feed("<think>quick thought</think>the answer"))
        all_results.extend(parser.flush())

        reasoning_parts = [t for f, t in all_results if f == "reasoning_content"]
        content_parts = [t for f, t in all_results if f == "content"]
        assert "".join(reasoning_parts) == "quick thought"
        assert "".join(content_parts) == "the answer"
