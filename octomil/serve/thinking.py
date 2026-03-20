"""Thinking-token post-processing for reasoning models.

Strips ``<think>...</think>`` blocks from model output and separates them
into ``reasoning_content`` (following vLLM/DeepSeek convention).

Provides:
- ``strip_thinking(text)`` — stateless, for non-streaming responses
- ``ThinkingStreamParser`` — stateful, for streaming token-by-token
"""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)


def strip_thinking(text: str) -> tuple[str, str | None]:
    """Split a completed response into (content, reasoning_content).

    Returns the content with ``<think>...</think>`` removed, and the
    thinking trace as a separate string. Returns ``None`` for
    reasoning_content when no thinking block is present.
    """
    m = _THINK_RE.match(text)
    if m:
        return m.group(2).strip(), m.group(1).strip() or None
    # Unclosed <think> — treat entire output as thinking, return empty content
    if text.lstrip().startswith("<think>"):
        inner = text.lstrip().removeprefix("<think>").strip()
        return "", inner or None
    return text, None


class ThinkingStreamParser:
    """Stateful streaming parser that separates thinking tokens from content.

    Handles partial tags split across token boundaries (e.g. MLX emitting
    ``<thi`` then ``nk>`` as separate chunks).

    Starts in ``unknown`` state — if the first non-whitespace text is
    ``<think>``, enters thinking mode. Otherwise, passes through as content
    directly (model isn't using thinking mode).

    Usage::

        parser = ThinkingStreamParser()
        for token in stream:
            for field, text in parser.feed(token):
                # field is "reasoning_content" or "content"
                yield {field: text}
        # Flush any remaining buffered text
        for field, text in parser.flush():
            yield {field: text}
    """

    def __init__(self) -> None:
        self._state: str = "unknown"  # "unknown" | "thinking" | "content"
        self._buf: str = ""

    def feed(self, token: str) -> list[tuple[str, str]]:
        """Feed a token, return list of (field, text) pairs.

        ``field`` is ``"reasoning_content"`` while inside ``<think>`` block,
        ``"content"`` after ``</think>`` or when model doesn't use thinking.
        """
        self._buf += token
        return self._drain()

    def flush(self) -> list[tuple[str, str]]:
        """Flush any remaining buffered text at end of stream."""
        results: list[tuple[str, str]] = []
        if not self._buf:
            return results
        if self._state == "unknown":
            # Never saw <think> — emit as content
            results.append(("content", self._buf))
        elif self._state == "thinking":
            # Unclosed <think> — emit remainder as reasoning
            results.append(("reasoning_content", self._buf))
        elif self._state == "content":
            results.append(("content", self._buf))
        self._buf = ""
        return results

    def _drain(self) -> list[tuple[str, str]]:
        """Process buffer and emit completed segments."""
        results: list[tuple[str, str]] = []

        while self._buf:
            if self._state == "unknown":
                # Strip leading whitespace before deciding
                stripped = self._buf.lstrip()
                if not stripped:
                    # Only whitespace so far — keep buffering
                    break

                if stripped.startswith("<think>"):
                    # Confirmed: model is reasoning
                    self._state = "thinking"
                    # Discard the <think> tag and any leading whitespace
                    self._buf = stripped[len("<think>") :]
                    continue
                elif stripped.startswith("<") and "<think>".startswith(stripped):
                    # Partial match — could be start of <think>, keep buffering
                    break
                else:
                    # Not a thinking model — pass through as content
                    self._state = "content"
                    continue

            elif self._state == "thinking":
                # Look for </think> closing tag
                close_idx = self._buf.find("</think>")
                if close_idx >= 0:
                    # Emit everything before </think> as reasoning
                    reasoning = self._buf[:close_idx]
                    if reasoning:
                        results.append(("reasoning_content", reasoning))
                    # Skip past </think> and optional following whitespace
                    after = self._buf[close_idx + len("</think>") :]
                    self._buf = after.lstrip() if after and after[0] in (" ", "\n", "\r", "\t") else after
                    self._state = "content"
                    continue
                else:
                    # Check if buffer ends with a partial </think> tag
                    partial = self._check_partial_close()
                    if partial > 0:
                        # Emit everything before the partial tag
                        safe = self._buf[:-partial]
                        if safe:
                            results.append(("reasoning_content", safe))
                            self._buf = self._buf[-partial:]
                        break
                    else:
                        # No close tag — emit all as reasoning
                        results.append(("reasoning_content", self._buf))
                        self._buf = ""
                        break

            elif self._state == "content":
                # In content mode — emit everything
                results.append(("content", self._buf))
                self._buf = ""
                break

        return results

    def _check_partial_close(self) -> int:
        """Check if buffer ends with a partial ``</think>`` tag.

        Returns the length of the partial match (0 if none).
        """
        tag = "</think>"
        for length in range(min(len(tag) - 1, len(self._buf)), 0, -1):
            if self._buf.endswith(tag[:length]):
                return length
        return 0
