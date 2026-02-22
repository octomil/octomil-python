"""
Prompt compression / context distillation for edge inference.

Compresses long system prompts and conversation histories before inference
to reduce context window usage and speed up prefill on memory-constrained
devices.

Two compression strategies:

1. **Token pruning** -- Removes low-information tokens (stopwords, filler,
   repeated whitespace) from messages.  Fast, no model needed.

2. **Sliding window with summary** -- Keeps the last *K* conversation turns
   verbatim and compresses earlier turns into a compact summary prefix.
   Preserves recent context while shrinking history.

Usage::

    from edgeml.compression import PromptCompressor, CompressionConfig

    cfg = CompressionConfig(strategy="sliding_window", max_turns_verbatim=4)
    compressor = PromptCompressor(cfg)
    compressed, stats = compressor.compress(messages)

"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords / low-information tokens for token pruning
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "but",
        "and",
        "or",
        "if",
        "while",
        "that",
        "this",
        "it",
        "its",
        "also",
        "about",
        "up",
        "which",
        "what",
        "who",
        "whom",
        "their",
        "them",
        "they",
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
        "we",
        "us",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
    }
)

# Filler phrases that can be stripped without information loss
_FILLER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(um|uh|ah|hmm|well|okay|ok|so|like|you know|I mean)\b", re.IGNORECASE
    ),
    re.compile(r"\s{2,}"),  # collapse multiple whitespace
]


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompressionStats:
    """Statistics from a single compression pass."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy: str
    duration_ms: float
    messages_before: int
    messages_after: int

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def savings_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100


@dataclass
class CompressionConfig:
    """Configuration for prompt compression.

    Parameters
    ----------
    enabled:
        Master switch.  When ``False``, compression is a no-op.
    strategy:
        ``"token_pruning"`` or ``"sliding_window"``.
    target_ratio:
        Target compression ratio (0.0--1.0).  E.g. 0.5 means try to
        reduce token count by ~50%.  Used by token pruning to decide
        how aggressively to prune.
    max_turns_verbatim:
        For ``sliding_window`` strategy: number of most-recent
        conversation turns to keep verbatim.  Older turns are
        summarised.
    token_threshold:
        Minimum estimated token count before compression kicks in.
        Messages shorter than this are returned unchanged.
    preserve_system:
        When ``True``, system messages are never compressed.
    """

    enabled: bool = True
    strategy: str = "token_pruning"
    target_ratio: float = 0.5
    max_turns_verbatim: int = 4
    token_threshold: int = 256
    preserve_system: bool = True


# ---------------------------------------------------------------------------
# Token estimation (fast heuristic -- no tokenizer needed)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate token count using a word-based heuristic.

    Roughly 1 token per 0.75 words for English text, which aligns with
    typical BPE tokenisers.  Good enough for compression decisions.
    """
    if not text:
        return 0
    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) / 0.75))


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate total token count across all messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        # Add overhead per message (role token + formatting)
        total += estimate_tokens(content) + 4
    return total


# ---------------------------------------------------------------------------
# Token pruning
# ---------------------------------------------------------------------------


def _prune_tokens(text: str, target_ratio: float) -> str:
    """Remove low-information tokens from *text*.

    Removes stopwords and filler phrases to approach *target_ratio*
    reduction.  Preserves sentence structure and key content words.
    """
    # First pass: strip filler phrases
    result = text
    for pattern in _FILLER_PATTERNS:
        result = pattern.sub(" ", result)

    # Estimate how much we need to prune
    original_tokens = estimate_tokens(text)
    target_tokens = int(original_tokens * (1 - target_ratio))

    if target_tokens <= 0 or original_tokens <= 0:
        return result.strip()

    current_tokens = estimate_tokens(result)

    # If we already hit the target after filler removal, return
    if current_tokens <= target_tokens:
        return re.sub(r"\s+", " ", result).strip()

    # Second pass: selectively remove stopwords
    # Split into sentences to preserve structure
    sentences = re.split(r"([.!?]+\s*)", result)
    pruned_parts: list[str] = []

    for part in sentences:
        # Keep sentence delimiters
        if re.match(r"^[.!?]+\s*$", part):
            pruned_parts.append(part)
            continue

        words = part.split()
        kept: list[str] = []
        for word in words:
            clean = re.sub(r"[^\w]", "", word.lower())
            if clean in _STOPWORDS:
                # Check if removing would bring us close enough
                current_tokens = estimate_tokens(" ".join(pruned_parts + kept))
                if current_tokens > target_tokens:
                    continue  # skip this stopword
            kept.append(word)
        pruned_parts.append(" ".join(kept))

    pruned = "".join(pruned_parts)
    return re.sub(r"\s+", " ", pruned).strip()


# ---------------------------------------------------------------------------
# Sliding window summarisation
# ---------------------------------------------------------------------------


def _summarise_turns(
    messages: list[dict[str, str]],
) -> str:
    """Create a compact summary of conversation turns.

    This is a heuristic extractive summary -- it takes the first sentence
    of each message and combines them.  No model needed.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue

        # Extract first sentence (up to ~100 chars) as representative
        first_sentence = content
        for sep in (".", "!", "?", "\n"):
            idx = content.find(sep)
            if 0 < idx < 200:
                first_sentence = content[: idx + 1]
                break
        else:
            if len(content) > 150:
                first_sentence = content[:150] + "..."

        parts.append(f"[{role}]: {first_sentence}")

    return " | ".join(parts)


def _apply_sliding_window(
    messages: list[dict[str, str]],
    max_turns_verbatim: int,
    preserve_system: bool,
) -> list[dict[str, str]]:
    """Keep last K turns verbatim, compress earlier turns into summary.

    System messages are optionally preserved in full at the start.
    """
    if not messages:
        return messages

    # Separate system messages from conversation
    system_msgs: list[dict[str, str]] = []
    conversation: list[dict[str, str]] = []

    for msg in messages:
        if msg.get("role") == "system" and preserve_system:
            system_msgs.append(msg)
        else:
            conversation.append(msg)

    # If conversation is short enough, keep everything
    if len(conversation) <= max_turns_verbatim:
        return system_msgs + conversation

    # Split: older turns to summarise, recent turns to keep
    older = conversation[:-max_turns_verbatim]
    recent = conversation[-max_turns_verbatim:]

    # Create summary of older turns
    summary_text = _summarise_turns(older)
    summary_msg: dict[str, str] = {
        "role": "system",
        "content": f"[Conversation summary: {summary_text}]",
    }

    result = list(system_msgs)
    result.append(summary_msg)
    result.extend(recent)
    return result


# ---------------------------------------------------------------------------
# PromptCompressor
# ---------------------------------------------------------------------------


class PromptCompressor:
    """Compresses chat messages before inference.

    Parameters
    ----------
    config:
        Compression configuration.  See :class:`CompressionConfig`.
    """

    def __init__(self, config: Optional[CompressionConfig] = None) -> None:
        self.config = config or CompressionConfig()

    def compress(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], CompressionStats]:
        """Compress a list of chat messages.

        Returns the (possibly modified) message list and compression
        statistics.  If compression is disabled or the messages are
        below the token threshold, the original messages are returned
        unchanged.
        """
        start = time.monotonic()
        original_tokens = estimate_messages_tokens(messages)
        messages_before = len(messages)

        # Short-circuit: disabled or below threshold
        if not self.config.enabled or original_tokens < self.config.token_threshold:
            elapsed_ms = (time.monotonic() - start) * 1000
            return messages, CompressionStats(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                strategy="none",
                duration_ms=elapsed_ms,
                messages_before=messages_before,
                messages_after=messages_before,
            )

        strategy = self.config.strategy

        if strategy == "sliding_window":
            compressed = self._compress_sliding_window(messages)
        elif strategy == "token_pruning":
            compressed = self._compress_token_pruning(messages)
        else:
            logger.warning(
                "Unknown compression strategy '%s', falling back to token_pruning",
                strategy,
            )
            compressed = self._compress_token_pruning(messages)

        compressed_tokens = estimate_messages_tokens(compressed)
        elapsed_ms = (time.monotonic() - start) * 1000
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        stats = CompressionStats(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            strategy=strategy,
            duration_ms=elapsed_ms,
            messages_before=messages_before,
            messages_after=len(compressed),
        )

        logger.info(
            "Prompt compressed: %d -> %d tokens (%.1f%% reduction, strategy=%s, %.1fms)",
            original_tokens,
            compressed_tokens,
            stats.savings_pct,
            strategy,
            elapsed_ms,
        )

        return compressed, stats

    def _compress_token_pruning(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Apply token pruning to each message's content."""
        result: list[dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Optionally preserve system messages
            if role == "system" and self.config.preserve_system:
                result.append(dict(msg))
                continue

            pruned = _prune_tokens(content, self.config.target_ratio)
            result.append({"role": role, "content": pruned})

        return result

    def _compress_sliding_window(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Apply sliding window with summary compression."""
        return _apply_sliding_window(
            messages,
            max_turns_verbatim=self.config.max_turns_verbatim,
            preserve_system=self.config.preserve_system,
        )
