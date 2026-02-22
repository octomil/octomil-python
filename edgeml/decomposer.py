"""
Query decomposition for multi-task requests.

Detects when a user query contains multiple independent sub-tasks and
splits them for parallel dispatch across model tiers.  Uses regex-based
heuristics -- no LLM call required for the decomposition itself.

Usage::

    from edgeml.decomposer import QueryDecomposer, ResultMerger

    decomposer = QueryDecomposer()
    result = decomposer.decompose(messages)
    if result.decomposed:
        # route each sub-task independently
        for task in result.tasks:
            ...

    merger = ResultMerger()
    final = merger.merge(sub_task_results)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SubTask:
    """A single extracted sub-task from a decomposed query."""

    text: str  # The extracted sub-task prompt
    index: int  # Position in original query
    depends_on: list[int] = field(default_factory=list)  # Indices of prerequisites


@dataclass
class DecompositionResult:
    """Result of query decomposition."""

    decomposed: bool  # Whether decomposition happened
    tasks: list[SubTask]  # Sub-tasks (1 if not decomposed)
    original_query: str  # The original query text


@dataclass
class SubTaskResult:
    """Result from executing a single sub-task."""

    task: SubTask
    response: str
    model_used: str
    tier: str


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# Numbered list: "1. ... 2. ... 3. ..." or "1) ... 2) ... 3) ..."
# Handles both newline-separated and inline numbered lists.
_NUMBERED_LIST = re.compile(
    r"(?:^|\n|\s)(\d+)[.)]\s+(.+?)(?=(?:\s\d+[.)]\s)|\Z)",
    re.DOTALL,
)

# Explicit connectors that split independent tasks
_CONNECTOR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\band\s+also\b", re.IGNORECASE),
    re.compile(r"\badditionally\b", re.IGNORECASE),
    re.compile(r"\bplus\b", re.IGNORECASE),
    re.compile(r"\bas\s+well\s+as\b", re.IGNORECASE),
]

# Sequential markers (imply dependency)
_SEQUENTIAL_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"\bthen\b", re.IGNORECASE),
    re.compile(r"\bafter\s+that\b", re.IGNORECASE),
    re.compile(r"\bnext\b", re.IGNORECASE),
    re.compile(r"\bfinally\b", re.IGNORECASE),
]

# Dependency indicators: pronouns / references to prior task output
_DEPENDENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btranslate\s+(it|that|the\s+result)\b", re.IGNORECASE),
    re.compile(r"\bsummarize\s+(it|that|the\s+result)\b", re.IGNORECASE),
    re.compile(
        r"\b(the\s+summary|the\s+translation|the\s+output|the\s+result)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(format|rewrite|edit|revise)\s+(it|that|the\s+result)\b", re.IGNORECASE
    ),
]

# Common imperative verbs that start independent tasks
_IMPERATIVE_VERBS = {
    "summarize",
    "translate",
    "explain",
    "describe",
    "list",
    "create",
    "write",
    "generate",
    "find",
    "calculate",
    "compare",
    "analyze",
    "format",
    "convert",
    "extract",
    "outline",
    "rewrite",
    "draft",
    "tell",
    "give",
    "show",
    "provide",
    "make",
    "build",
    "define",
}


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------


def _split_numbered_list(text: str) -> list[str] | None:
    """Extract tasks from a numbered list (1. ... 2. ... or 1) ... 2) ...).

    Returns None if fewer than 2 items found.
    """
    matches = _NUMBERED_LIST.findall(text)
    if len(matches) >= 2:
        return [m[1].strip() for m in matches]
    return None


def _split_by_connectors(text: str) -> list[str] | None:
    """Split on explicit connectors: 'and also', 'additionally', etc.

    Returns None if no split happened.
    """
    # Try each connector pattern
    for pattern in _CONNECTOR_PATTERNS:
        parts = pattern.split(text)
        if len(parts) >= 2:
            cleaned = [p.strip().rstrip(",. ") for p in parts if p.strip()]
            if len(cleaned) >= 2:
                return cleaned
    return None


def _split_comma_separated_imperatives(text: str) -> list[str] | None:
    """Split comma-separated imperative clauses.

    Detects patterns like: "Summarize X, translate Y, and format Z"
    Requires ALL parts to start with an imperative verb to avoid
    splitting mid-sentence lists (e.g. "implement X, Y, and Z").
    Returns None if no valid split.
    """
    # Remove a trailing period
    text = text.rstrip(".")

    # Split on comma, and also on ", and " / ", then "
    parts = re.split(r",\s*(?:and\s+|then\s+)?", text)
    if len(parts) < 2:
        return None

    # Require ALL parts to start with an imperative verb
    cleaned: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_word = part.split()[0].lower() if part.split() else ""
        if first_word not in _IMPERATIVE_VERBS:
            return None
        cleaned.append(part)

    if len(cleaned) >= 2:
        return cleaned
    return None


def _split_sequential_markers(text: str) -> list[str] | None:
    """Split on sequential markers: 'first...then...finally'.

    Returns None if no valid split.
    """
    # Check for "first" at the beginning
    first_match = re.match(r"^first[,:]?\s+", text, re.IGNORECASE)
    if not first_match:
        return None

    # Split on then/after that/next/finally
    combined_pattern = re.compile(
        r"\b(?:then|after\s+that|next|finally)[,:]?\s+",
        re.IGNORECASE,
    )
    parts = combined_pattern.split(text)
    if len(parts) >= 2:
        # Remove "first" prefix from first part
        parts[0] = re.sub(r"^first[,:]?\s+", "", parts[0], flags=re.IGNORECASE)
        cleaned = [p.strip().rstrip(",. ") for p in parts if p.strip()]
        if len(cleaned) >= 2:
            return cleaned
    return None


def _split_multi_sentence_imperatives(text: str) -> list[str] | None:
    """Split multi-sentence text where each sentence starts with an imperative verb.

    Only triggers when ALL sentences start with imperative verbs and
    the sentences represent distinct tasks (not elaborations on a single
    request).  Returns None if fewer than 2 qualifying sentences found.
    """
    # Split on sentence boundaries (period followed by space + uppercase or end)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 2:
        return None

    cleaned: list[str] = []
    for sent in sentences:
        sent = sent.strip().rstrip(".")
        if not sent:
            continue
        first_word = sent.split()[0].lower() if sent.split() else ""
        if first_word not in _IMPERATIVE_VERBS:
            # Not all sentences are imperative -- bail out
            return None
        cleaned.append(sent)

    if len(cleaned) < 2:
        return None

    # Reject when sentences are elaborations on a single task.
    # Elaboration verbs ("implement", "include", "ensure", "handle", "add",
    # "use", "consider") following a creation verb ("write", "create", "build",
    # "design", "develop") are not independent tasks.
    _CREATION_VERBS = {"write", "create", "build", "design", "develop", "generate"}
    _ELABORATION_VERBS = {
        "implement",
        "include",
        "ensure",
        "handle",
        "add",
        "use",
        "consider",
        "make",
        "provide",
        "show",
    }
    first_verb = cleaned[0].split()[0].lower()
    if first_verb in _CREATION_VERBS:
        following_verbs = {s.split()[0].lower() for s in cleaned[1:]}
        if following_verbs & _ELABORATION_VERBS:
            return None

    return cleaned


# ---------------------------------------------------------------------------
# Dependency detection
# ---------------------------------------------------------------------------


def _detect_dependencies(tasks: list[str]) -> list[list[int]]:
    """Build dependency lists for each task.

    A task depends on a prior task if it contains:
    - Pronouns referring to prior output ("translate it", "summarize that")
    - Explicit references ("the summary", "the translation", "the output")
    - Sequential markers ("then", "after that") at the start
    """
    deps: list[list[int]] = [[] for _ in range(len(tasks))]

    for i, task_text in enumerate(tasks):
        if i == 0:
            continue  # First task has no dependencies

        # Check for dependency patterns
        has_dep = False

        for pattern in _DEPENDENCY_PATTERNS:
            if pattern.search(task_text):
                has_dep = True
                break

        # Check for sequential markers at the start of the task
        if not has_dep:
            for pattern in _SEQUENTIAL_MARKERS:
                if pattern.search(task_text) and task_text.lower().startswith(
                    ("then", "after that", "next", "finally")
                ):
                    has_dep = True
                    break

        if has_dep:
            # Depend on the immediately preceding task
            deps[i].append(i - 1)

    return deps


# ---------------------------------------------------------------------------
# QueryDecomposer
# ---------------------------------------------------------------------------


class QueryDecomposer:
    """Decomposes multi-part queries into independent sub-tasks.

    Uses regex-based heuristics to detect numbered lists, comma-separated
    imperatives, explicit connectors, and sequential markers.  Only
    decomposes when 2+ sub-tasks are detected.

    Parameters
    ----------
    min_words:
        Minimum word count for decomposition to be attempted.
        Queries shorter than this are always returned as-is.
    """

    def __init__(self, min_words: int = 15) -> None:
        self.min_words = min_words

    def decompose(self, messages: list[dict[str, str]]) -> DecompositionResult:
        """Decompose a message list into sub-tasks.

        Extracts the last user message and attempts decomposition.
        Returns a ``DecompositionResult`` with ``decomposed=False`` and
        the original query as a single task if no decomposition is possible.

        Parameters
        ----------
        messages:
            OpenAI-style message list (dicts with ``role`` and ``content``).
        """
        # Extract the last user message
        user_text = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_text = msg.get("content", "")

        if not user_text:
            return DecompositionResult(
                decomposed=False,
                tasks=[SubTask(text="", index=0)],
                original_query="",
            )

        original = user_text.strip()

        # Skip short queries
        word_count = len(original.split())
        if word_count < self.min_words:
            logger.debug(
                "Skipping decomposition: query too short (%d words < %d)",
                word_count,
                self.min_words,
            )
            return DecompositionResult(
                decomposed=False,
                tasks=[SubTask(text=original, index=0)],
                original_query=original,
            )

        # Try decomposition strategies in priority order
        parts = _split_numbered_list(original)
        if parts is None:
            parts = _split_sequential_markers(original)
        if parts is None:
            parts = _split_by_connectors(original)
        if parts is None:
            parts = _split_comma_separated_imperatives(original)
        if parts is None:
            parts = _split_multi_sentence_imperatives(original)

        # If no decomposition found or only 1 part, return as-is
        if parts is None or len(parts) < 2:
            return DecompositionResult(
                decomposed=False,
                tasks=[SubTask(text=original, index=0)],
                original_query=original,
            )

        # Build dependency DAG
        dep_lists = _detect_dependencies(parts)

        tasks = [
            SubTask(text=part, index=i, depends_on=dep_lists[i])
            for i, part in enumerate(parts)
        ]

        # Check if decomposition overhead > benefit: need at least 2 independent
        # tasks (tasks with no dependencies) to justify parallelism
        independent_count = sum(1 for t in tasks if not t.depends_on)
        if independent_count < 2 and len(tasks) == len(
            [t for t in tasks if t.depends_on]
        ):
            # All tasks are sequential -- still decompose for per-task routing
            # but log that parallelism isn't available
            logger.debug(
                "All %d sub-tasks are sequential; decomposing for tier routing",
                len(tasks),
            )

        logger.info(
            "Decomposed query into %d sub-tasks (%d independent)",
            len(tasks),
            independent_count,
        )

        return DecompositionResult(
            decomposed=True,
            tasks=tasks,
            original_query=original,
        )


# ---------------------------------------------------------------------------
# ResultMerger
# ---------------------------------------------------------------------------


class ResultMerger:
    """Merges sub-task results into a coherent final response.

    For 2 tasks, uses inline formatting (paragraph separation).
    For 3+ tasks, uses numbered sections.
    """

    def merge(self, results: list[SubTaskResult]) -> str:
        """Merge sub-task results into a single response string.

        Results are sorted by their original task index before merging.
        """
        if not results:
            return ""

        if len(results) == 1:
            return results[0].response

        # Sort by original index
        sorted_results = sorted(results, key=lambda r: r.task.index)

        if len(sorted_results) == 2:
            # Inline: separate with double newline
            return "\n\n".join(r.response for r in sorted_results)

        # 3+ tasks: numbered sections
        sections: list[str] = []
        for i, result in enumerate(sorted_results, 1):
            sections.append(f"**{i}.** {result.response}")
        return "\n\n".join(sections)
