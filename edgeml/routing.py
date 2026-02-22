"""
Query routing for multi-model serving with tier-0 deterministic answers.

Routes incoming queries to the most appropriate model based on estimated
complexity.  Simple queries (greetings, short factual questions) go to
the smallest/fastest model; complex queries (code generation, multi-step
reasoning) go to the largest/most capable model.

**Tier 0 (deterministic)**: Intercepts arithmetic, unit conversions, and
other queries that can be answered without invoking any model.  Uses safe
AST-based evaluation -- never ``eval()`` or ``exec()``.

The complexity heuristic is pure Python — no ML model required.

Usage::

    from edgeml.routing import QueryRouter, ModelInfo

    models = {
        "smollm-360m": ModelInfo(name="smollm-360m", tier="fast", param_b=0.36),
        "phi-4-mini":  ModelInfo(name="phi-4-mini",  tier="balanced", param_b=3.8),
        "llama-3.2-3b": ModelInfo(name="llama-3.2-3b", tier="quality", param_b=3.0),
    }
    router = QueryRouter(models, strategy="complexity")
    chosen = router.route(messages)

    # Tier-0 deterministic routing:
    from edgeml.routing import check_deterministic
    result = check_deterministic("what is 2+2?")
    if result is not None:
        print(result.answer)  # "4"
"""

from __future__ import annotations

import ast
import logging
import math
import operator
import re
from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

# Tiers ordered from smallest to largest capability.
TIER_ORDER: list[str] = ["fast", "balanced", "quality"]


@dataclass
class ModelInfo:
    """Metadata about a loaded model used for routing decisions."""

    name: str
    tier: str = "balanced"
    param_b: float = 0.0  # parameter count in billions (for display)
    loaded: bool = True

    @property
    def tier_index(self) -> int:
        """Numeric rank of the tier (higher = more capable)."""
        try:
            return TIER_ORDER.index(self.tier)
        except ValueError:
            return 1  # default to balanced


# ---------------------------------------------------------------------------
# Deterministic result
# ---------------------------------------------------------------------------


@dataclass
class DeterministicResult:
    """Result from a deterministic (no-model) computation."""

    answer: str
    method: str  # e.g. "arithmetic", "unit_conversion"
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """Result of a routing decision with metadata for telemetry."""

    model_name: str
    complexity_score: float
    tier: str
    strategy: str
    fallback_chain: list[str] = field(default_factory=list)
    deterministic_result: Optional[DeterministicResult] = None


@dataclass
class DecomposedRoutingDecision:
    """Routing decision for a decomposed multi-task query.

    Contains one ``RoutingDecision`` per sub-task, allowing each
    sub-task to be routed to a different model tier.
    """

    sub_decisions: list[RoutingDecision]
    tasks: list  # list[SubTask] — typed loosely to avoid circular import
    original_query: str
    decomposed: bool = True


# ---------------------------------------------------------------------------
# Complexity heuristic signals
# ---------------------------------------------------------------------------

# Words that indicate simple / conversational queries.
_SIMPLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(hi|hello|hey|howdy|greetings|yo|sup)\b", re.IGNORECASE),
    re.compile(r"\b(thanks|thank you|bye|goodbye|see ya)\b", re.IGNORECASE),
    re.compile(r"\b(what time|what day|what year|how old)\b", re.IGNORECASE),
    re.compile(r"\bwhat is (a |an |the )?\w+\b", re.IGNORECASE),
    re.compile(r"\bdefine \w+\b", re.IGNORECASE),
]

# Words/phrases that indicate complex queries.
_COMPLEX_PATTERNS: list[re.Pattern[str]] = [
    # Code generation
    re.compile(
        r"\b(write|implement|code|program|function|class|algorithm|refactor|debug)\b",
        re.IGNORECASE,
    ),
    # Reasoning
    re.compile(
        r"\b(explain why|reason|analyze|compare|contrast|evaluate|critique)\b",
        re.IGNORECASE,
    ),
    # Multi-step
    re.compile(
        r"\b(step by step|step-by-step|first .* then|multi-step)\b",
        re.IGNORECASE,
    ),
    # Math / logic
    re.compile(
        r"\b(prove|derive|calculate|compute|integral|derivative|equation|theorem)\b",
        re.IGNORECASE,
    ),
    # Creative / long-form
    re.compile(
        r"\b(write a (story|essay|article|poem|report|document))\b",
        re.IGNORECASE,
    ),
    # Technical terms
    re.compile(
        r"\b(API|REST|GraphQL|microservice|kubernetes|docker|terraform|CICD|CI/CD)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(neural network|transformer|attention|backpropagation|gradient)\b",
        re.IGNORECASE,
    ),
]

# Technical / uncommon vocabulary (presence raises complexity).
_TECHNICAL_WORDS: set[str] = {
    "asynchronous",
    "concurrency",
    "parallelism",
    "mutex",
    "semaphore",
    "deadlock",
    "recursion",
    "polymorphism",
    "inheritance",
    "abstraction",
    "encapsulation",
    "middleware",
    "serialization",
    "deserialization",
    "latency",
    "throughput",
    "idempotent",
    "deterministic",
    "stochastic",
    "heuristic",
    "eigenvalue",
    "convolution",
    "embedding",
    "tokenization",
    "quantization",
    "optimization",
    "regularization",
    "normalization",
    "hyperparameter",
    "inference",
    "fine-tuning",
    "federated",
    "differential",
    "cryptographic",
    "authentication",
    "authorization",
    "orchestration",
    "containerization",
}


def _word_count(text: str) -> int:
    """Count whitespace-delimited tokens."""
    return len(text.split())


def _estimate_complexity(
    text: str,
    system_prompt: str = "",
    turn_count: int = 1,
) -> float:
    """Estimate query complexity on a 0.0 (trivial) to 1.0 (complex) scale.

    Signals used:
    1. Token/word count — short queries tend to be simpler.
    2. Vocabulary complexity — presence of technical terms.
    3. Pattern matching — greeting vs. code/reasoning patterns.
    4. System prompt length — longer system prompts imply complex tasks.
    5. Multi-turn context — deeper conversations are generally harder.

    Returns a float in [0.0, 1.0].
    """
    signals: list[float] = []

    # 1. Length signal (0-1, log-scaled)
    wc = _word_count(text)
    # 1 word → ~0.0, 10 words → ~0.33, 50 words → ~0.56, 200 words → ~0.77
    length_score = min(math.log1p(wc) / math.log1p(200), 1.0)
    signals.append(length_score * 0.20)

    # 2. Simple pattern match (each match lowers complexity)
    simple_hits = sum(1 for p in _SIMPLE_PATTERNS if p.search(text))
    simple_penalty = min(simple_hits * 0.15, 0.30)
    signals.append(-simple_penalty)

    # 3. Complex pattern match (each match raises complexity)
    complex_hits = sum(1 for p in _COMPLEX_PATTERNS if p.search(text))
    complex_boost = min(complex_hits * 0.12, 0.40)
    signals.append(complex_boost)

    # 4. Technical vocabulary ratio
    words_lower = {w.lower().strip(".,;:!?()[]{}\"'") for w in text.split()}
    tech_count = len(words_lower & _TECHNICAL_WORDS)
    tech_score = min(tech_count / 5.0, 1.0) * 0.15
    signals.append(tech_score)

    # 5. Code indicators (backticks, indentation patterns)
    code_indicators = text.count("```") + text.count("def ") + text.count("class ")
    code_score = min(code_indicators / 3.0, 1.0) * 0.10
    signals.append(code_score)

    # 6. System prompt length (longer → more complex task)
    if system_prompt:
        sys_wc = _word_count(system_prompt)
        sys_score = min(math.log1p(sys_wc) / math.log1p(500), 1.0) * 0.10
        signals.append(sys_score)

    # 7. Multi-turn depth (deeper → harder)
    turn_score = min((turn_count - 1) / 10.0, 1.0) * 0.05
    signals.append(turn_score)

    # Base complexity (avoids everything summing to exactly 0)
    base = 0.25

    raw = base + sum(signals)
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Safe AST-based arithmetic evaluator (Tier-0 deterministic)
# ---------------------------------------------------------------------------

# Whitelisted AST node types for safe arithmetic evaluation
_SAFE_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Num,  # Python 3.7 compat (deprecated but still parsed)
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Call,
    ast.Name,
    ast.Load,
)

# Allowed function names in expressions
_SAFE_FUNCTIONS: dict[str, callable] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}

# Allowed constants (names that resolve to values, not functions)
_SAFE_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node, allowing only arithmetic operations.

    Raises ``ValueError`` for any disallowed node type.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")

    # Python 3.7 ast.Num (deprecated but still emitted by some parsers)
    if isinstance(node, ast.Num):
        return float(node.n)

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Disallowed unary op: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        op_func = ops.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Disallowed binary op: {type(node.op).__name__}")

        # Guard against exponent bombs (e.g. 2**999999999)
        if isinstance(node.op, ast.Pow) and abs(right) > 10000:
            raise ValueError("Exponent too large")

        return op_func(left, right)

    if isinstance(node, ast.Call):
        # Only allow whitelisted function names
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Disallowed function: {func_name}")
        func = _SAFE_FUNCTIONS[func_name]
        if not callable(func):
            raise ValueError(f"{func_name} is not callable")
        args = [_safe_eval_node(arg) for arg in node.args]
        if node.keywords:
            raise ValueError("Keyword arguments not allowed")
        return func(*args)

    if isinstance(node, ast.Name):
        name = node.id
        if name in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[name]
        raise ValueError(f"Disallowed name: {name}")

    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def _safe_arithmetic_eval(expr: str) -> Optional[float]:
    """Safely evaluate a pure arithmetic expression string.

    Returns the numeric result, or ``None`` if the expression is not
    valid arithmetic.  Never uses ``eval()``/``exec()``.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    # Reject any node type not in our whitelist
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            return None

    try:
        result = _safe_eval_node(tree)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, ArithmeticError):
        return None

    return result


# ---------------------------------------------------------------------------
# Query normalisation helpers
# ---------------------------------------------------------------------------

# Phrases to strip from the beginning of queries to extract the core expression
_STRIP_PREFIXES = [
    r"what\s+is\s+",
    r"what's\s+",
    r"calculate\s+",
    r"compute\s+",
    r"evaluate\s+",
    r"how\s+much\s+is\s+",
    r"solve\s+",
    r"find\s+",
]

# Compiled pattern that matches any prefix (case-insensitive)
_PREFIX_PATTERN = re.compile(
    r"^(?:" + "|".join(_STRIP_PREFIXES) + r")",
    re.IGNORECASE,
)

# Percentage pattern: "N% of M"
_PERCENTAGE_PATTERN = re.compile(
    r"^([\d.]+)\s*%\s*(?:of)\s+([\d.]+)$",
    re.IGNORECASE,
)

# Caret exponent: "2^10" → "2**10"
_CARET_PATTERN = re.compile(r"(\d+)\s*\^\s*(\d+)")


def _normalise_query(query: str) -> str:
    """Strip natural language wrappers to get the core arithmetic expression."""
    text = query.strip()

    # Remove trailing question mark / period
    text = text.rstrip("?.")

    # Remove common prefixes ("what is", "calculate", etc.)
    text = _PREFIX_PATTERN.sub("", text).strip()

    return text


def _try_percentage(expr: str) -> Optional[float]:
    """Try to evaluate 'N% of M' patterns."""
    m = _PERCENTAGE_PATTERN.match(expr)
    if m:
        pct = float(m.group(1))
        base = float(m.group(2))
        return (pct / 100.0) * base
    return None


def _prepare_expression(expr: str) -> str:
    """Convert human-friendly math notation into Python-parseable expressions.

    Handles:
    - ``^`` → ``**``  (exponentiation)
    - Implicit multiplication before parentheses: ``2(3)`` → ``2*(3)``
    """
    # Replace ^ with **
    result = _CARET_PATTERN.sub(r"\1**\2", expr)
    # Implicit multiplication: digit followed by '(' → digit * '('
    result = re.sub(r"(\d)\s*\(", r"\1*(", result)
    return result


# ---------------------------------------------------------------------------
# Format result
# ---------------------------------------------------------------------------


def _format_number(value: float) -> str:
    """Format a numeric result for display.

    Integers are shown without decimals. Floats are rounded to 10 significant
    digits and trailing zeros are stripped.
    """
    if value == float("inf") or value == float("-inf") or math.isnan(value):
        return str(value)

    # If it's effectively an integer, display without decimal
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))

    # Otherwise, use reasonable precision
    formatted = f"{value:.10g}"
    return formatted


# ---------------------------------------------------------------------------
# Main deterministic checker
# ---------------------------------------------------------------------------


def check_deterministic(query: str) -> Optional[DeterministicResult]:
    """Check if a query can be answered deterministically (no model needed).

    Handles:
    - Pure arithmetic: ``2+2``, ``15*3``, ``100/4``
    - Exponents: ``2^10``, ``2**10``
    - Functions: ``sqrt(16)``, ``abs(-5)``
    - Percentages: ``15% of 200``

    Returns a ``DeterministicResult`` if the query is answerable, or ``None``
    if a model is needed.

    Security: uses AST parsing only. No ``eval()`` or ``exec()``.
    """
    normalised = _normalise_query(query)

    if not normalised:
        return None

    # Quick heuristic: if the normalised expression contains letters that
    # aren't function names or constants, it's probably a natural language
    # query that needs a model.
    # Allow: digits, operators, parens, dots, whitespace, known function/const names
    # Check by removing known function names and seeing what letters remain
    check_str = normalised
    for name in sorted(_SAFE_FUNCTIONS.keys(), key=len, reverse=True):
        check_str = check_str.replace(name, "")
    # Remove 'of' (used in percentage patterns)
    check_str = re.sub(r"\bof\b", "", check_str, flags=re.IGNORECASE)
    # If there are still alphabetic characters, this isn't pure arithmetic
    if re.search(r"[a-zA-Z_]", check_str):
        return None

    # Try percentage pattern first
    pct_result = _try_percentage(normalised)
    if pct_result is not None:
        return DeterministicResult(
            answer=_format_number(pct_result),
            method="percentage",
            confidence=1.0,
        )

    # Prepare expression (convert ^ to **, etc.)
    prepared = _prepare_expression(normalised)

    # Try safe arithmetic evaluation
    result = _safe_arithmetic_eval(prepared)
    if result is not None:
        return DeterministicResult(
            answer=_format_number(result),
            method="arithmetic",
            confidence=1.0,
        )

    return None


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------


class QueryRouter:
    """Routes queries to the best model based on complexity.

    Supports an optional tier-0 deterministic layer that intercepts simple
    arithmetic and returns answers without invoking any model.

    Parameters
    ----------
    models:
        Dict mapping model name to ``ModelInfo``.  Models should be
        ordered from smallest to largest (or tagged with tiers).
    strategy:
        Routing strategy.  Currently only ``"complexity"`` is supported.
    thresholds:
        Two-element tuple ``(low, high)`` defining tier boundaries.
        Complexity in ``[0, low)`` → fast, ``[low, high)`` → balanced,
        ``[high, 1.0]`` → quality.
    enable_deterministic:
        If ``True`` (default), check for tier-0 deterministic answers
        before scoring complexity.
    """

    def __init__(
        self,
        models: dict[str, ModelInfo],
        strategy: str = "complexity",
        thresholds: tuple[float, float] = (0.3, 0.7),
        enable_deterministic: bool = True,
    ) -> None:
        if not models:
            raise ValueError("At least one model must be provided")
        if strategy != "complexity":
            raise ValueError(
                f"Unknown routing strategy '{strategy}'. "
                "Supported strategies: complexity"
            )

        self.models = models
        self.strategy = strategy
        self.thresholds = thresholds
        self.enable_deterministic = enable_deterministic

        # Build tier → model mapping (pick first model per tier)
        self._tier_models: dict[str, str] = {}
        for name, info in models.items():
            if info.tier not in self._tier_models:
                self._tier_models[info.tier] = name

        # Ordered list of model names from smallest to largest tier
        self._ordered_models: list[str] = sorted(
            models.keys(),
            key=lambda n: models[n].tier_index,
        )

        logger.info(
            "QueryRouter initialised: strategy=%s, models=%s, thresholds=%s, deterministic=%s",
            strategy,
            list(models.keys()),
            thresholds,
            enable_deterministic,
        )

    def route(self, messages: list[dict[str, str]]) -> RoutingDecision:
        """Determine which model should handle this request.

        Parameters
        ----------
        messages:
            OpenAI-style message list (dicts with ``role`` and ``content``).

        Returns
        -------
        RoutingDecision with the chosen model name, complexity score,
        tier, and fallback chain.  If the query is answered deterministically,
        ``deterministic_result`` is populated and ``tier`` is ``"deterministic"``.
        """
        # Extract relevant text
        user_text = ""
        system_prompt = ""
        turn_count = 0

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_text = content  # use the last user message
                turn_count += 1

        # Tier 0: deterministic (no model needed)
        if self.enable_deterministic and user_text:
            det = check_deterministic(user_text)
            if det is not None:
                logger.debug(
                    "Deterministic hit: %r -> %s (method=%s)",
                    user_text,
                    det.answer,
                    det.method,
                )
                return RoutingDecision(
                    model_name="",
                    complexity_score=0.0,
                    tier="deterministic",
                    strategy=self.strategy,
                    deterministic_result=det,
                )

        complexity = _estimate_complexity(
            user_text,
            system_prompt=system_prompt,
            turn_count=turn_count,
        )

        # Map complexity to tier
        low, high = self.thresholds
        if complexity < low:
            target_tier = "fast"
        elif complexity < high:
            target_tier = "balanced"
        else:
            target_tier = "quality"

        # Find the best model for the target tier
        model_name = self._resolve_model(target_tier)

        # Build fallback chain: all models from current tier upward
        fallback_chain = self._build_fallback_chain(model_name)

        decision = RoutingDecision(
            model_name=model_name,
            complexity_score=round(complexity, 4),
            tier=target_tier,
            strategy=self.strategy,
            fallback_chain=fallback_chain,
        )

        logger.debug(
            "Routing decision: complexity=%.3f, tier=%s, model=%s",
            complexity,
            target_tier,
            model_name,
        )

        return decision

    def _resolve_model(self, target_tier: str) -> str:
        """Find the best available model for a tier, falling back upward."""
        # Try exact tier match
        if target_tier in self._tier_models:
            return self._tier_models[target_tier]

        # Fall back to the next larger tier
        try:
            target_idx = TIER_ORDER.index(target_tier)
        except ValueError:
            target_idx = 0

        for tier in TIER_ORDER[target_idx:]:
            if tier in self._tier_models:
                return self._tier_models[tier]

        # Last resort: fall back downward
        for tier in reversed(TIER_ORDER[:target_idx]):
            if tier in self._tier_models:
                return self._tier_models[tier]

        # Absolute fallback: first model
        return self._ordered_models[0]

    def _build_fallback_chain(self, primary: str) -> list[str]:
        """Build ordered list of fallback models (excluding primary).

        Order: models with higher capability than primary first, then lower.
        """
        primary_idx = self.models[primary].tier_index
        higher = [
            n
            for n in self._ordered_models
            if n != primary and self.models[n].tier_index > primary_idx
        ]
        lower = [
            n
            for n in self._ordered_models
            if n != primary and self.models[n].tier_index < primary_idx
        ]
        # Same-tier models that aren't the primary
        same = [
            n
            for n in self._ordered_models
            if n != primary and self.models[n].tier_index == primary_idx
        ]
        return higher + same + lower

    def route_decomposed(
        self,
        messages: list[dict[str, str]],
    ) -> Union[RoutingDecision, DecomposedRoutingDecision]:
        """Route with optional query decomposition.

        If the query contains multiple independent sub-tasks, returns a
        ``DecomposedRoutingDecision`` with per-task routing.  Otherwise
        returns a standard ``RoutingDecision``.
        """
        from .decomposer import QueryDecomposer

        decomposer = QueryDecomposer()
        decomposition = decomposer.decompose(messages)

        if not decomposition.decomposed:
            return self.route(messages)

        # Route each sub-task independently
        sub_decisions: list[RoutingDecision] = []
        for task in decomposition.tasks:
            sub_messages = [{"role": "user", "content": task.text}]
            # Carry over any system prompt from the original messages
            for msg in messages:
                if msg.get("role") == "system":
                    sub_messages.insert(0, msg)
                    break
            decision = self.route(sub_messages)
            sub_decisions.append(decision)

        return DecomposedRoutingDecision(
            sub_decisions=sub_decisions,
            tasks=decomposition.tasks,
            original_query=decomposition.original_query,
        )

    def get_fallback(self, failed_model: str) -> Optional[str]:
        """Return the next model to try after ``failed_model`` fails.

        Tries models with higher capability first, then lower.
        Returns ``None`` if no fallback is available.
        """
        failed_idx = self.models.get(failed_model, ModelInfo(name="")).tier_index
        # Try higher-tier models first
        for name in self._ordered_models:
            if name != failed_model and self.models[name].tier_index > failed_idx:
                return name
        # Then try same-tier
        for name in self._ordered_models:
            if name != failed_model and self.models[name].tier_index == failed_idx:
                return name
        # Then try lower-tier
        for name in reversed(self._ordered_models):
            if name != failed_model and self.models[name].tier_index < failed_idx:
                return name
        return None


def assign_tiers(
    model_names: list[str],
    thresholds: tuple[float, float] = (0.3, 0.7),
) -> dict[str, ModelInfo]:
    """Auto-assign tiers to an ordered list of model names.

    Assumes ``model_names`` is ordered from smallest to largest.
    Splits into three tiers: fast, balanced, quality.

    For 1 model:  all traffic goes to it (balanced).
    For 2 models: first is fast, second is quality.
    For 3+:       even split across fast/balanced/quality.
    """
    n = len(model_names)
    if n == 0:
        return {}

    if n == 1:
        return {model_names[0]: ModelInfo(name=model_names[0], tier="balanced")}

    if n == 2:
        return {
            model_names[0]: ModelInfo(name=model_names[0], tier="fast"),
            model_names[1]: ModelInfo(name=model_names[1], tier="quality"),
        }

    # 3+ models: divide into thirds
    result: dict[str, ModelInfo] = {}
    fast_end = n // 3
    quality_start = n - (n // 3)

    for i, name in enumerate(model_names):
        if i < fast_end:
            tier = "fast"
        elif i >= quality_start:
            tier = "quality"
        else:
            tier = "balanced"
        result[name] = ModelInfo(name=name, tier=tier)

    return result
