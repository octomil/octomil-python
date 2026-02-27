"""
Thin policy-based query routing client.

Routes incoming queries to the most appropriate model tier by fetching
a routing policy from the Octomil server and applying simple threshold
comparisons locally.  Complex scoring logic lives server-side.

**Online mode**: POST /api/v1/route/query for full server-side routing.
**Offline mode**: Apply cached policy with simple word-count / keyword
thresholds.  Falls back to an embedded default policy on first use.

Backward-compatible public API:
- ``ModelInfo``, ``RoutingDecision``, ``DecomposedRoutingDecision``,
  ``DeterministicResult`` dataclasses
- ``QueryRouter`` class with ``route()``, ``route_decomposed()``,
  ``get_fallback()``
- ``check_deterministic()`` and ``assign_tiers()`` module-level functions
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

TIER_ORDER: list[str] = ["fast", "balanced", "quality"]


@dataclass
class ModelInfo:
    """Metadata about a loaded model used for routing decisions."""

    name: str
    tier: str = "balanced"
    param_b: float = 0.0
    loaded: bool = True

    @property
    def tier_index(self) -> int:
        """Numeric rank of the tier (higher = more capable)."""
        try:
            return TIER_ORDER.index(self.tier)
        except ValueError:
            return 1


# ---------------------------------------------------------------------------
# Deterministic result
# ---------------------------------------------------------------------------


@dataclass
class DeterministicResult:
    """Result from a deterministic (no-model) computation."""

    answer: str
    method: str
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
    """Routing decision for a decomposed multi-task query."""

    sub_decisions: list[RoutingDecision]
    tasks: list  # list[SubTask] — typed loosely to avoid circular import
    original_query: str
    decomposed: bool = True


# ---------------------------------------------------------------------------
# Routing policy (fetched from server, cached locally)
# ---------------------------------------------------------------------------

_MATH_PATTERN = re.compile(
    r"^[\s\d+\-*/^().%]+$"
    r"|[\d]+\s*[+\-*/^%]\s*[\d]"
    r"|\b(sqrt|abs|log|sin|cos|tan|ceil|floor)\s*\("
    r"|[\d.]+\s*%\s*of\s+[\d.]+",
)

_DEFAULT_POLICY: dict[str, Any] = {
    "version": 1,
    "thresholds": {"fast_max_words": 10, "quality_min_words": 50},
    "complex_indicators": [
        "implement",
        "refactor",
        "debug",
        "analyze",
        "compare",
        "step by step",
        "prove",
        "derive",
        "calculate",
        "write a story",
        "write a essay",
        "write a report",
        "algorithm",
        "kubernetes",
        "docker",
        "neural network",
        "transformer",
    ],
    "deterministic_enabled": True,
    "ttl_seconds": 3600,
}


@dataclass
class RoutingPolicy:
    """Cached routing policy from the server."""

    version: int
    fast_max_words: int
    quality_min_words: int
    complex_indicators: list[str]
    deterministic_enabled: bool
    ttl_seconds: int
    fetched_at: float = 0.0
    etag: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoutingPolicy:
        thresholds = d.get("thresholds", {})
        return cls(
            version=d.get("version", 1),
            fast_max_words=thresholds.get("fast_max_words", 10),
            quality_min_words=thresholds.get("quality_min_words", 50),
            complex_indicators=d.get("complex_indicators", []),
            deterministic_enabled=d.get("deterministic_enabled", True),
            ttl_seconds=d.get("ttl_seconds", 3600),
            fetched_at=d.get("fetched_at", 0.0),
            etag=d.get("etag", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "thresholds": {
                "fast_max_words": self.fast_max_words,
                "quality_min_words": self.quality_min_words,
            },
            "complex_indicators": self.complex_indicators,
            "deterministic_enabled": self.deterministic_enabled,
            "ttl_seconds": self.ttl_seconds,
            "fetched_at": self.fetched_at,
            "etag": self.etag,
        }

    @property
    def is_expired(self) -> bool:
        if self.fetched_at == 0.0:
            return True
        return (time.time() - self.fetched_at) > self.ttl_seconds


# ---------------------------------------------------------------------------
# Policy client — fetches and caches policy from server
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(os.environ.get("OCTOMIL_CACHE_DIR", Path.home() / ".octomil"))
_POLICY_FILE = "routing_policy.json"


class PolicyClient:
    """Fetches routing policy from the server and caches to disk."""

    def __init__(
        self,
        api_base: str = "https://api.octomil.com/api/v1",
        api_key: str = "",
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self._policy: Optional[RoutingPolicy] = None

    def get_policy(self) -> RoutingPolicy:
        """Return the current policy, refreshing from server if expired."""
        if self._policy is not None and not self._policy.is_expired:
            return self._policy

        # Try loading from disk cache
        if self._policy is None:
            self._policy = self._load_from_disk()

        # If still valid after loading from disk, return it
        if self._policy is not None and not self._policy.is_expired:
            return self._policy

        # Try fetching from server
        fetched = self._fetch_from_server()
        if fetched is not None:
            self._policy = fetched
            self._save_to_disk(fetched)
            return self._policy

        # Use disk cache even if expired, or fall back to defaults
        if self._policy is not None:
            logger.debug("Using expired cached policy (server unreachable)")
            return self._policy

        logger.debug("Using default embedded policy (no cache, server unreachable)")
        self._policy = RoutingPolicy.from_dict(_DEFAULT_POLICY)
        return self._policy

    def _fetch_from_server(self) -> Optional[RoutingPolicy]:
        """Fetch policy from GET /api/v1/route/policy."""
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not available — cannot fetch routing policy")
            return None

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self._policy and self._policy.etag:
            headers["If-None-Match"] = self._policy.etag

        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(
                    f"{self.api_base}/route/policy",
                    headers=headers,
                )

            if resp.status_code == 304:
                # Not modified — refresh TTL on existing policy
                if self._policy:
                    self._policy.fetched_at = time.time()
                    self._save_to_disk(self._policy)
                return self._policy

            if resp.status_code != 200:
                logger.debug("Policy fetch returned HTTP %d", resp.status_code)
                return None

            data = resp.json()

            # Extract TTL from Cache-Control header if present
            cc = resp.headers.get("cache-control", "")
            ttl = data.get("ttl_seconds", 3600)
            if "max-age=" in cc:
                try:
                    ttl = int(cc.split("max-age=")[1].split(",")[0].strip())
                except (ValueError, IndexError):
                    pass
            data["ttl_seconds"] = ttl

            etag = resp.headers.get("etag", "")
            data["etag"] = etag
            data["fetched_at"] = time.time()

            return RoutingPolicy.from_dict(data)

        except Exception:
            logger.debug("Failed to fetch routing policy from server", exc_info=True)
            return None

    def _load_from_disk(self) -> Optional[RoutingPolicy]:
        """Load cached policy from disk."""
        path = _CACHE_DIR / _POLICY_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return RoutingPolicy.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def _save_to_disk(self, policy: RoutingPolicy) -> None:
        """Persist policy to disk cache."""
        path = _CACHE_DIR / _POLICY_FILE
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(policy.to_dict(), indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.debug("Failed to write policy cache to %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Deterministic detection (thin — delegates to server when online)
# ---------------------------------------------------------------------------


def check_deterministic(query: str) -> Optional[DeterministicResult]:
    """Check if a query looks like pure arithmetic.

    In online mode the server handles actual evaluation.  This function
    only detects obvious math patterns and returns ``None`` for anything
    that needs a model.  Returns a ``DeterministicResult`` stub when a
    math pattern is detected, suitable for the server to evaluate.
    """
    text = query.strip().rstrip("?.")

    # Strip common prefixes
    for prefix in (
        "what is ",
        "what's ",
        "calculate ",
        "compute ",
        "evaluate ",
        "how much is ",
        "solve ",
        "find ",
    ):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break

    if not text:
        return None

    # Quick check: if remaining text is purely numeric/operator characters
    # or contains a known math function call, treat as arithmetic
    if _MATH_PATTERN.search(text):
        # Reject if there are alphabetic chars that aren't math function names
        check = text
        for fname in (
            "sqrt",
            "abs",
            "round",
            "ceil",
            "floor",
            "log2",
            "log10",
            "log",
            "sin",
            "cos",
            "tan",
            "pi",
        ):
            check = check.replace(fname, "")
        check = re.sub(r"\bof\b", "", check, flags=re.IGNORECASE)
        if re.search(r"[a-zA-Z_]", check):
            return None

        # Basic safe evaluation for simple expressions
        result = _safe_eval(text)
        if result is not None:
            return DeterministicResult(
                answer=_format_number(result),
                method="arithmetic",
                confidence=1.0,
            )

    return None


def _safe_eval(expr: str) -> Optional[float]:
    """Evaluate a simple arithmetic expression safely via ast."""
    import ast
    import math
    import operator

    # Normalise: ^ to **, implicit multiplication
    expr = re.sub(r"(\d)\s*\^\s*(\d)", r"\1**\2", expr)
    expr = re.sub(r"(\d)\s*\(", r"\1*(", expr)

    # Handle percentage: "N% of M"
    pct = re.match(r"^([\d.]+)\s*%\s*(?:of)\s+([\d.]+)$", expr, re.IGNORECASE)
    if pct:
        return (float(pct.group(1)) / 100.0) * float(pct.group(2))

    _SAFE_NODES = (
        ast.Expression,
        ast.Constant,
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
    if hasattr(ast, "Num"):
        _SAFE_NODES = _SAFE_NODES + (ast.Num,)

    _FUNCS: dict[str, Any] = {
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
    }
    _CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e}

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.UnaryOp):
            val = _eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError
        if isinstance(node, ast.BinOp):
            left, right = _eval_node(node.left), _eval_node(node.right)
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }
            fn = ops.get(type(node.op))
            if fn is None:
                raise ValueError
            if isinstance(node.op, ast.Pow) and abs(right) > 10000:
                raise ValueError("Exponent too large")
            return fn(left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _FUNCS or node.keywords:
                raise ValueError
            return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            return None

    try:
        return _eval_node(tree)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, ArithmeticError):
        return None


def _format_number(value: float) -> str:
    """Format a numeric result for display."""
    import math as _math

    if value == float("inf") or value == float("-inf") or _math.isnan(value):
        return str(value)
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------


class QueryRouter:
    """Routes queries to the best model based on server-side policy.

    Parameters
    ----------
    models:
        Dict mapping model name to ``ModelInfo``.
    strategy:
        Routing strategy. Currently only ``"complexity"`` is supported.
    thresholds:
        Two-element tuple ``(low, high)`` defining tier boundaries.
    enable_deterministic:
        If ``True``, check for tier-0 deterministic answers.
    api_base:
        Server URL for policy fetching and server-side routing.
    api_key:
        API key for server authentication.
    """

    def __init__(
        self,
        models: dict[str, ModelInfo],
        strategy: str = "complexity",
        thresholds: tuple[float, float] = (0.3, 0.7),
        enable_deterministic: bool = True,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
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

        self._tier_models: dict[str, str] = {}
        for name, info in models.items():
            if info.tier not in self._tier_models:
                self._tier_models[info.tier] = name

        self._ordered_models: list[str] = sorted(
            models.keys(),
            key=lambda n: models[n].tier_index,
        )

        _base = api_base or os.environ.get(
            "OCTOMIL_API_BASE", "https://api.octomil.com/api/v1"
        )
        _key = api_key or os.environ.get("OCTOMIL_API_KEY", "")
        self._policy_client = PolicyClient(api_base=_base, api_key=_key)

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
        tier, and fallback chain.
        """
        user_text = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_text = msg.get("content", "")

        # Tier 0: deterministic
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

        # Apply policy-based routing
        policy = self._policy_client.get_policy()
        target_tier, score = self._apply_policy(user_text, policy)

        model_name = self._resolve_model(target_tier)
        fallback_chain = self._build_fallback_chain(model_name)

        decision = RoutingDecision(
            model_name=model_name,
            complexity_score=round(score, 4),
            tier=target_tier,
            strategy=self.strategy,
            fallback_chain=fallback_chain,
        )

        logger.debug(
            "Routing decision: score=%.3f, tier=%s, model=%s",
            score,
            target_tier,
            model_name,
        )

        return decision

    def _apply_policy(self, query: str, policy: RoutingPolicy) -> tuple[str, float]:
        """Apply cached policy to classify query into a tier.

        Returns (tier, approximate_complexity_score).
        """
        word_count = len(query.split())
        query_lower = query.lower()

        has_complex = any(kw in query_lower for kw in policy.complex_indicators)

        if word_count < policy.fast_max_words and not has_complex:
            return "fast", round(word_count / 100.0, 4)
        elif word_count > policy.quality_min_words or has_complex:
            return "quality", round(min(word_count / 100.0 + 0.5, 1.0), 4)
        else:
            return "balanced", round(word_count / 100.0 + 0.25, 4)

    def _resolve_model(self, target_tier: str) -> str:
        """Find the best available model for a tier, falling back upward."""
        if target_tier in self._tier_models:
            return self._tier_models[target_tier]

        try:
            target_idx = TIER_ORDER.index(target_tier)
        except ValueError:
            target_idx = 0

        for tier in TIER_ORDER[target_idx:]:
            if tier in self._tier_models:
                return self._tier_models[tier]

        for tier in reversed(TIER_ORDER[:target_idx]):
            if tier in self._tier_models:
                return self._tier_models[tier]

        return self._ordered_models[0]

    def _build_fallback_chain(self, primary: str) -> list[str]:
        """Build ordered list of fallback models (excluding primary)."""
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

        sub_decisions: list[RoutingDecision] = []
        for task in decomposition.tasks:
            sub_messages = [{"role": "user", "content": task.text}]
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

        Returns ``None`` if no fallback is available.
        """
        failed_idx = self.models.get(failed_model, ModelInfo(name="")).tier_index

        for name in self._ordered_models:
            if name != failed_model and self.models[name].tier_index > failed_idx:
                return name
        for name in self._ordered_models:
            if name != failed_model and self.models[name].tier_index == failed_idx:
                return name
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
