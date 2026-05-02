"""Backend-aware text normalization for TTS dispatch.

The kernel applies normalization automatically before handing text
to the synthesis backend. Each backend declares a profile name via
``text_normalization_profile()``; the kernel calls
:func:`normalize_for_profile` between voice validation and backend
dispatch. Consumers do NOT need to call anything — that's the
point. Asking every consumer to remember a ``for_kokoro(text)``
helper is the same anti-pattern as making them call ``warmup()``
to get reasonable cold-start latency.

Profiles
========

``espeak_compat``
    For backends driven by espeak-ng (sherpa-onnx Kokoro / Piper).
    Fixes the known espeak misreads:

      - Currency reordering: ``$1200`` → ``1200 dollars``,
        ``€500`` → ``500 euros``, ``£42`` → ``42 pounds``,
        ``¥1000`` → ``1000 yen``. Decimals + thousands separators
        normalize: ``$1,234.56`` → ``1234.56 dollars``.
      - Percentage: ``50%`` → ``50 percent`` (espeak version
        coverage is uneven).
      - Common abbreviations: ``Mr.`` / ``Mrs.`` / ``Ms.`` / ``Dr.``
        / ``St.`` / ``Mt.`` / ``vs.`` / ``etc.`` expand to their
        spoken forms.
      - Degree symbols: ``°F`` / ``°C`` → "degrees Fahrenheit/Celsius".

``none``
    No-op pass-through. For backends with their own LM-based
    text frontend (Parler-TTS, future codec-LM TTS) where any
    SDK-side normalization would be a regression.

What we explicitly DO NOT touch
================================

  - Bare numbers (``12``, ``1200``, ``1.5M``). espeak handles these
    correctly via its own number expansion. Touching them risks
    introducing locale-specific bugs.
  - Names / proper nouns. Too high false-positive risk.
  - Sentence boundaries. The kernel's ``_count_sentences`` is
    downstream of normalization; rewriting sentence structure
    here would break the progressive-TTS verification gate.
  - Pronunciation overrides ("read" vs "read"). That's a
    phonemizer concern, not a text frontend concern.

Privacy posture
===============

Normalized text passes through the dispatch path and reaches the
backend. It does NOT enter the privacy-gated TTS observability
events (``tts.stream.*``) — those are content-free per the
contract. Normalization is a pure string transformation; it
never logs the text.
"""

from __future__ import annotations

import re

__all__ = [
    "normalize_for_profile",
    "for_kokoro",
    "available_profiles",
    "PROFILE_NONE",
    "PROFILE_ESPEAK_COMPAT",
]


PROFILE_NONE = "none"
PROFILE_ESPEAK_COMPAT = "espeak_compat"


def available_profiles() -> tuple[str, ...]:
    """Return the registered profile names. Used by the kernel's
    opt-out validation and by tests pinning the public surface."""
    return (PROFILE_NONE, PROFILE_ESPEAK_COMPAT)


def normalize_for_profile(text: str, profile: str) -> str:
    """Apply the named profile's normalization rules to ``text``.

    Returns ``text`` unchanged when ``profile`` is :data:`PROFILE_NONE`
    or unrecognized — unrecognized profiles fall through to no-op
    so a future backend declaring a profile this SDK version doesn't
    know about still produces audio rather than raising.
    """
    if not text or profile == PROFILE_NONE:
        return text
    if profile == PROFILE_ESPEAK_COMPAT:
        return _normalize_espeak_compat(text)
    return text


def for_kokoro(text: str) -> str:
    """Convenience helper for consumers that want to display the
    normalized text in a UI before calling ``audio.speech.stream``.
    Equivalent to ``normalize_for_profile(text, PROFILE_ESPEAK_COMPAT)``.

    The dispatch path applies this automatically — consumers do NOT
    need to call this before every TTS call. Use it only when you
    need the post-normalization string yourself (subtitle rendering,
    log review, etc.).
    """
    return normalize_for_profile(text, PROFILE_ESPEAK_COMPAT)


# ---------------------------------------------------------------------------
# espeak_compat profile
# ---------------------------------------------------------------------------


# Currency: prefix-symbol form. Match optional negative sign, the
# symbol, and a number (optional commas, optional decimal). Reorder
# to "<amount> <unit>" because espeak reads the symbol literally in
# source order — "dollar 1200" instead of "1200 dollars". Comma
# thousands separators are stripped because espeak reads bare
# digit groups correctly but mis-parses comma-separated forms in
# some locales.
_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "dollars",
    "€": "euros",
    "£": "pounds",
    "¥": "yen",
}
# Match digits + optional comma-grouped thousands + optional decimal.
# Each comma must be FOLLOWED by exactly three digits — that's the
# thousands-separator grammar — so a trailing ``$1200,`` (comma is
# sentence punctuation, not a thousands separator) leaves the comma
# in the source text. ``\d+`` first consumes greedy leading digits;
# the ``(?:,\d{3})*`` only fires for genuine thousands groups.
_CURRENCY_PATTERN = re.compile(
    r"(?P<neg>-)?(?P<sym>[$€£¥])" r"(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)",
)


def _replace_currency(match: re.Match[str]) -> str:
    sym = match.group("sym")
    amount = match.group("amount").replace(",", "")
    unit = _CURRENCY_SYMBOLS.get(sym, sym)
    if match.group("neg"):
        return f"negative {amount} {unit}"
    return f"{amount} {unit}"


# Percent: ``50%`` → ``50 percent``. Match a number then a literal
# %. Bare "%" without a leading number is left alone (rare; can
# legitimately mean the modulo operator in a math-related read).
_PERCENT_PATTERN = re.compile(r"(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)\s*%")


def _replace_percent(match: re.Match[str]) -> str:
    amount = match.group("amount").replace(",", "")
    return f"{amount} percent"


# Degree symbols. ``°F`` and ``°C`` are the two espeak misreads;
# bare "°" without F/C is left alone (could be angles).
_DEGREE_PATTERN = re.compile(r"°\s*([FC])")


def _replace_degree(match: re.Match[str]) -> str:
    unit = match.group(1)
    label = "Fahrenheit" if unit == "F" else "Celsius"
    return f" degrees {label}"


# Abbreviations: word-boundary match the abbrev followed by a
# period; expand to spoken form. Order matters — longer matches
# first so ``Mrs.`` doesn't get caught by ``Mr.``.
_ABBREVIATIONS: tuple[tuple[str, str], ...] = (
    ("Mrs.", "Misses"),
    ("Ms.", "Miss"),
    ("Mr.", "Mister"),
    ("Dr.", "Doctor"),
    ("St.", "Saint"),  # ambiguous w/ "Street"; "Saint" is more common
    ("Mt.", "Mount"),
    ("vs.", "versus"),
    ("etc.", "etcetera"),
)


def _replace_abbreviations(text: str) -> str:
    for abbrev, expansion in _ABBREVIATIONS:
        # Use word boundary on the leading char only — trailing "." is
        # part of the literal match; we don't want \b after the period
        # because ``\b`` doesn't fire between "." and a space.
        pattern = re.compile(r"\b" + re.escape(abbrev))
        text = pattern.sub(expansion, text)
    return text


def _normalize_espeak_compat(text: str) -> str:
    """Apply every espeak-compat rule to ``text``.

    Rules are applied in a deterministic order. Currency runs
    before percent so ``$50%`` (rare but possible in pricing copy)
    becomes ``50 dollars percent`` rather than ``$50 percent``,
    and the percent rule then leaves it alone. Abbreviations run
    last so ``Dr.`` immediately followed by a number doesn't get
    eaten by the currency regex.
    """
    text = _CURRENCY_PATTERN.sub(_replace_currency, text)
    text = _PERCENT_PATTERN.sub(_replace_percent, text)
    text = _DEGREE_PATTERN.sub(_replace_degree, text)
    text = _replace_abbreviations(text)
    return text
