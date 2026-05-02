"""Public ``octomil.audio.text_normalize`` contract.

The kernel applies normalization automatically before backend
dispatch, so consumers never need to call these directly. Tests
here pin the rule set so the espeak-compat profile stays
predictable across releases.
"""

from __future__ import annotations

import pytest

from octomil.audio.text_normalize import (
    PROFILE_ESPEAK_COMPAT,
    PROFILE_NONE,
    available_profiles,
    for_kokoro,
    normalize_for_profile,
)

# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------


def test_available_profiles_lists_known_profiles():
    profiles = available_profiles()
    assert PROFILE_NONE in profiles
    assert PROFILE_ESPEAK_COMPAT in profiles


def test_profile_constants_are_stable_strings():
    """Consumers may import these constants — pin the literal values
    so a rename would be a deliberate breaking change."""
    assert PROFILE_NONE == "none"
    assert PROFILE_ESPEAK_COMPAT == "espeak_compat"


# ---------------------------------------------------------------------------
# none / unknown profiles
# ---------------------------------------------------------------------------


def test_profile_none_passes_text_through_unchanged():
    text = "I owe him $1200, Mr. Smith said."
    assert normalize_for_profile(text, PROFILE_NONE) == text


def test_unknown_profile_passes_through_unchanged():
    """Forward compat: a future backend may declare a profile name
    this SDK version doesn't know. Falling through to no-op is the
    conservative posture — better silent passthrough than crash."""
    text = "I owe him $1200."
    assert normalize_for_profile(text, "future_codec_lm_v2") == text


def test_empty_text_passes_through():
    assert normalize_for_profile("", PROFILE_ESPEAK_COMPAT) == ""


# ---------------------------------------------------------------------------
# espeak_compat — currency reordering (the Eternum bug)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("I owe him $1200.", "I owe him 1200 dollars."),
        ("$50", "50 dollars"),
        ("It costs $1,234.56 total.", "It costs 1234.56 dollars total."),
        ("£42 fee", "42 pounds fee"),
        ("¥1000 in change", "1000 yen in change"),
        ("Refunds: -$100", "Refunds: negative 100 dollars"),
        # Multiple in one sentence
        ("$5 to $50", "5 dollars to 50 dollars"),
    ],
)
def test_currency_reordering(raw, expected):
    assert for_kokoro(raw) == expected


def test_currency_does_not_touch_bare_numbers():
    """Bare numbers without a currency symbol are left alone —
    espeak handles them correctly via its own number expansion."""
    text = "There were 1200 people, and 1.5M attended overall."
    assert for_kokoro(text) == text


# ---------------------------------------------------------------------------
# espeak_compat — currency P2: no unit-word duplication
# ---------------------------------------------------------------------------
#
# When the author already wrote a unit word after the amount, the
# normalizer must NOT append another. Pre-fix, ``$1200 dollars`` →
# ``1200 dollars dollars`` and ``€500 euros`` → ``500 euros euros``
# (the original test even pinned the bug). Reviewer P2 on 4.16.0.


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Pluralized — most common author pattern.
        ("$1200 dollars", "1200 dollars"),
        ("€500 euros wasted", "500 euros wasted"),
        ("£42 pounds remaining", "42 pounds remaining"),
        ("¥1000 yen in change", "1000 yen in change"),
        # Singular — author wrote unit in singular form.
        ("$1 dollar", "1 dollar"),
        ("£1 pound", "1 pound"),
        # ISO codes — author used the formal form.
        ("$50 USD", "50 USD"),
        ("€100 EUR", "100 EUR"),
        # Negative + already-unit.
        ("Refunds: -$100 dollars", "Refunds: negative 100 dollars"),
        # Mixed: one with unit, one without.
        ("$5 to $50 dollars", "5 dollars to 50 dollars"),
        # No double-expansion across the whole sentence.
        ("I have $1200 dollars and $50 in change.", "I have 1200 dollars and 50 dollars in change."),
    ],
)
def test_currency_does_not_duplicate_existing_unit(raw, expected):
    assert for_kokoro(raw) == expected


def test_currency_does_not_swallow_non_unit_words():
    """The trailing-unit consume must only fire on actual currency
    unit words. ``$50 worth`` must NOT eat ``worth`` — ``worth`` is
    not a unit and the regex's alternation must not match it."""
    assert for_kokoro("$50 worth of gear") == "50 dollars worth of gear"
    assert for_kokoro("$1200 reasons") == "1200 dollars reasons"


def test_currency_unit_match_respects_word_boundary():
    """``$50 dollarized`` (made-up but plausible) must NOT match
    ``dollar`` as a trailing unit and leave a stray ``ized`` —
    the alternation requires a word boundary so partial-word
    matches don't eat real text."""
    assert for_kokoro("$50 dollarized") == "50 dollars dollarized"


# ---------------------------------------------------------------------------
# espeak_compat — percent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("50% off today.", "50 percent off today."),
        ("From 25% to 75% capacity", "From 25 percent to 75 percent capacity"),
        ("Tip: 18.5%", "Tip: 18.5 percent"),
    ],
)
def test_percent_expands(raw, expected):
    assert for_kokoro(raw) == expected


# ---------------------------------------------------------------------------
# espeak_compat — degree symbols
# ---------------------------------------------------------------------------


def test_degree_fahrenheit_expands():
    assert for_kokoro("It's 72°F outside.") == "It's 72 degrees Fahrenheit outside."


def test_degree_celsius_expands():
    assert for_kokoro("Boiling at 100°C.") == "Boiling at 100 degrees Celsius."


def test_bare_degree_symbol_left_alone():
    """Bare ``°`` without F/C is ambiguous (could be angles).
    Conservative: leave it alone."""
    assert for_kokoro("Turn 90° clockwise") == "Turn 90° clockwise"


# ---------------------------------------------------------------------------
# espeak_compat — abbreviations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Mr. Smith", "Mister Smith"),
        ("Mrs. Smith", "Misses Smith"),
        ("Ms. Davis", "Miss Davis"),
        ("Dr. Mendez", "Doctor Mendez"),
        ("Mt. Everest", "Mount Everest"),
        ("Lakers vs. Celtics", "Lakers versus Celtics"),
        ("apples, oranges, etc.", "apples, oranges, etcetera"),
        # Mrs. must NOT be caught by Mr. — order matters in the
        # rule list. Pinned by the dual ``Mrs. Smith`` /
        # ``Mr. Smith`` parametrization above plus this combined
        # case.
        ("Mr. and Mrs. Smith", "Mister and Misses Smith"),
    ],
)
def test_abbreviation_expansion(raw, expected):
    assert for_kokoro(raw) == expected


def test_st_abbreviation_intentionally_left_alone():
    """``St.`` is ambiguous between Saint and Street. Reviewer P2
    on 4.16.0: ``"Meet me on St. John St."`` was normalizing to
    ``"Meet me on Saint John Saint"``. Conservative posture: leave
    ``St.`` alone entirely. Espeak reads it as "saint" approximately
    correctly without our help, and prose-wrecking false positives
    on street addresses are NOT acceptable in an automatic
    transform.

    A future context-sensitive expansion (``\\bSt\\. [A-Z]`` only at
    title position) could revisit this safely; until then, no
    expansion."""
    assert for_kokoro("Meet me on St. John St.") == "Meet me on St. John St."
    assert for_kokoro("St. Nicholas Day") == "St. Nicholas Day"
    assert for_kokoro("123 Main St.") == "123 Main St."


# ---------------------------------------------------------------------------
# Eternum-style integration sample
# ---------------------------------------------------------------------------


def test_eternum_dialogue_sample():
    """The exact phrase Eternum reported plus a richer dialogue
    sample exercising multiple rules at once."""
    raw = "Mr. Smith said he owes me $1200, but Dr. Lee paid 50% upfront."
    assert for_kokoro(raw) == ("Mister Smith said he owes me 1200 dollars, " "but Doctor Lee paid 50 percent upfront.")


def test_for_kokoro_is_idempotent():
    """Running normalization twice must produce the same string —
    a backend that re-normalizes (or a consumer that pre-normalizes
    AND leaves auto on) shouldn't double-expand."""
    raw = "I owe Mr. Smith $1200."
    once = for_kokoro(raw)
    twice = for_kokoro(once)
    assert once == twice
