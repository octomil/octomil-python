"""Parse ``@app/{slug}/{capability}`` model references."""

from __future__ import annotations

import re

_APP_REF_PATTERN = re.compile(r"^@app/([^/]+)/([^/]+)$")


def parse_app_ref(model: str) -> tuple[str | None, str | None]:
    """Parse an ``@app/{slug}/{capability}`` model ref.

    Returns ``(app_slug, capability)`` if *model* matches the pattern,
    or ``(None, None)`` otherwise.
    """
    m = _APP_REF_PATTERN.match(model)
    if m:
        return m.group(1), m.group(2)
    return None, None


def is_app_ref(model: str) -> bool:
    """Return ``True`` when *model* looks like an app ref."""
    return model.startswith("@app/")
