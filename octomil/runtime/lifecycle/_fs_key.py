"""Shared filesystem-key helper for planner-supplied identifiers.

PrepareManager and FileLock both turn untrusted planner ids into single
path components: PrepareManager into ``<cache>/artifacts/<key>/``, and
FileLock into ``<cache>/.locks/<key>.lock``. They must agree on the key
shape so two layers of the prepare-lifecycle pipeline cannot disagree
about safety; this module is the one place that decides.

Key requirements:

* **Bounded byte length.** NAME_MAX on every common filesystem
  (ext4, APFS, NTFS) is 255 bytes, *not* 255 characters. A naive char-
  count cap admits filenames many times over NAME_MAX once non-ASCII is
  involved (one emoji is up to 4 bytes UTF-8).
* **Windows-safe.** Windows reserves ``< > : " / \\ | ? *`` in filenames.
* **Stable mapping.** Same input → same output, so cache hits are
  reproducible across processes.
* **Disambiguating.** Distinct planner ids that sanitize to the same
  visible name still get distinct keys via a SHA-256 suffix taken over
  the *original* (unmodified) input.
"""

from __future__ import annotations

import hashlib
import re

# ASCII allow-list. Anything outside this set — including all non-ASCII
# characters and Windows-reserved punctuation — is replaced with '_'.
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")

# Visible-portion cap. The full key is ``<visible>-<12-char hash>``;
# 96 + 1 + 12 = 109 byte ASCII payload, well under NAME_MAX (255 bytes)
# even with the consumer's own suffix (e.g. ``.lock``).
DEFAULT_MAX_VISIBLE_CHARS = 96


def safe_filesystem_key(name: str, *, max_visible: int = DEFAULT_MAX_VISIBLE_CHARS) -> str:
    """Return a NAME_MAX-safe, Windows-safe, deterministic key for ``name``.

    The result is always pure ASCII, ``len(result) <= max_visible + 13``
    bytes, and stable across processes. Empty / dot-only inputs collapse
    to ``"id"`` plus the hash suffix so the consumer always has at least
    a 14-character (1 + 1 + 12) component to work with.

    Raises ``ValueError`` only when ``name`` contains a NUL byte —
    callers that already check this can ignore the contract. Every other
    structurally-invalid input (absolute paths, traversal, Windows
    reserved chars, non-UTF-8 surrogates) sanitizes safely.
    """
    if "\x00" in name:
        raise ValueError("filesystem key must not contain NUL bytes")
    sanitized = _SAFE_CHARS.sub("_", name).strip("_.")
    if sanitized in ("", ".", ".."):
        sanitized = "id"
    if len(sanitized) > max_visible:
        sanitized = sanitized[:max_visible].rstrip("_.")
        if not sanitized:
            sanitized = "id"
    digest_prefix = hashlib.sha256(name.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    return f"{sanitized}-{digest_prefix}"
