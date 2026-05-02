"""Lightweight text segmentation for streaming-capability verification.

Used by the kernel's ``_verify_capability`` to decide whether an
observed multi-chunk run is plausibly *progressive* (sub-sentence)
or merely *sentence-boundary* (one chunk per sentence). Without this
distinction, a multi-sentence input that yielded one chunk per
sentence would falsely verify a backend's ``progressive`` claim.

This is deliberately simple — a yes/no on "is multi-sentence,"
matched to the same regex sherpa-onnx's ``max_num_sentences=1``
would split on (period / exclamation / question mark + whitespace
+ non-space). Not an ICU sentence segmenter; we just need a
conservative count for the kernel's verification gate.

Sherpa's engine.py re-exports :func:`count_sentences` so older
callers (and the engine's own ``streaming_capability``) keep
working without an extra import path.
"""

from __future__ import annotations

import re

# A sentence terminator (``. ! ? 。 ！ ？``, possibly repeated) followed by
# whitespace and a non-space character. Without the trailing non-space
# requirement, ``"Hello."`` would report 2 because the period would match
# even at end-of-string.
_SENTENCE_TERMINATORS = re.compile(r"[.!?。！？]+\s+\S")


def count_sentences(text: str) -> int:
    """Return a conservative sentence count for ``text``.

    Returns ``0`` for empty / whitespace-only input. Returns ``1``
    for any non-empty text with no internal terminator+whitespace
    boundary (single-sentence — including unterminated fragments).
    Returns ``1 + len(matches)`` otherwise.
    """
    if not text or not text.strip():
        return 0
    matches = _SENTENCE_TERMINATORS.findall(text)
    return 1 + len(matches)


__all__ = ["count_sentences"]
