"""Replace card-like and government-id-like numbers before text goes anywhere: traces, notes, drafts, Slack."""

from __future__ import annotations

import re

_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact(text: str) -> tuple[str, int]:
    """Returns (text, count of replacements)."""
    count = 0

    def sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        return "[redacted]"

    text = _CARD.sub(sub, text)
    text = _SSN.sub(sub, text)
    return text, count
