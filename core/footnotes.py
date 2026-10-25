"""Validated extraction helpers for SEC transaction-price footnotes."""

from __future__ import annotations

import re


class PriceRangeParseError(ValueError):
    pass


_NUMBER = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
_MONEY = rf"\$?\s*{_NUMBER}"
_PATTERNS = (
    re.compile(rf"prices?\s+ranging\s+from\s+{_MONEY}\s+to\s+{_MONEY}", re.I),
    re.compile(rf"prices?.{{0,80}}?between\s+{_MONEY}\s+and\s+{_MONEY}", re.I),
    re.compile(rf"(?:price\s+)?range.{{0,40}}?{_MONEY}\s+(?:to|through)\s+{_MONEY}", re.I),
    re.compile(rf"(?:price\s+)?range.{{0,40}}?{_MONEY}\s*[-–—−]\s*\$?\s*{_NUMBER}", re.I),
)


def extract_price_range(text: str) -> tuple[float, float]:
    """Extract and order a two-sided execution-price range from a footnote.

    Whitespace and common Unicode dash variants are normalized before matching.
    A candidate that does not match a supported, price-anchored template is
    rejected rather than silently admitted with guessed bounds.
    """
    if not isinstance(text, str) or not text.strip():
        raise PriceRangeParseError("price footnote is blank")
    normalized = " ".join(text.replace("\u00a0", " ").split())
    for pattern in _PATTERNS:
        match = pattern.search(normalized)
        if match:
            low, high = (float(value.replace(",", "")) for value in match.groups())
            if low > high:
                low, high = high, low
            return low, high
    raise PriceRangeParseError(f"unsupported execution-price range template: {normalized}")
