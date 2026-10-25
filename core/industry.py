"""Canonical labels for the Kenneth French 10-industry portfolios."""

from __future__ import annotations

import re


_ALIASES = {
    "NoDur": ("nodur", "consumer nondurables", "consumer non durables", "nondurables"),
    "Durbl": ("durbl", "consumer durables", "durables", "durable"),
    "Manuf": ("manuf", "manufacturing"),
    "Enrgy": ("enrgy", "energy"),
    "HiTec": ("hitec", "high tech", "high technology", "technology"),
    "Telcm": ("telcm", "telecommunications", "telecom"),
    "Shops": ("shops", "retail", "shops retail"),
    "Hlth": ("hlth", "health", "healthcare"),
    "Utils": ("utils", "utilities"),
    "Other": ("other",),
}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


_LOOKUP = {
    _key(alias): canonical
    for canonical, aliases in _ALIASES.items()
    for alias in (canonical, *aliases)
}


def canonical_industry_name(value) -> str | None:
    if not isinstance(value, str):
        return None
    key = _key(value)
    # Governed runtime tables use names such as ``durbl_pct`` for percentage
    # return columns.  The suffix changes representation, not industry identity.
    if key.endswith("pct"):
        key = key[:-3]
    return _LOOKUP.get(key)


def canonical_industry_mapping(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    normalized = {}
    for label, item in value.items():
        canonical = canonical_industry_name(label)
        if canonical is None or canonical in normalized:
            return None
        normalized[canonical] = item
    return normalized


# Kenneth French's 10-industry "HiTec" (business equipment) definition, as SIC
# ranges. This is the objective membership rule for "technology" companies once
# the SEC universe carries SIC codes instead of hand-assigned sectors.
KEN_FRENCH_10_HITEC_SIC_RANGES = (
    (3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379),
)


def is_hitec_sic(sic) -> bool:
    """True when a SIC code falls inside Ken French's 10-industry HiTec group."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return False
    return any(low <= code <= high for low, high in KEN_FRENCH_10_HITEC_SIC_RANGES)
