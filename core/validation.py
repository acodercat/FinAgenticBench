from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
import math
from numbers import Real
from typing import Any

import pandas as pd
import numpy as np


@dataclass
class ValidatorResult:
    success: bool
    message: str
    variables_not_set: bool = False
    variable_results: dict[str, dict[str, Any]] | None = None


def turn_validator(check):
    """Adapt an outputs-only check to the turn validator contract.

    A turn validator is called as `(response, runtime, turn)`, which is what a
    case needs when it must inspect the runtime itself. Most cases only compare
    the values the turn asked the agent to store, so they write that comparison
    against a plain dict and wrap it here; the retrieval and the turn's `stores`
    list stay in one place instead of being repeated per case.

    `response` is deliberately not forwarded: a validator must not read prose to
    decide whether a number is right (CASE_DESIGN_STANDARD P2). A case that
    genuinely needs the response or the live runtime registers its own
    three-argument callable in `validators` instead of calling this.
    """
    def validate_turn(response, runtime, turn):
        names = turn.stores if turn and turn.stores else []
        return check({name: runtime.retrieve(name) for name in names})

    return validate_turn


def finite_number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


def numeric_equal(actual, expected, *, decimals: int) -> bool:
    if not finite_number(actual) or not finite_number(expected):
        return False
    if decimals < 0:
        raise ValueError("decimals must be non-negative")
    # Counts and other zero-decimal outputs are deterministic in the frozen data.
    # Do not let a fractional near miss pass as display-rounding variance.
    tolerance = 0.0 if decimals == 0 else 0.6 * 10 ** (-decimals)
    return abs(float(actual) - float(expected)) <= tolerance


def _reported(value: float, decimals: int) -> Decimal:
    """``value`` to ``decimals`` places, half away from zero, on its shortest repr.

    Decimal arithmetic on ``repr`` rather than ``round()``, which works on the
    binary float and reports 2.675 as 2.67.
    """
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def reports_as(actual, expected, *, decimals: int) -> bool:
    """Whether ``actual`` is ``expected`` reported to ``decimals`` places.

    True iff ``actual`` rounds, half away from zero, to the same figure as the
    unrounded ``expected`` and lies within half a unit of its last place. The
    correctly rounded and the unrounded answer both pass; a value from another
    method passes only if it would be reported identically. ``numeric_equal``'s
    band is centred on the unrounded truth and admits anything within
    ``0.6 * 10**-decimals``, including a different method's figure that rounds
    one unit away. Zero-decimal outputs keep exact equality, as in
    ``numeric_equal``.
    """
    if not finite_number(actual) or not finite_number(expected):
        return False
    if decimals < 0:
        raise ValueError("decimals must be non-negative")
    if decimals == 0:
        return float(actual) == float(expected)
    gap = abs(Decimal(repr(float(actual))) - Decimal(repr(float(expected))))
    if gap > Decimal(5).scaleb(-decimals - 1):
        return False
    return _reported(actual, decimals) == _reported(expected, decimals)


def datetime_equal(actual, expected) -> bool:
    """Compare timestamp-like values by instant rather than ISO formatting."""
    if actual is None or expected is None or isinstance(actual, bool):
        return False
    try:
        actual_timestamp = pd.to_datetime(actual, utc=True)
        expected_timestamp = pd.to_datetime(expected, utc=True)
    except (TypeError, ValueError, OverflowError):
        return False
    if pd.isna(actual_timestamp) or pd.isna(expected_timestamp):
        return False
    return bool(actual_timestamp == expected_timestamp)


def boolean_equal(actual, expected: bool) -> bool:
    """Accept Python and NumPy booleans without accepting truthy strings/ints."""
    return isinstance(actual, (bool, np.bool_)) and bool(actual) is expected


def select_extreme_key(values: dict, *, largest: bool = True):
    """Select an extremum with deterministic natural-key tie breaking."""
    if not values:
        raise ValueError("cannot select an extremum from an empty mapping")
    extreme = (max if largest else min)(values.values())
    tied = [key for key, value in values.items() if value == extreme]
    return min(tied)


def validate_ordered_outputs(
    outputs: dict,
    variables,
    expected,
    decimals: list[int | None],
) -> ValidatorResult:
    """Validate a case's ordered scalar outputs with pinned numeric precision.

    ``None`` means exact equality. Numeric slots reject booleans, NaN and strings.
    The helper deliberately does not coerce dates or labels; cases needing semantic
    timestamp or alias handling should keep a case-specific validator.
    """
    if not (len(variables) == len(expected) == len(decimals)):
        raise ValueError("variables, expected values and decimals must have equal length")
    errors = []
    variable_results = {}
    missing = False
    for variable, target, places in zip(variables, expected, decimals):
        actual = outputs.get(variable.name)
        if actual is None:
            missing = True
            variable_results[variable.name] = {
                "is_set": False, "correct": False, "message": "not set",
            }
            continue
        matches = actual == target if places is None else numeric_equal(
            actual, target, decimals=places
        )
        if isinstance(matches, np.bool_):
            matches = bool(matches)
        if not matches:
            formatted = repr(target) if places is None else f"{target:.{places}f}"
            message = f"{variable.name}={actual!r}, expected {formatted}"
            errors.append(message)
        else:
            message = "correct"
        variable_results[variable.name] = {
            "is_set": True, "correct": bool(matches), "message": message,
        }
    message = "required output not set" if missing else "; ".join(errors) or "correct"
    return ValidatorResult(
        not missing and not errors,
        message,
        missing,
        variable_results,
    )
