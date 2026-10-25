import pandas as pd
import numpy as np
from cave_agent import Variable

from core.footnotes import PriceRangeParseError, extract_price_range
from core.industry import canonical_industry_mapping, canonical_industry_name
from core.types import Turn
from core.validation import (
    ValidatorResult,
    boolean_equal,
    datetime_equal,
    numeric_equal,
    reports_as,
    select_extreme_key,
    turn_validator,
    validate_ordered_outputs,
)


def test_datetime_equal_accepts_equivalent_iso_representations():
    expected = "2025-03-20T22:36:42.000Z"
    assert datetime_equal("2025-03-20 22:36:42+00:00", expected)
    assert datetime_equal(pd.Timestamp("2025-03-20 18:36:42-04:00"), expected)
    assert not datetime_equal("2025-03-20T22:36:43Z", expected)
    assert not datetime_equal("not-a-timestamp", expected)


def test_price_range_extractor_accepts_supported_sec_variants():
    assert extract_price_range("prices ranging from $69.6653 to $70.005") == (69.6653, 70.005)
    assert extract_price_range("prices ranging from\n$10.25\nto\n$11.75") == (10.25, 11.75)
    assert extract_price_range("prices were executed between $8.10 and $8.90") == (8.1, 8.9)
    assert extract_price_range("price range: $101.20−$102.40") == (101.2, 102.4)


def test_price_range_extractor_rejects_unanchored_numbers():
    try:
        extract_price_range("The reporting person bought 100 shares on 2025-01-02.")
    except PriceRangeParseError:
        pass
    else:
        raise AssertionError("unanchored numeric footnote should not be admitted")


def test_boolean_equal_accepts_numpy_boolean_but_not_truthy_lookalikes():
    assert boolean_equal(np.bool_(False), False)
    assert boolean_equal(True, True)
    assert not boolean_equal("False", False)
    assert not boolean_equal(1, True)


def test_zero_decimal_numeric_outputs_are_exact():
    assert numeric_equal(323, 323.0, decimals=0)
    assert not numeric_equal(322.6, 323, decimals=0)
    assert numeric_equal(3.2344, 3.234, decimals=3)


def test_reports_as_accepts_the_report_and_the_unrounded_value_only():
    # The truth -1.38461 reports as -1.38 at two decimals.
    assert reports_as(-1.38, -1.38461, decimals=2)
    assert reports_as(-1.38461, -1.38461, decimals=2)
    # Both sit inside numeric_equal's 0.006 band but report as -1.39.
    assert numeric_equal(-1.39, -1.38461, decimals=2)
    assert not reports_as(-1.39, -1.38461, decimals=2)
    assert not reports_as(-1.38539, -1.38461, decimals=2)


def test_reports_as_rounds_half_away_from_zero_in_decimal():
    # round(2.675, 2) gives 2.67 on the binary float; the report is 2.68.
    assert reports_as(2.68, 2.675, decimals=2)
    assert reports_as(-2.68, -2.675, decimals=2)


def test_reports_as_keeps_counts_exact_and_rejects_non_numbers():
    assert reports_as(12, 12, decimals=0)
    assert not reports_as(12.4, 12, decimals=0)
    for bad in (None, True, float("nan"), float("inf"), "1.00"):
        assert not reports_as(bad, 1.0, decimals=2)


def test_extreme_selection_uses_natural_key_tie_breaking():
    values = {"WMT": 5.0, "COST": 5.0, "AAPL": 4.0}
    assert select_extreme_key(values) == "COST"
    assert select_extreme_key({("2020-04", "A"): -2, ("2020-03", "Z"): -2}, largest=False) == (
        "2020-03", "Z"
    )


def test_industry_names_normalize_semantic_aliases():
    assert canonical_industry_name("Energy") == "Enrgy"
    assert canonical_industry_name("HiTec") == "HiTec"
    assert canonical_industry_mapping({"enrgy": 1, "Utilities": 2}) == {
        "Enrgy": 1, "Utils": 2,
    }


def test_ordered_validator_returns_structured_variable_diagnostics():
    variables = [Variable("count", None, "count"), Variable("ratio", None, "ratio")]
    verdict = validate_ordered_outputs(
        {"count": 3, "ratio": 1.25}, variables, (3, 1.5), [0, 2]
    )
    assert not verdict.success
    assert verdict.variable_results["count"]["correct"] is True
    assert verdict.variable_results["ratio"]["correct"] is False
    missing = validate_ordered_outputs(
        {"count": None, "ratio": 1.5}, variables, (3, 1.5), [0, 2]
    )
    assert missing.variables_not_set
    assert missing.variable_results["count"] == {
        "is_set": False, "correct": False, "message": "not set",
    }


def test_a_turn_validator_checks_the_resolved_scope_not_an_empty_one():
    """A short-form turn declares no stores; judging nothing must be impossible.

    The evaluator and the verification channel both resolve the scope before
    calling a validator. If either stopped doing so, this check would silently
    receive `{}` and report a pass on an unanswered turn.
    """
    seen = {}

    def check(outputs):
        seen.update(outputs)
        return ValidatorResult(bool(outputs), "checked")

    class Runtime:
        def retrieve(self, name):
            return {"a": 1, "b": 2}[name]

    resolved = Turn(query="q", validator="validate", stores=["a", "b"])
    verdict = turn_validator(check)("response", Runtime(), resolved)
    assert verdict.success and seen == {"a": 1, "b": 2}

    # An unresolved turn is the failure mode being guarded against.
    seen.clear()
    unresolved = turn_validator(check)("response", Runtime(), Turn(query="q"))
    assert not unresolved.success and seen == {}
