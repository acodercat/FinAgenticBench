"""Boundaries for the Form 4 holding-row beneficial-ownership case."""

import pytest

from evals.calculation_reconciliation import (
    insider_indirect_holdings_outside_transaction_stream as case,
)


def _clear():
    for name in ("_population", "analysis", "ground_truth"):
        getattr(case, name).cache_clear()


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


def test_population_and_row_type_asymmetry():
    answers = case.analysis()
    assert answers["eligible_filing_count"] == 5196
    assert answers["filings_with_holding_rows"] == 1635
    assert answers["holding_row_count"] == 3250
    # The whole point of the case: holding rows are overwhelmingly indirect
    # while the same filings' transaction rows are overwhelmingly direct.
    assert round(answers["holding_row_indirect_pct"], 4) == 83.6923
    assert round(answers["transaction_row_indirect_pct"], 4) == 15.6661
    assert answers["holding_row_indirect_pct"] > 5 * answers["transaction_row_indirect_pct"]


def test_retention_rule_is_what_makes_the_total_finite():
    answers = case.analysis()
    assert answers["issuer_owner_pair_count"] == 3393
    assert answers["latest_filing_holding_row_count"] == 1742
    # Retaining one filing per owner must discard rows, or the rule is inert.
    assert answers["latest_filing_holding_row_count"] < answers["holding_row_count"]


def test_selected_issuer_and_stake():
    answers = case.analysis()
    assert answers["selected_issuer_cik"] == "0001045810"
    assert answers["selected_owner_count"] == 6
    assert round(answers["selected_holding_shares"], 2) == 824225372.00
    assert round(answers["selected_indirect_holding_shares"], 2) == 812629402.00
    assert round(answers["largest_holding_row_shares"], 2) == 602819280.00
    assert round(answers["selected_holdings_to_shares_outstanding_pct"], 4) == 3.3656
    # The indirect portion is the bulk, and the largest single row is a trust
    # line big enough to dominate the issuer total on its own.
    assert answers["selected_indirect_holding_shares"] / answers["selected_holding_shares"] > 0.98
    assert answers["largest_holding_row_shares"] / answers["selected_holding_shares"] > 0.7


def test_naive_cross_filing_sum_is_rejected():
    """Summing holding rows over every eligible filing, rather than one filing
    per reporting owner, is the listed overclaim; it must not validate."""
    answers = case.analysis()
    naive = _reported()
    naive["latest_filing_holding_row_count"] = answers["holding_row_count"]
    assert not case.validate(naive).success


def test_strict_display_boundary_for_each_numeric_output():
    answers = case.analysis()
    reported = _reported()
    for variable in case.variables:
        precision = case.PRECISION[variable.name]
        if precision in (None, 0):
            continue
        outside = dict(reported)
        outside[variable.name] = answers[variable.name] + 0.7 * 10 ** (-precision)
        assert not case.validate(outside).success, variable.name


@pytest.mark.parametrize("missing", ["insider", "facts"])
def test_both_sources_are_load_bearing(monkeypatch, missing):
    """sec_companyfacts supplies the outstanding-share denominator and
    sec_insider the Form 4 rows; removing either must break the case."""
    _clear()
    try:
        if missing == "insider":
            def boom(_name):
                raise RuntimeError("sec_insider unavailable")
            monkeypatch.setattr(case, "load_insider_table", boom)
        else:
            def boom():
                raise RuntimeError("sec_companyfacts unavailable")
            monkeypatch.setattr(case, "load_facts", boom)
        with pytest.raises(RuntimeError):
            case.analysis()
    finally:
        monkeypatch.undo()
        _clear()
        case.analysis()
