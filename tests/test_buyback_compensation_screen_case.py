"""Boundaries for the repurchase versus share-based compensation screen.

Anchors and properties are kept in separate tests so that a relational
assertion is never decided by a literal equality asserted earlier in the same
function.
"""

from decimal import Decimal

import pytest

from core.data import load_financial_statements
from evals.risk_screening import buyback_share_compensation_cash_coverage_screen as case


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


def _clear():
    for cached in (case._annual_reports, case._tagged, case._eligible, case._comparable,
                   case._dividend_facts,
                   case.analysis, case.ground_truth):
        cached.cache_clear()


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["eligible_filer_count"] == 1251
    assert answers["negative_value_excluded_filer_count"] == 8
    assert answers["repurchasing_filer_count"] == 1039
    assert round(answers["operating_cash_flow_bn"], 6) == 1585.787171
    assert round(answers["free_cash_flow_bn"], 6) == 1138.760931
    assert answers["negative_free_cash_flow_filers"] == 202
    assert round(answers["repurchase_bn"], 6) == 607.377827
    assert answers["repurchase_above_fcf_filers"] == 281
    assert round(answers["share_based_compensation_bn"], 6) == 176.353726
    assert round(answers["compensation_to_operating_cash_flow_median_pct"], 4) == 8.1852
    assert answers["repurchase_below_compensation_filers"] == 352
    assert answers["repurchase_above_operating_cash_flow_filers"] == 177
    assert answers["comparable_filer_count"] == 989
    assert round(answers["below_compensation_share_rise_pct"], 4) == 72.5076
    assert round(answers["above_compensation_share_rise_pct"], 4) == 19.7568
    assert round(answers["below_compensation_median_share_change_pct"], 4) == 0.7811
    assert answers["dividend_reporting_filer_count"] == 122
    assert answers["single_concept_reporting_filer_count"] == 49
    assert answers["aggregate_and_component_filer_count"] == 1
    assert round(answers["dividend_bn"], 6) == 281.56683


# --- properties (no literals) ----------------------------------------------

def test_the_dollar_comparison_predicts_the_share_count():
    """The point of the case. If the two sides rose at similar rates the
    comparison drawn in turn three would carry no information."""
    answers = case.analysis()
    assert answers["below_compensation_share_rise_pct"] - answers["above_compensation_share_rise_pct"] > 30


def test_the_two_sides_move_their_counts_in_opposite_directions():
    comparable = case._comparable()
    below = comparable.loc[comparable[case.REPURCHASE] < comparable[case.COMPENSATION]]
    above = comparable.loc[comparable[case.REPURCHASE] >= comparable[case.COMPENSATION]]
    assert len(below) > 0 and len(above) > 0
    assert below.share_change_pct.median() > 0 > above.share_change_pct.median()


def test_the_add_back_is_material():
    """The noncash expense is large enough to matter for dilution analysis."""
    answers = case.analysis()
    assert 0 < answers["compensation_to_operating_cash_flow_median_pct"] < 50
    assert answers["share_based_compensation_bn"] > 0.05 * answers["operating_cash_flow_bn"]


def test_capital_expenditure_reduces_the_measure():
    answers = case.analysis()
    assert answers["free_cash_flow_bn"] < answers["operating_cash_flow_bn"]


def test_reported_subsets_are_strict():
    answers = case.analysis()
    assert 0 < answers["repurchase_below_compensation_filers"] < answers["repurchasing_filer_count"]
    assert 0 < answers["comparable_filer_count"] < answers["repurchasing_filer_count"]
    assert 0 < answers["repurchasing_filer_count"] < answers["eligible_filer_count"]
    assert 0 < answers["negative_free_cash_flow_filers"] < answers["eligible_filer_count"]


def test_the_negative_value_screen_removes_filers():
    answers = case.analysis()
    assert answers["negative_value_excluded_filer_count"] > 0
    assert len(case._tagged()) == answers["eligible_filer_count"] + answers["negative_value_excluded_filer_count"]


def test_some_repurchases_exceed_the_cash_that_funded_them():
    answers = case.analysis()
    assert 0 < answers["repurchase_above_operating_cash_flow_filers"] < answers["repurchase_above_fcf_filers"]


def test_the_single_concept_finds_a_minority_of_reporters():
    """The fifth turn only says something if the concept the first table carries
    is partial; if it found everyone, the fact file would be decoration."""
    answers = case.analysis()
    assert 0 < answers["single_concept_reporting_filer_count"] < answers["dividend_reporting_filer_count"]
    assert answers["single_concept_reporting_filer_count"] < 0.75 * answers["dividend_reporting_filer_count"]


def test_an_aggregate_supersedes_its_components():
    """Adding an aggregate to its own components counts one payment twice, and
    the difference is large enough to change the reported total."""
    facts = case._dividend_facts()
    answers = case.analysis()
    naive = float(sum(Decimal(str(x)) for x in facts.value) / case.BILLION)
    assert round(naive, 6) != round(answers["dividend_bn"], 6)
    assert naive > answers["dividend_bn"]
    outside = _reported()
    outside["dividend_bn"] = round(naive, 6)
    assert not case.validate(outside).success


def test_a_restated_year_is_taken_as_first_filed():
    facts = case._dividend_facts()
    assert not facts.duplicated(["cik", "concept"]).any()
    assert facts.fact_days.between(*case.ANNUAL_FACT_DAYS).all()


def test_the_stated_accession_tie_break_does_not_bind():
    """The retention rule names an accession tie-break the data never needs;
    the case asserts that, so a later data change is visible."""
    statements = load_financial_statements()
    window = statements.loc[
        statements.form.eq("10-K") & statements.fiscal_period.eq("FY")
        & statements.report_date.astype(str).between(*case.WINDOW)
    ]
    per_filing = window.groupby(["cik", "report_date", "filed_date"]).size()
    assert int((per_filing > 1).sum()) == 0


# --- rejections -------------------------------------------------------------

def test_sign_flipping_excluded_values_is_a_different_answer():
    """Flipping an unexpected negative tag invents an unreported amount."""
    tagged = case._tagged()
    flipped = tagged.copy()
    for column in case.NON_NEGATIVE:
        flipped[column] = flipped[column].abs()
    answers = case.analysis()
    assert len(flipped) > answers["eligible_filer_count"]
    naive = float(sum(flipped[case.REPURCHASE]) / 1e9)
    assert round(naive, 6) != round(answers["repurchase_bn"], 6)
    outside = _reported()
    outside["repurchase_bn"] = round(naive, 6)
    assert not case.validate(outside).success


def test_removing_the_pairing_window_is_a_different_answer(monkeypatch):
    reference = _reported()
    _clear()
    try:
        monkeypatch.setattr(case, "PAIR_GAP_DAYS", (1, 3650))
        loosened = case.analysis()
        widened = round(loosened["below_compensation_share_rise_pct"], 4)
        count = loosened["comparable_filer_count"]
    finally:
        monkeypatch.undo()
        _clear()
    assert count > reference["comparable_filer_count"]
    assert widened != reference["below_compensation_share_rise_pct"]
    outside = dict(reference)
    outside["below_compensation_share_rise_pct"] = widened
    assert not case.validate(outside).success


def test_amendments_exist_and_are_excluded():
    statements = load_financial_statements()
    amended = statements.loc[
        statements.form.eq("10-K/A") & statements.fiscal_period.eq("FY")
        & statements.report_date.astype(str).between(*case.WINDOW)
    ]
    assert len(amended) > 0
    assert case._tagged().form.eq("10-K").all()


def test_dividends_are_too_thinly_tagged_to_join():
    """The listed overclaim: adding dividends here would describe a differently
    selected population, not the same one."""
    annual = case._annual(case.WINDOW)
    assert annual.PaymentsOfDividends_ytd.notna().sum() < 0.25 * len(annual)


def test_strict_display_boundary_both_sides():
    answers = case.analysis()
    reported = _reported()
    for variable in case.variables:
        precision = case.PRECISION[variable.name]
        if precision is None:
            continue
        for direction in (1, -1):
            outside = dict(reported)
            step = 0.7 * 10 ** (-precision) if precision else 1
            outside[variable.name] = answers[variable.name] + direction * step
            assert not case.validate(outside).success, f"{variable.name} {direction:+d}"


# --- source dependence ------------------------------------------------------

def test_share_change_matches_the_two_filings():
    """Stronger than checking the loader is called: each reported change must
    be the ratio of the counts the two filings carry."""
    comparable = case._comparable()
    recomputed = (comparable[case.BASIC_SHARES] / comparable.prior_basic_shares - 1) * 100
    assert (recomputed - comparable.share_change_pct).abs().max() < 1e-9
    assert comparable.gap_days.between(*case.PAIR_GAP_DAYS).all()


@pytest.mark.parametrize("missing", ["load_financial_statements", "load_facts"])
def test_both_sources_are_load_bearing(monkeypatch, missing):
    _clear()

    def guarded(*args, **kwargs):
        raise RuntimeError(f"{missing} unavailable")

    try:
        monkeypatch.setattr(case, missing, guarded)
        with pytest.raises(RuntimeError):
            case.analysis()
    finally:
        monkeypatch.undo()
        _clear()
        case.analysis()
