"""Boundaries for the held-to-maturity capital recognition case.

Anchors and properties are kept in separate tests so that a relational
assertion is never decided by a literal equality asserted earlier in the same
function.
"""

from decimal import Decimal

import pytest

from core.data import load_expansion_table
from evals.risk_screening import bank_htm_unrealized_loss_capital_recognition as case


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


def _clear():
    for cached in (case._banks, case._ratio_group, case.analysis, case.ground_truth):
        cached.cache_clear()


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["bank_count"] == 4435
    assert answers["htm_holder_count"] == 1639
    assert answers["htm_loss_holder_count"] == 1199
    assert round(answers["htm_unrealized_bn"], 6) == -221.749484
    assert round(answers["afs_unrealized_bn"], 6) == -115.374711
    assert answers["cblr_elector_count"] == 1725
    assert answers["ratio_bank_count"] == 2710
    assert round(answers["cet1_capital_bn"], 6) == 2192.620735
    assert round(answers["aggregate_cet1_ratio_pct"], 4) == 14.3683
    assert round(answers["aggregate_adjusted_ratio_pct"], 4) == 12.9253
    assert answers["below_seven_before_count"] == 4
    assert answers["below_seven_after_count"] == 9
    assert answers["worst_bank_rssd"] == "480330"
    assert round(answers["worst_htm_loss_to_cet1_pct"], 4) == 235.2045
    assert answers["htm_loss_exceeds_cet1_count"] == 2
    assert round(answers["pledged_share_pct"], 4) == 40.3287


# --- properties (no literals) ----------------------------------------------

def test_the_electors_are_exactly_the_banks_without_risk_weighted_assets():
    """The structural fact behind the ratio group, and the reason the three
    counts add up rather than two of them plus a residual."""
    frame = case._banks()
    weighted = frame.standardized_risk_weighted_assets_thousand_usd
    assert int((frame.cblr_election & weighted.notna()).sum()) == 0
    assert int((~frame.cblr_election & weighted.isna()).sum()) == 0
    answers = case.analysis()
    assert answers["cblr_elector_count"] + answers["ratio_bank_count"] == answers["bank_count"]


def test_recognising_the_loss_lowers_the_ratio():
    answers = case.analysis()
    assert answers["aggregate_adjusted_ratio_pct"] < answers["aggregate_cet1_ratio_pct"]
    assert answers["below_seven_after_count"] > answers["below_seven_before_count"]


def test_the_aggregate_moves_less_than_the_worst_banks():
    """The claim in turn three rests on this: a size-weighted aggregate is not
    the typical bank."""
    answers = case.analysis()
    aggregate_move = answers["aggregate_cet1_ratio_pct"] - answers["aggregate_adjusted_ratio_pct"]
    assert aggregate_move < 5
    assert answers["worst_htm_loss_to_cet1_pct"] > 100


def test_reported_subsets_are_strict():
    answers = case.analysis()
    assert 0 < answers["htm_loss_holder_count"] < answers["htm_holder_count"] < answers["bank_count"]
    assert 0 < answers["htm_loss_exceeds_cet1_count"] < answers["ratio_bank_count"]
    assert 0 < answers["below_seven_before_count"] < answers["ratio_bank_count"]


def test_both_portfolios_carry_a_loss_and_the_held_one_is_larger():
    answers = case.analysis()
    assert answers["htm_unrealized_bn"] < 0
    assert answers["afs_unrealized_bn"] < 0
    assert abs(answers["htm_unrealized_bn"]) > abs(answers["afs_unrealized_bn"])


def test_a_material_share_of_the_book_is_pledged():
    answers = case.analysis()
    assert 10 < answers["pledged_share_pct"] < 90


# --- the stated rule that does not bind -------------------------------------

def test_the_worst_bank_is_a_strict_leader():
    """House rule: the ranking states a tie rule and the case asserts nothing
    currently ties, so a later data change is visible."""
    group = case._ratio_group()
    positive = group.loc[group.common_equity_tier1_capital_thousand_usd > 0]
    load = (-positive.htm_unrealized / positive.common_equity_tier1_capital_thousand_usd).sort_values(
        ascending=False
    )
    assert load.iloc[0] > load.iloc[1]


def test_the_holder_count_is_pinned_to_a_positive_cost():
    """One bank reports a zero amortised cost against a positive fair value,
    so the looser reading of holding a portfolio gives a different count."""
    frame = case._banks()
    answers = case.analysis()
    loose = int(
        ((frame.htm_amortized_cost_thousand_usd > 0) | (frame.htm_fair_value_thousand_usd > 0)).sum()
    )
    assert loose == answers["htm_holder_count"] + 1
    outside = _reported()
    outside["htm_holder_count"] = loose
    assert not case.validate(outside).success


def test_the_computed_ratio_matches_each_bank_s_own_reported_one():
    """Stronger than checking the loader is called: the constructed ratio must
    reproduce the ratio the bank itself reports, to its published precision."""
    group = case._ratio_group()
    computed = (
        group.common_equity_tier1_capital_thousand_usd
        / group.standardized_risk_weighted_assets_thousand_usd
        * 100
    ).round(4)
    reported = group.reported_standardized_cet1_ratio_pct.astype(float).round(4)
    agree = (computed - reported).abs() < 1e-9
    assert agree.mean() > 0.99


def test_the_two_schedules_agree_on_basis_inside_the_ratio_group():
    group = case._ratio_group()
    assert group.reporting_basis.astype(str).eq(
        group.capital_reporting_basis.astype(str)
    ).all()


# --- rejections -------------------------------------------------------------

def test_taking_held_to_maturity_at_fair_value_is_a_different_denominator():
    """The query pins the carrying basis of each classification; marking the
    held portfolio to market instead moves the pledged share."""
    frame = case._banks()
    answers = case.analysis()
    marked = float(
        sum(Decimal(str(x)) for x in frame.pledged_securities_book_value_thousand_usd)
        / sum(Decimal(str(x)) for x in (frame.htm_fair_value_thousand_usd + frame.afs_fair_value_thousand_usd))
        * 100
    )
    assert round(marked, 4) != round(answers["pledged_share_pct"], 4)
    outside = _reported()
    outside["pledged_share_pct"] = round(marked, 4)
    assert not case.validate(outside).success


def test_adding_the_available_for_sale_mark_too_is_a_different_answer():
    """The listed overclaim: whether that mark is already inside reported
    capital depends on an election the schedules do not record."""
    group = case._ratio_group()
    answers = case.analysis()
    capital = sum(Decimal(str(x)) for x in group.common_equity_tier1_capital_thousand_usd)
    weighted = sum(Decimal(str(x)) for x in group.standardized_risk_weighted_assets_thousand_usd)
    both = sum(Decimal(str(x)) for x in (group.htm_unrealized + group.afs_unrealized))
    naive = float((capital + both) / weighted * 100)
    assert round(naive, 4) != round(answers["aggregate_adjusted_ratio_pct"], 4)
    outside = _reported()
    outside["aggregate_adjusted_ratio_pct"] = round(naive, 4)
    assert not case.validate(outside).success


def test_using_the_whole_file_loss_in_the_adjustment_is_a_different_answer():
    """The adjustment uses the ratio group's own loss, not the file's."""
    answers = case.analysis()
    group = case._ratio_group()
    weighted = float(sum(Decimal(str(x)) for x in group.standardized_risk_weighted_assets_thousand_usd))
    capital = float(sum(Decimal(str(x)) for x in group.common_equity_tier1_capital_thousand_usd))
    naive = (capital + answers["htm_unrealized_bn"] * 1e6) / weighted * 100
    assert round(naive, 4) != round(answers["aggregate_adjusted_ratio_pct"], 4)


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

def test_the_two_schedules_join_one_to_one():
    frame = case._banks()
    securities = load_expansion_table("ffiec_call_reports_securities")
    assert len(frame) == len(securities)
    assert not frame.bank_rssd_id.duplicated().any()


@pytest.mark.parametrize("missing", ["ffiec_call_reports_securities", "ffiec_call_reports_capital"])
def test_both_sources_are_load_bearing(monkeypatch, missing):
    _clear()
    real = case.load_expansion_table

    def guarded(name):
        if name == missing:
            raise RuntimeError(f"{name} unavailable")
        return real(name)

    try:
        monkeypatch.setattr(case, "load_expansion_table", guarded)
        with pytest.raises(RuntimeError):
            case.analysis()
    finally:
        monkeypatch.undo()
        _clear()
        case.analysis()
