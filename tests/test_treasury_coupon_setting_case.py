"""Boundaries for the Treasury coupon-setting and tranche-pricing case.

Anchors pin the literals. Property tests name no literal, so each can fail on
its own rather than being decided by an equality asserted earlier in the same
function.
"""

import numpy as np
import pandas as pd
import pytest

from evals.calculation_reconciliation import treasury_coupon_setting_and_tranche_pricing as case


def _clear():
    for cached in (case._auctions, case._window_auctions, case._tranches,
                   case.analysis, case.ground_truth):
        cached.cache_clear()


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] else answers[v.name]
        for v in case.variables
    }


def _originals():
    frame = case._window_auctions()
    return frame.loc[~frame.reopened]


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["original_issue_count"] == 623
    assert round(answers["max_yield_above_coupon_bp"], 1) == 12.4
    assert answers["premium_original_auction_date"] == "2021-02-23"
    assert answers["reopening_count"] == 239
    assert answers["reopening_premium_count"] == 83
    assert answers["outstanding_security_count"] == 350
    assert answers["reopened_security_count"] == 141
    assert answers["mixed_tranche_security_count"] == 62
    assert answers["high_yield_rule_failures"] == 2
    assert round(answers["issued_less_outstanding_billion_usd"], 1) == 219.3
    assert round(answers["issued_less_outstanding_weighted_coupon_pct"], 4) == 2.7307
    assert round(answers["median_coupon_pct"], 3) == 3.625


# --- the rule the data decides ---------------------------------------------

def test_the_rule_reproduces_every_original_and_nothing_simpler_does():
    originals = _originals()
    assert np.isclose(case.coupon_from_yield(originals.high_yield), originals.int_rate).all()
    rounded = np.round(originals.high_yield / case.EIGHTH) * case.EIGHTH
    assert not np.isclose(rounded, originals.int_rate).all()
    no_floor = np.floor(originals.high_yield / case.EIGHTH + 1e-9) * case.EIGHTH
    assert not np.isclose(no_floor, originals.int_rate).all()


def test_only_the_minimum_coupon_puts_an_original_above_par():
    originals = _originals()
    gap = originals.high_yield - originals.int_rate
    assert int((gap < 0).sum()) == 1
    assert int((originals.high_yield < case.EIGHTH).sum()) == 1
    above = originals.loc[originals.price_per100 > 100]
    assert above.high_yield.iloc[0] < case.EIGHTH


def test_the_largest_gap_stays_inside_one_eighth():
    assert 0 < case.analysis()["max_yield_above_coupon_bp"] < case.EIGHTH * case.BASIS_POINTS


# --- properties (no literals) ----------------------------------------------

def test_a_reopening_is_above_par_exactly_when_its_yield_is_below_the_coupon():
    frame = case._window_auctions()
    reopenings = frame.loc[frame.reopened]
    assert ((reopenings.price_per100 > 100) == (reopenings.high_yield < reopenings.int_rate)).all()


def test_premium_reopenings_are_a_material_minority():
    answers = case.analysis()
    assert 0.1 < answers["reopening_premium_count"] / answers["reopening_count"] < 0.5


def test_mixed_securities_are_a_subset_of_reopened_ones():
    answers = case.analysis()
    assert 0 < answers["mixed_tranche_security_count"] < answers["reopened_security_count"]
    assert answers["reopened_security_count"] < answers["outstanding_security_count"]


def test_every_tranche_of_a_security_shares_one_coupon():
    tranches = case._tranches()
    assert tranches.groupby("cusip").interest_rate_pct.nunique().eq(1).all()


def test_the_rule_failures_are_explained_by_the_average_yield():
    """The two outstanding originals the rule misses were sold in 1996 at
    multiple-price auctions, and the average accepted yield reproduces both."""
    auctions = case.load_expansion_table("treasury_auctions").copy()
    auctions["cusip"] = auctions.cusip.astype(str)
    auctions["issue_date"] = auctions.issue_date.astype(str)
    auctions["average"] = pd.to_numeric(auctions.avg_med_yield, errors="coerce")
    auctions["high"] = pd.to_numeric(auctions.high_yield, errors="coerce")
    first = case._tranches().sort_values("issue_date").drop_duplicates("cusip")
    matched = first.merge(auctions[["cusip", "issue_date", "average", "high", "auction_date"]],
                          on=["cusip", "issue_date"])
    missed = matched.loc[~np.isclose(case.coupon_from_yield(matched.high), matched.interest_rate_pct)]
    assert len(missed) == case.analysis()["high_yield_rule_failures"]
    assert np.isclose(case.coupon_from_yield(missed.average), missed.interest_rate_pct).all()
    assert (missed.auction_date.astype(str) < "1998-11-01").all()


def test_early_retirement_falls_on_low_coupons():
    answers = case.analysis()
    assert answers["issued_less_outstanding_weighted_coupon_pct"] < answers["median_coupon_pct"] - 0.5


def test_the_median_coupon_needs_no_interpolation():
    coupons = np.sort(case._tranches().groupby("cusip").interest_rate_pct.first().to_numpy())
    assert coupons[len(coupons) // 2 - 1] == coupons[len(coupons) // 2]


def test_outstanding_is_never_above_issued_and_is_not_in_the_auction_record():
    """The amount outstanding is the debt statement's contribution: the
    auction record's accepted totals match the amounts issued, not those
    outstanding."""
    tranches = case._tranches()
    stock = tranches.groupby("cusip").agg(
        issued=("issued_million_usd", "sum"), outstanding=("outstanding_million_usd", "sum")
    )
    assert (stock.outstanding <= stock.issued + 0.5).all()
    auctions = case.load_expansion_table("treasury_auctions").copy()
    auctions["cusip"] = auctions.cusip.astype(str)
    auctions["accepted"] = pd.to_numeric(auctions.total_accepted, errors="coerce") / 1e6
    accepted = auctions.loc[auctions.cusip.isin(stock.index)].groupby("cusip").accepted.sum()
    assert abs(accepted.sum() - stock.issued.sum()) < 0.001 * stock.issued.sum()
    assert stock.issued.sum() - stock.outstanding.sum() > 100_000


# --- readings measured to leave the answer unchanged -------------------------

def test_the_tranche_yield_source_does_not_change_the_mixed_count():
    auctions = case._auctions()[["cusip", "issue_date", "high_yield"]]
    tranches = case._tranches().merge(auctions, on=["cusip", "issue_date"])
    flags = tranches.assign(
        above=tranches.high_yield < tranches.interest_rate_pct,
        below=tranches.high_yield > tranches.interest_rate_pct,
    ).groupby("cusip").agg(above=("above", "any"), below=("below", "any"))
    assert int((flags.above & flags.below).sum()) == case.analysis()["mixed_tranche_security_count"]


def test_class_totals_are_not_counted_as_securities():
    tranches = case._tranches()
    assert not tranches.cusip.str.startswith("Total").any()
    assert tranches.maturity_date.notna().all()


def test_the_reported_tranche_yield_is_a_different_answer():
    """The debt statement's own yield for a 1996 tranche is the average the
    coupon was set from, so the rule applied to it misses nothing."""
    first = case._tranches().sort_values("issue_date").drop_duplicates("cusip")
    misses = int((~np.isclose(case.coupon_from_yield(first.yield_pct), first.interest_rate_pct)).sum())
    assert misses != case.analysis()["high_yield_rule_failures"]
    outside = _reported()
    outside["high_yield_rule_failures"] = misses
    assert not case.validate(outside).success


def test_the_auction_record_issued_amount_is_a_different_answer():
    """Taking the issued amount from the auction record rather than the debt
    statement moves the difference by about a billion dollars."""
    tranches = case._tranches()
    auctions = case.load_expansion_table("treasury_auctions").copy()
    auctions["cusip"] = auctions.cusip.astype(str)
    auctions["issue_date"] = auctions.issue_date.astype(str)
    auctions["accepted"] = pd.to_numeric(auctions.total_accepted, errors="coerce") / 1e6
    matched = tranches.merge(auctions[["cusip", "issue_date", "accepted"]], on=["cusip", "issue_date"])
    alternative = (matched.accepted.sum() - tranches.outstanding_million_usd.sum()) / 1e3
    outside = _reported()
    outside["issued_less_outstanding_billion_usd"] = round(float(alternative), 1)
    assert not case.validate(outside).success


# --- rejections -------------------------------------------------------------

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

@pytest.mark.parametrize("missing", ["treasury_auctions", "treasury_marketable_securities"])
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
