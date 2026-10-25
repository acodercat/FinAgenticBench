"""Boundaries for the Treasury auction bid-dispersion case.

Anchors and properties are kept in separate tests so that a relational assertion
is never decided by a literal equality asserted earlier in the same function.
"""

import pandas as pd
import pytest

from core.data import load_expansion_table
from evals.comparison_ranking import treasury_auction_bid_dispersion_stop_allotment as case


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["auction_count"] == 168
    assert answers["distinct_original_terms"] == 7
    assert round(answers["median_dispersion_bp"], 4) == 5.5
    assert round(answers["median_stop_allotment_pct"], 4) == 53.13
    assert round(answers["allotment_cover_correlation"], 4) == 0.1091
    assert answers["widest_dispersion_term"] == "30-Year"
    assert round(answers["widest_dispersion_median_bp"], 4) == 6.35
    assert answers["lowest_allotment_term"] == "30-Year"
    assert round(answers["lowest_allotment_median_pct"], 4) == 35.865
    assert round(answers["median_concession_bp"], 4) == 1.5
    assert answers["widest_concession_term"] == "2-Year"
    assert round(answers["widest_concession_median_bp"], 4) == 3.5
    assert answers["widest_auction_cusip"] == "91282CJZ5"
    assert answers["widest_auction_date"] == "2024-04-10"
    assert round(answers["widest_auction_allotment_pct"], 4) == 54.1
    assert round(answers["widest_auction_concession_bp"], 4) == 1.0


# --- properties (no literals) ----------------------------------------------

def test_two_demand_measures_are_near_independent():
    """The premise of the case: allotment at the stop and bid-to-cover are not
    the same reading. A strong correlation would make the question pointless."""
    assert abs(case.analysis()["allotment_cover_correlation"]) < 0.4


def test_dispersion_is_positive_and_bounded():
    answers = case.analysis()
    assert 0 < answers["median_dispersion_bp"] < answers["widest_dispersion_median_bp"] * 2
    assert answers["widest_dispersion_median_bp"] > answers["median_dispersion_bp"]


def test_every_original_term_maps_to_a_par_tenor():
    frame = case._auctions()
    assert set(frame.original_security_term) <= set(case.TENOR_COLUMN)
    assert frame.original_security_term.nunique() == case.analysis()["distinct_original_terms"]


# --- the exclusions that change the answer ---------------------------------

def test_tips_and_frn_exclusion_is_load_bearing():
    """A TIPS stop is a real yield; leaving it in produces a concession of the
    wrong order of magnitude, and the exclusion removes a material count."""
    raw = load_expansion_table("treasury_auctions")
    window = raw.loc[
        raw.auction_date.astype(str).between(*case.WINDOW)
        & raw.security_type.isin(["Note", "Bond"])
    ]
    assert len(window) > case.analysis()["auction_count"]
    tips = window.loc[window.inflation_index_security.eq("Yes")]
    assert len(tips) > 0
    real_yields = pd.to_numeric(tips.high_yield, errors="coerce").dropna()
    nominal = pd.to_numeric(case._auctions().high_yield, errors="coerce")
    assert real_yields.min() < nominal.min() - 1.0


def test_cusip_alone_does_not_identify_an_auction():
    """A reopened security is auctioned more than once, so the CUSIP and the
    auction date together are what pin the record; the selected auction is a
    reopening whose term at auction is not its original term."""
    frame = case._auctions()
    counts = frame.groupby("cusip").size()
    assert int((counts > 1).sum()) > 0
    answers = case.analysis()
    row = frame.loc[frame.cusip.eq(answers["widest_auction_cusip"])
                    & frame.auction_date.astype(str).eq(answers["widest_auction_date"])].iloc[0]
    assert row.original_security_term == "10-Year"
    assert row.security_term != row.original_security_term


def test_tie_on_the_widest_gap_is_resolved_by_date():
    frame = case._auctions()
    widest = frame.loc[frame.dispersion_bp.eq(frame.dispersion_bp.max())]
    assert len(widest) > 1
    assert case.analysis()["widest_auction_date"] == str(widest.auction_date.min())


def test_the_stated_rules_that_do_not_bind():
    """House rule: a selection states its rule even when nothing trips it, and
    the case asserts that nothing does, so a later data change is visible."""
    raw = load_expansion_table("treasury_auctions")
    window = raw.loc[
        raw.auction_date.astype(str).between(*case.WINDOW)
        & raw.security_type.isin(["Note", "Bond"])
        & raw.floating_rate.eq("No")
        & raw.inflation_index_security.eq("No")
    ]
    fields = pd.concat([pd.to_numeric(window[c], errors="coerce") for c in case.NUMERIC], axis=1)
    assert not fields.isna().any(axis=1).any()

    frame = case._auctions()
    for column in ("dispersion_bp", "allocation_pctage"):
        medians = frame.groupby("original_security_term")[column].median()
        assert not medians.duplicated().any()


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

def test_concession_comes_from_the_published_curve():
    """Stronger than checking the loader is called: the reported concession must
    equal the stop less the curve value this case says to use."""
    answers = case.analysis()
    frame = case._auctions()
    row = frame.loc[frame.cusip.eq(answers["widest_auction_cusip"])
                    & frame.auction_date.astype(str).eq(answers["widest_auction_date"])].iloc[0]
    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["Date"] = curve.Date.astype(str)
    tenor = case.TENOR_COLUMN[row.original_security_term]
    par = pd.to_numeric(
        curve.loc[curve.Date.le(str(row.auction_date))].sort_values("Date").iloc[-1][tenor],
        errors="coerce",
    )
    assert round((row.high_yield - float(par)) * 100, 4) == round(
        answers["widest_auction_concession_bp"], 4
    )


@pytest.mark.parametrize("missing", ["treasury_auctions", "treasury_yield_curve"])
def test_both_sources_are_load_bearing(monkeypatch, missing):
    for cached in (case._auctions, case.analysis, case.ground_truth):
        cached.cache_clear()
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
        for cached in (case._auctions, case.analysis, case.ground_truth):
            cached.cache_clear()
        case.analysis()
