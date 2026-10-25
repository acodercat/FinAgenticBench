"""Boundaries for the Treasury bill discount and investment rate basis case.

Anchors pin the literals. Property tests name no literal, so each can fail on
its own rather than being decided by an equality asserted earlier in the same
function.
"""

import calendar

import numpy as np
import pandas as pd
import pytest

from evals.calculation_reconciliation import treasury_bill_discount_investment_rate_basis as case


def _clear():
    for cached in (case._bills, case._against_curve, case.analysis, case.ground_truth):
        cached.cache_clear()


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] else answers[v.name]
        for v in case.variables
    }


def _reproduced(year_days):
    bills = case._bills()
    short = bills.days.le(case.SHORT_BILL_DAYS)
    rate = (100 - bills.high_price) / bills.high_price * year_days / bills.days * 100
    return int(rate.round(3).eq(bills.high_investment_rate.round(3))[short].sum())


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["bill_auction_count"] == 867
    assert answers["short_bill_count"] == 821
    assert answers["leap_basis_count"] == 273
    assert round(answers["median_gap_4_week_bp"], 2) == 8.40
    assert round(answers["median_gap_13_week_bp"], 2) == 13.45
    assert round(answers["median_gap_26_week_bp"], 2) == 18.65
    assert round(answers["median_gap_52_week_bp"], 2) == 22.20
    assert round(answers["day_part_4_week_bp"], 4) == 6.7167
    assert round(answers["price_part_4_week_bp"], 4) == 1.6806
    assert round(answers["day_part_26_week_bp"], 4) == 6.6424
    assert round(answers["price_part_26_week_bp"], 4) == 11.6583
    assert round(answers["discount_less_curve_26_week_bp"], 2) == -18.50
    assert round(answers["investment_less_curve_26_week_bp"], 2) == 0.00
    assert round(answers["discount_less_curve_52_week_bp"], 2) == -23.00
    assert round(answers["investment_less_curve_52_week_bp"], 2) == -0.30


# --- the rule the data decides ---------------------------------------------

def test_only_the_year_following_issue_reproduces_every_short_bill():
    """The answer is unique because the data admits one rule, not because the
    query states one. Each competing reading misses a material share."""
    bills = case._bills()
    issue = pd.to_datetime(bills.issue_date)
    maturity = pd.to_datetime(bills.maturity_date)
    short = int(bills.days.le(case.SHORT_BILL_DAYS).sum())
    assert _reproduced(bills.year_days) == short

    def leap_day_inside(start, end):
        return any(
            start < pd.Timestamp(year, 2, 29) <= end
            for year in range(start.year, end.year + 1) if calendar.isleap(year)
        )

    competing = {
        "leap day inside life": pd.Series(
            [366 if leap_day_inside(s, e) else 365 for s, e in zip(issue, maturity)],
            index=bills.index,
        ),
        "issue year is leap": issue.dt.year.map(lambda y: 366 if calendar.isleap(y) else 365),
        "maturity year is leap": maturity.dt.year.map(lambda y: 366 if calendar.isleap(y) else 365),
        "flat 365": pd.Series(365, index=bills.index),
        "flat 360": pd.Series(360, index=bills.index),
    }
    for label, year_days in competing.items():
        assert _reproduced(year_days) < 0.9 * short, label


def test_the_naive_leap_rule_is_a_different_answer():
    bills = case._bills()
    issue = pd.to_datetime(bills.issue_date)
    maturity = pd.to_datetime(bills.maturity_date)
    inside = sum(
        1 for s, e in zip(issue, maturity)
        if any(s < pd.Timestamp(y, 2, 29) <= e for y in range(s.year, e.year + 1) if calendar.isleap(y))
    )
    assert inside < case.analysis()["leap_basis_count"]
    outside = _reported()
    outside["leap_basis_count"] = inside
    assert not case.validate(outside).success


def test_the_rule_bites_on_a_bill_with_no_leap_day_in_its_life():
    bills = case._bills()
    row = bills.loc[bills.issue_date.eq("2023-03-07") & bills.security_term_week_year.eq("4-Week")]
    assert len(row) == 1 and int(row.year_days.iloc[0]) == 366
    assert row.maturity_date.iloc[0] < "2024-02-29"


def test_a_bill_issued_on_the_leap_day_takes_365():
    bills = case._bills()
    leap = bills.loc[bills.issue_date.eq("2024-02-29")]
    assert len(leap) > 0 and leap.year_days.eq(365).all()


def test_the_published_price_follows_from_the_discount_rate_on_360_days():
    bills = case._bills()
    price = (100 * (1 - bills.high_discnt_rate / 100 * bills.days / 360)).round(6)
    assert price.eq(bills.high_price.round(6)).all()


def test_the_price_must_be_used_as_published():
    """Recomputed without its six-decimal rounding the price misses one bill."""
    bills = case._bills()
    short = bills.days.le(case.SHORT_BILL_DAYS)
    price = 100 * (1 - bills.high_discnt_rate / 100 * bills.days / 360)
    rate = (100 - price) / price * bills.year_days / bills.days * 100
    assert int(rate.round(3).eq(bills.high_investment_rate.round(3))[short].sum()) < int(short.sum())


# --- properties (no literals) ----------------------------------------------

def test_the_investment_rate_is_never_below_the_discount_rate():
    assert case._bills().gap_bp.ge(0).all()


def test_the_difference_widens_with_term():
    answers = case.analysis()
    gaps = [answers[f"median_gap_{w}_week_bp"] for w in (4, 13, 26, 52)]
    assert gaps == sorted(gaps) and len(set(gaps)) == len(gaps)


def test_the_day_part_is_term_neutral_and_the_price_part_is_not():
    answers = case.analysis()
    day_4, day_26 = answers["day_part_4_week_bp"], answers["day_part_26_week_bp"]
    price_4, price_26 = answers["price_part_4_week_bp"], answers["price_part_26_week_bp"]
    assert abs(day_26 - day_4) < 0.1 * day_4
    assert price_26 > 3 * price_4


def test_only_the_investment_rate_sits_on_the_curve():
    answers = case.analysis()
    for weeks in (26, 52):
        discount = answers[f"discount_less_curve_{weeks}_week_bp"]
        investment = answers[f"investment_less_curve_{weeks}_week_bp"]
        assert discount < -10
        assert abs(investment) < 0.1 * abs(discount)


def test_every_auction_has_its_investment_rate_closer_to_the_curve():
    frame = case._against_curve()
    assert (frame.investment_less_curve_bp.abs() < frame.discount_less_curve_bp.abs()).all()


def test_the_interpolation_rule_binds_on_the_even_groups():
    bills = case._bills()
    for term in ("13-Week", "26-Week"):
        values = np.sort(bills.loc[bills.security_term_week_year.eq(term), "gap_bp"].to_numpy())
        assert len(values) % 2 == 0
        assert values[len(values) // 2 - 1] != values[len(values) // 2]


# --- the conventions the query states ----------------------------------------

def test_grouping_by_original_term_is_a_different_answer():
    bills = case._bills()
    reopened = bills.security_term_week_year.ne(bills.original_security_term.astype(str))
    assert int(reopened.sum()) > 0
    by_original = float(
        bills.loc[bills.original_security_term.astype(str).eq("26-Week"), "gap_bp"].median()
    )
    outside = _reported()
    outside["median_gap_26_week_bp"] = round(by_original, 2)
    assert not case.validate(outside).success


def test_grouping_by_day_count_is_a_different_answer():
    """One weekly offering runs to several day counts, so reading 26 weeks as
    182 days drops part of the group and moves the median."""
    bills = case._bills()
    outside = _reported()
    for weeks, days in ((13, "91-Day"), (26, "182-Day"), (52, "364-Day")):
        by_days = float(
            bills.loc[bills.security_term_day_month.astype(str).eq(days), "gap_bp"].median()
        )
        outside[f"median_gap_{weeks}_week_bp"] = round(by_days, 2)
    assert not case.validate(outside).success


def test_every_auction_date_has_a_curve_point():
    frame = case._against_curve()
    assert frame.curve.notna().all()


# --- rejections -------------------------------------------------------------

def test_strict_display_boundary_both_sides():
    answers = case.analysis()
    reported = _reported()
    for variable in case.variables:
        precision = case.PRECISION[variable.name]
        for direction in (1, -1):
            outside = dict(reported)
            step = 0.7 * 10 ** (-precision) if precision else 1
            outside[variable.name] = answers[variable.name] + direction * step
            assert not case.validate(outside).success, f"{variable.name} {direction:+d}"


# --- source dependence ------------------------------------------------------

@pytest.mark.parametrize("missing", ["treasury_auctions", "treasury_yield_curve"])
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
