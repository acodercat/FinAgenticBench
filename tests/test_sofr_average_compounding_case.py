"""Boundaries for the SOFR average compounding and bill basis case.

Anchors pin the literals. Property tests name no literal, so each can fail on
its own rather than being decided by an equality asserted earlier in the same
function.
"""

import numpy as np
import pandas as pd
import pytest

from evals.calculation_reconciliation import sofr_average_compounding_and_bill_basis as case


def _clear():
    for cached in (case._rates, case._calendar, case._publications, case._spreads,
                   case.analysis, case.ground_truth):
        cached.cache_clear()


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] else answers[v.name]
        for v in case.variables
    }


def _reproduced(average):
    frame = case._publications()
    return {
        length: sum(
            abs(round(average(date, length) + 1e-12, 5) - value) < 1e-9
            for date, value in frame[column].items()
        )
        for length, column in case.AVERAGE_COLUMN.items()
    }


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["publication_date_count"] == 748
    assert round(answers["unrounded_30_day_average_pct"], 7) == 3.7865938
    assert round(answers["unrounded_180_day_average_pct"], 7) == 4.1999627
    assert round(answers["median_premium_30_day_bp"], 4) == 0.9102
    assert round(answers["median_premium_180_day_bp"], 4) == 5.8227
    assert round(answers["max_premium_180_day_bp"], 4) == 7.0790
    assert answers["first_cut_date"] == "2024-09-19"
    assert round(answers["median_raw_spread_before_bp"], 3) == -18.626
    assert round(answers["median_adjusted_spread_before_bp"], 3) == -11.232
    assert round(answers["median_raw_spread_after_bp"], 3) == 0.262
    assert round(answers["median_adjusted_spread_after_bp"], 3) == 6.313
    assert answers["first_clean_30_day_date"] == "2024-10-21"
    assert answers["first_clean_180_day_date"] == "2025-03-18"
    assert round(answers["thirty_less_180_day_on_clean_date_bp"], 3) == -45.488


# --- the convention the data decides ---------------------------------------

def test_the_reproducing_calculation_matches_every_publication():
    count = len(case._publications())
    assert _reproduced(case.compounded_average) == {30: count, 90: count, 180: count}


def _calendar_compounding(date, length):
    rate, _ = case._window(date, length)
    return (np.prod(1 + rate / case.DAY_BASIS) - 1) * case.DAY_BASIS / length * 100


def _year_of_365(date, length):
    rate, business = case._window(date, length)
    opens = business.copy()
    opens[0] = True
    starts = np.flatnonzero(opens)
    spans = np.diff(np.append(starts, len(rate)))
    return (np.prod(1 + rate[starts] * spans / 365) - 1) * 365 / length * 100


@pytest.mark.parametrize("competing", [_calendar_compounding, _year_of_365, case.simple_average])
def test_no_competing_calculation_reproduces_any_publication(competing):
    assert sum(_reproduced(competing).values()) == 0


def test_the_unrounded_values_separate_the_readings():
    answers = case.analysis()
    for competing in (_calendar_compounding, _year_of_365, case.simple_average):
        outside = _reported()
        outside["unrounded_30_day_average_pct"] = round(competing(case.REPORT_DATE, 30), 7)
        outside["unrounded_180_day_average_pct"] = round(competing(case.REPORT_DATE, 180), 7)
        assert not case.validate(outside).success
    assert answers["unrounded_30_day_average_pct"] != answers["unrounded_180_day_average_pct"]


def test_the_index_steps_the_same_way():
    """The published SOFR Index grows by one simple step per business day."""
    sofr, averages, _ = case._rates()
    index = averages["SOFR Index"].astype(float)
    dates = list(index.index)
    for previous, current in zip(dates[:-1], dates[1:]):
        days = (pd.Timestamp(current) - pd.Timestamp(previous)).days
        expected = 1 + sofr[previous] / 100 * days / case.DAY_BASIS
        assert abs(index[current] / index[previous] - expected) < 5e-7


# --- properties (no literals) ----------------------------------------------

def test_compounding_always_adds_and_grows_with_the_window():
    answers = case.analysis()
    assert 0 < answers["median_premium_30_day_bp"] < answers["median_premium_180_day_bp"]
    assert answers["median_premium_180_day_bp"] < answers["max_premium_180_day_bp"]


def test_the_basis_restatement_lifts_both_spreads():
    answers = case.analysis()
    for side in ("before", "after"):
        lift = answers[f"median_adjusted_spread_{side}_bp"] - answers[f"median_raw_spread_{side}_bp"]
        assert 4 < lift < 9


def test_the_spread_changes_sign_across_the_first_cut():
    answers = case.analysis()
    assert answers["median_adjusted_spread_before_bp"] < 0 < answers["median_adjusted_spread_after_bp"]


def test_the_interpolation_rule_binds_on_the_full_sample_only():
    frame = case._publications()
    assert len(frame) % 2 == 0
    for length in (30, 180):
        values = np.sort([
            case.compounded_average(d, length) - case.simple_average(d, length) for d in frame.index
        ])
        assert values[len(values) // 2 - 1] != values[len(values) // 2]
    spreads = case._spreads()
    cut = case.analysis()["first_cut_date"]
    assert (spreads.index < cut).sum() % 2 == 1
    assert (spreads.index >= cut).sum() % 2 == 1


def test_every_publication_date_has_a_curve_point():
    assert case._spreads().notna().all().all()


def test_the_longer_window_passes_the_cut_through_later():
    answers = case.analysis()
    assert answers["first_cut_date"] < answers["first_clean_30_day_date"] < answers["first_clean_180_day_date"]
    assert answers["thirty_less_180_day_on_clean_date_bp"] < 0


def test_the_clean_dates_are_the_first_after_the_cut():
    """The publication before each clean date still opens on a pre-cut fixing."""
    cut = case.analysis()["first_cut_date"]
    averages = case._rates()[1]
    dates = list(averages.index)
    for length, key in ((30, "first_clean_30_day_date"), (180, "first_clean_180_day_date")):
        clean = case.analysis()[key]
        assert case._window_opening_rate_date(clean, length) >= cut
        before = dates[dates.index(clean) - 1]
        assert case._window_opening_rate_date(before, length) < cut


def test_the_published_figure_is_a_different_premium():
    frame = case._publications()
    published = pd.Series([
        (frame.loc[d, case.AVERAGE_COLUMN[180]] - case.simple_average(d, 180)) * case.BASIS_POINTS
        for d in frame.index
    ])
    outside = _reported()
    outside["median_premium_180_day_bp"] = round(float(published.median()), 4)
    outside["max_premium_180_day_bp"] = round(float(published.max()), 4)
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


def test_the_cut_date_is_matched_verbatim():
    outside = _reported()
    outside["first_cut_date"] = "2024-09-18"
    assert not case.validate(outside).success


# --- source dependence ------------------------------------------------------

@pytest.mark.parametrize("missing", ["nyfed_reference_rates", "treasury_yield_curve"])
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
