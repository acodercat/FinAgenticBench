"""Boundaries for the TIPS reference-CPI inflation trap.

Anchors pin the literals. Property tests name no literal, so each can fail on
its own rather than being decided by an equality asserted earlier in the same
function.
"""

import numpy as np

from core.data import load_expansion_table
from evals.temporal_path import tips_reference_cpi_lag_inflation_recovery as case


def _within(value, truth, decimals):
    return abs(value - truth) <= 0.6 * 10 ** -decimals


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["july_2025_index_level"] == 323.048
    assert round(answers["july_2025_twelve_month_pct"], 2) == 2.70
    assert round(answers["adjusted_most_recent_twelve_month_pct"], 4) == 2.7426
    assert round(answers["adjusted_first_release_twelve_month_pct"], 4) == 2.7688
    assert answers["revised_month_count"] == 12
    assert answers["september_2025_index_level"] == 324.800
    assert round(answers["september_2025_twelve_month_pct"], 2) == 3.01
    assert answers["october_2025_index_level"] == 325.604


# --- the trap: wrong paths fall outside the tolerance ------------------------

def test_same_month_reference_values_miss_the_level_and_the_rate():
    ref = case._reference_cpi()
    answers = case.analysis()
    for day in ("2025-07-15", "2025-07-31"):
        assert not _within(ref[day], answers["july_2025_index_level"], 3)
    for later, earlier in (("2025-07-15", "2024-07-15"), ("2025-07-31", "2024-07-31")):
        naive = (ref[later] / ref[earlier] - 1) * 100
        assert not _within(naive, answers["july_2025_twelve_month_pct"], 2)


def test_adjusted_shortcuts_miss_the_unadjusted_rate():
    answers = case.analysis()
    realtime = load_expansion_table("philadelphia_fed_rtdsm")
    month = realtime.loc[
        realtime.series.astype(str).eq(case.ADJUSTED_SERIES) & realtime.period.astype(str).eq("2025:07")
    ]
    for column in ("first_release", "most_recent"):
        assert not _within(float(month[column].iloc[0]), answers["july_2025_twelve_month_pct"], 2)
    for name in ("adjusted_most_recent_twelve_month_pct", "adjusted_first_release_twelve_month_pct"):
        assert not _within(answers[name], answers["july_2025_twelve_month_pct"], 2)


# --- properties (no literals) ----------------------------------------------

def test_two_routes_to_september_2024_agree():
    october_2024, _ = case._solve_pair("2025-01-15", "2025-01-31")
    via_december = case._earlier_from_later("2024-12-31", october_2024)
    _, august_2024 = case._solve_pair("2024-10-15", "2024-10-31")
    via_november = case._later_from_earlier("2024-11-29", august_2024)
    assert round(via_december, 3) == round(via_november, 3)


def test_recovered_levels_sit_away_from_rounding_boundaries():
    july_2025, august_2025 = case._solve_pair("2025-10-15", "2025-10-31")
    september_2025 = case._later_from_earlier("2025-11-28", august_2025)
    october_2025 = case._later_from_earlier("2025-12-31", september_2025)
    for level in (july_2025, september_2025, october_2025):
        assert abs(level * 1000 - round(level * 1000)) < 0.1


def test_october_2025_equals_the_missing_index_substitute():
    october_2024, _ = case._solve_pair("2025-01-15", "2025-01-31")
    september_2024 = round(case._earlier_from_later("2024-12-31", october_2024), 3)
    answers = case.analysis()
    september_2025 = answers["september_2025_index_level"]
    substitute = september_2025 * (september_2025 / september_2024) ** (1 / 12)
    assert round(substitute, 3) == answers["october_2025_index_level"]


def test_rates_from_unrounded_levels_agree_at_two_decimals():
    july_2024, _ = case._solve_pair("2024-10-15", "2024-10-31")
    july_2025, _ = case._solve_pair("2025-10-15", "2025-10-31")
    unrounded = (july_2025 / july_2024 - 1) * 100
    assert round(unrounded, 2) == round(case.analysis()["july_2025_twelve_month_pct"], 2)


def test_averaging_annualized_changes_differs_from_compounding():
    realtime = load_expansion_table("philadelphia_fed_rtdsm")
    window = realtime.loc[
        realtime.series.astype(str).eq(case.ADJUSTED_SERIES)
        & realtime.period.astype(str).between(*case.ADJUSTED_WINDOW)
    ]
    averaged = float(np.mean(window.most_recent.astype(float)))
    assert not _within(averaged, case.analysis()["adjusted_most_recent_twelve_month_pct"], 4)


def test_reopening_dated_dates_are_not_read():
    auctions = load_expansion_table("treasury_auctions")
    tips = auctions.loc[auctions.inflation_index_security.astype(str).eq("Yes")]
    reopenings = tips.loc[tips.reopening.astype(str).eq("Yes")].dropna(subset=["ref_cpi_on_dated_date"])
    ref = case._reference_cpi()
    mislabeled = [
        str(day) for day, value in zip(reopenings.dated_date, reopenings.ref_cpi_on_dated_date)
        if str(day) not in ref or ref[str(day)] != float(value)
    ]
    assert mislabeled


# --- validator ---------------------------------------------------------------

def test_validator_accepts_truth_and_rejects_a_shift():
    truth = dict(zip((v.name for v in case.variables), case.ground_truth()))
    assert case.validate(truth).success
    for name, precision in case.PRECISION.items():
        shifted = dict(truth)
        shifted[name] = truth[name] + (1 if precision == 0 else 10 ** -precision)
        assert not case.validate(shifted).success, name
