"""Boundaries for the pension discount-rate remeasurement case.

Anchors pin the literals. Property tests name no literal, so each can fail on
its own rather than being decided by an equality asserted earlier in the same
function.
"""

import pandas as pd
import pytest

from core.data import load_expansion_table
from evals.temporal_path import pension_discount_rate_remeasurement as case


def _clear():
    for cached in (case._facts, case._yield_on, case.analysis, case.ground_truth):
        cached.cache_clear()


@pytest.fixture
def restore():
    yield
    _clear()


def _with_facts(rows):
    original = case._facts
    case._facts = lambda: rows
    try:
        return case.analysis.__wrapped__()
    finally:
        case._facts = original


def _raw(any_presentation=False):
    facts = load_expansion_table("sec_notes_dimensional")
    rows = facts.loc[
        facts.tag.astype(str).isin(list(case.TAGS))
        & facts.form.astype(str).eq("10-K")
        & facts.axis.astype(str).eq(case.AXIS)
        & facts.member.astype(str).eq(case.MEMBER)
        & facts.value.notna()
    ]
    return rows if any_presentation else rows.loc[rows.iprx.eq(case.PRIMARY_PRESENTATION)]


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["rate_filer_count"] == 124
    assert answers["invalid_rate_count"] == 1
    assert round(answers["median_rate_change_bp"], 2) == 250.00
    assert round(answers["median_yield_change_bp"], 2) == 236.00
    assert answers["tracking_count"] == 118
    assert round(answers["median_end_rate_pct"], 2) == 5.20
    assert answers["below_treasury_count"] == 4
    assert round(answers["median_spread_bp"], 2) == 132.00
    assert answers["gain_filer_count"] == 112
    assert answers["negative_gain_count"] == 20
    assert answers["negative_gain_obligation_fell_count"] == 20
    assert round(answers["median_implied_duration_years"], 4) == 9.0457
    assert answers["funded_filer_count"] == 108
    assert answers["negative_return_count"] == 108
    assert answers["negative_return_improved_count"] == 71
    assert round(answers["median_start_funded_ratio"], 4) == 0.9495
    assert round(answers["median_end_funded_ratio"], 4) == 0.9794


# --- properties (no literals) ----------------------------------------------

def test_primary_presentation_leaves_one_value_per_filing():
    rows = _raw()
    assert rows.groupby(["cik", "tag", "ddate", "qtrs", "adsh"]).value.nunique().le(1).all()
    assert _raw(any_presentation=True).groupby(["cik", "tag", "ddate", "qtrs", "adsh"]).value.nunique().gt(1).any()


def test_yield_lookup_uses_latest_published_date_on_or_before():
    curve = load_expansion_table("treasury_yield_curve")
    curve = curve.assign(date=curve.Date.astype(str)).dropna(subset=[case.TENOR]).sort_values("date")
    lookup = case._yield_on()
    for day in ("2021-12-31", "2022-09-30", "2022-12-31"):
        assert lookup(day) == float(curve.loc[curve.date.le(day), case.TENOR].iloc[-1])


def test_tracking_band_has_no_boundary_case():
    start, end = case._year("rate", case.START_YEAR), case._year("rate", case.END_YEAR)
    both = start.index.intersection(end.index)
    lookup = case._yield_on()
    gaps = []
    for cik in both:
        s, e = float(start.at[cik, "value"]), float(end.at[cik, "value"])
        if 0 < s < 1 and 0 < e < 1:
            dy = (lookup(end.at[cik, "end_date"]) - lookup(start.at[cik, "end_date"])) * 100
            gaps.append(abs((e - s) * 10_000 - dy))
    # Rates and yields are whole basis points here, so only a gap of exactly the
    # band would make the inclusive reading matter.
    assert min(abs(g - case.TRACKING_BAND_BP) for g in gaps) > 0.5


def test_every_negative_gain_comes_with_a_falling_obligation():
    answers = case.analysis()
    assert answers["negative_gain_obligation_fell_count"] == answers["negative_gain_count"]


def test_funded_position_improves_for_most_losing_plans():
    answers = case.analysis()
    assert answers["negative_return_improved_count"] * 2 > answers["negative_return_count"]
    assert answers["median_end_funded_ratio"] > answers["median_start_funded_ratio"]


def test_rates_track_the_ten_year_far_better_than_lease_style_lag():
    answers = case.analysis()
    valid = answers["rate_filer_count"] - answers["invalid_rate_count"]
    assert answers["tracking_count"] > 0.9 * valid
    assert answers["below_treasury_count"] < 0.05 * valid


# --- alternative readings move outputs ---------------------------------------

def _rebuild(rows, keep):
    rows = rows.copy()
    rows["item"] = rows.tag.astype(str).map(case.TAGS)
    rows = rows.loc[rows.qtrs.eq(rows.item.isin(case.INSTANTS).map({True: 0, False: 4}))]
    rows["end_date"] = pd.to_datetime(rows.ddate.astype(str)).dt.strftime("%Y-%m-%d")
    rows["year"] = rows.end_date.str[:4].astype(int)
    rows = rows.sort_values(["filed", "adsh"]).drop_duplicates(["cik", "item", "end_date"], keep=keep)
    rows = rows.sort_values("end_date").drop_duplicates(["cik", "item", "year"], keep="last")
    return rows[["cik", "item", "year", "end_date", "value"]]


def test_latest_report_reading_moves_outputs():
    base = case.analysis()
    latest = _with_facts(_rebuild(_raw(), "last"))
    assert latest["invalid_rate_count"] != base["invalid_rate_count"]
    assert latest["gain_filer_count"] != base["gain_filer_count"]


def test_absolute_value_of_the_invalid_rate_moves_outputs():
    base = case.analysis()
    rows = case._facts().copy()
    is_rate = rows.item.eq("rate")
    rows.loc[is_rate, "value"] = rows.loc[is_rate, "value"].abs()
    repaired = _with_facts(rows)
    assert repaired["invalid_rate_count"] < base["invalid_rate_count"]
    assert repaired["tracking_count"] != base["tracking_count"]


def test_thirty_year_comparison_moves_tracking(monkeypatch, restore):
    base = case.analysis()["tracking_count"]
    monkeypatch.setattr(case, "TENOR", "30 Yr")
    _clear()
    assert case.analysis()["tracking_count"] != base


# --- validator ---------------------------------------------------------------

def test_validator_accepts_truth_and_rejects_a_shift():
    truth = dict(zip((v.name for v in case.variables), case.ground_truth()))
    assert case.validate(truth).success
    for name, precision in case.PRECISION.items():
        shifted = dict(truth)
        shifted[name] = truth[name] + (1 if precision == 0 else 10 ** -precision)
        assert not case.validate(shifted).success, name
