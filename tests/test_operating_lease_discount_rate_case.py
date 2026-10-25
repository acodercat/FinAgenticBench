"""Boundaries for the operating lease discount-rate lag case.

Anchors pin the literals. Property tests name no literal, so each can fail on
its own rather than being decided by an equality asserted earlier in the same
function.
"""

import pandas as pd
import pytest

from core.data import load_expansion_table, load_facts
from evals.temporal_path import operating_lease_discount_rate_lag as case


def _clear():
    for cached in (case._balances, case._yield_on, case.analysis, case.ground_truth):
        cached.cache_clear()


@pytest.fixture
def restore():
    yield
    _clear()


def _reconciled():
    frames = {item: case._year(item, case.FOCUS_YEAR) for item in ("payments", "interest", "liability")}
    joined = pd.concat({k: v[["end_date", "value"]] for k, v in frames.items()}, axis=1).dropna()
    same_date = joined[("payments", "end_date")].eq(joined[("liability", "end_date")]) & joined[
        ("interest", "end_date")
    ].eq(joined[("liability", "end_date")])
    return joined.loc[same_date]


def _carried():
    joined = _reconciled()
    exact = (joined[("payments", "value")] - joined[("interest", "value")]).eq(joined[("liability", "value")])
    rates = case._rates_against_yield(case.FOCUS_YEAR, joined.index[exact])
    return rates.loc[rates.end_date.eq(joined.loc[rates.index, ("liability", "end_date")])]


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["schedule_filer_count"] == 372
    assert answers["schedule_exact_count"] == 359
    assert answers["negative_interest_count"] == 1
    assert round(answers["median_interest_share_pct"], 4) == 17.0856
    assert answers["rate_filer_count"] == 339
    assert round(answers["median_discount_rate_pct"], 2) == 4.30
    assert answers["below_treasury_count"] == 166
    assert round(answers["median_spread_to_treasury_bp"], 2) == 2.00
    assert answers["first_majority_below_year"] == 2022
    assert answers["below_treasury_2021_count"] == 1
    assert answers["paired_filer_count"] == 330
    assert round(answers["median_rate_change_bp"], 2) == 100.00
    assert round(answers["median_yield_change_bp"], 2) == 312.00
    assert answers["rate_kept_pace_count"] == 13
    assert answers["asset_filer_count"] == 328
    assert answers["asset_below_liability_count"] == 281
    assert round(answers["median_asset_to_liability"], 4) == 0.9348


# --- independent recomputation ---------------------------------------------

def test_schedule_counts_recomputed_from_raw_facts():
    facts = load_facts()
    names = {
        "LesseeOperatingLeaseLiabilityPaymentsDue": "p",
        "LesseeOperatingLeaseLiabilityUndiscountedExcessAmount": "x",
        "OperatingLeaseLiability": "l",
    }
    rows = facts.loc[facts.concept.astype(str).isin(list(names)) & facts.form.astype(str).eq("10-K")]
    rows = rows.loc[rows.start_date.isna() & rows.value.notna()].copy()
    rows["k"] = rows.concept.astype(str).map(names)
    rows = rows.sort_values(["filed_date", "accession"]).groupby(["cik", "k", "end_date"]).head(1)
    rows = rows.loc[rows.end_date.str.startswith(str(case.FOCUS_YEAR))]
    rows = rows.sort_values("end_date").groupby(["cik", "k"]).tail(1)
    wide = rows.pivot(index="cik", columns="k", values=["end_date", "value"]).dropna()
    wide = wide.loc[wide[("end_date", "p")].eq(wide[("end_date", "l")]) & wide[("end_date", "x")].eq(wide[("end_date", "l")])]
    answers = case.analysis()
    assert len(wide) == answers["schedule_filer_count"]
    assert int((wide[("value", "p")] - wide[("value", "x")]).eq(wide[("value", "l")]).sum()) == answers["schedule_exact_count"]


def test_yield_lookup_uses_latest_published_date_on_or_before():
    curve = load_expansion_table("treasury_yield_curve")
    curve = curve.assign(date=curve.Date.astype(str)).dropna(subset=[case.TENOR]).sort_values("date")
    lookup = case._yield_on()
    for day in ("2024-12-28", "2024-12-31", "2021-12-31"):
        expected = float(curve.loc[curve.date.le(day), case.TENOR].iloc[-1])
        assert lookup(day) == expected


# --- properties (no literals) ----------------------------------------------

def test_same_balance_date_requirement_drops_no_filer():
    frames = {item: case._year(item, case.FOCUS_YEAR) for item in ("payments", "interest", "liability")}
    tagged = pd.concat({k: v[["end_date", "value"]] for k, v in frames.items()}, axis=1).dropna()
    joined = _reconciled()
    assert len(tagged) == len(joined)
    exact = joined.index[(joined[("payments", "value")] - joined[("interest", "value")]).eq(joined[("liability", "value")])]
    rates = case._year("rate", case.FOCUS_YEAR)
    assert rates.index.isin(exact).sum() == len(_carried())
    assets = case._year("asset", case.FOCUS_YEAR)
    assert assets.index.isin(_carried().index).sum() == case.analysis()["asset_filer_count"]


def test_boundary_ties_do_not_move_the_majority_year():
    carried = _carried().index
    year = case.analysis()["first_majority_below_year"]
    for earlier in range(min(case.PATH_YEARS), year + 1):
        frame = case._rates_against_yield(earlier, carried)
        strict = (frame.rate_pct < frame.yield_pct).sum() * 2 > len(frame)
        weak = (frame.rate_pct <= frame.yield_pct).sum() * 2 > len(frame)
        assert strict == weak


def test_wrong_scale_rates_move_no_output(restore):
    base = case.analysis()
    rows = case._balances()
    wrong = rows.item.eq("rate") & rows.value.ge(1)
    assert wrong.any()
    for variant in (rows.loc[~wrong], rows.assign(value=rows.value.where(~wrong, rows.value / 100))):
        case._balances.cache_clear()
        case.analysis.cache_clear()
        original = case._balances
        case._balances = lambda variant=variant: variant
        try:
            moved = case.analysis.__wrapped__()
        finally:
            case._balances = original
        for name, precision in case.PRECISION.items():
            assert round(moved[name], precision) == round(base[name], precision), name


def test_rate_changes_do_not_tie_with_yield_changes():
    early = case._rates_against_yield(case.BASE_YEAR, _carried().index)
    late = case._rates_against_yield(case.FOCUS_YEAR, early.index)
    paired = early.index.intersection(late.index)
    gap = (late.loc[paired, "rate_pct"] - early.loc[paired, "rate_pct"]) - (
        late.loc[paired, "yield_pct"] - early.loc[paired, "yield_pct"]
    )
    assert gap.abs().min() > 0.005


def test_majority_year_is_not_a_knife_edge():
    carried = _carried().index
    year = case.analysis()["first_majority_below_year"]
    rates = case._rates_against_yield(year, carried)
    below = int((rates.rate_pct < rates.yield_pct).sum())
    assert below > 0.6 * len(rates)
    for earlier in range(min(case.PATH_YEARS), year):
        frame = case._rates_against_yield(earlier, carried)
        assert (frame.rate_pct < frame.yield_pct).sum() < 0.05 * len(frame)


def test_many_rates_sit_below_every_tenor():
    curve = load_expansion_table("treasury_yield_curve")
    curve = curve.assign(date=curve.Date.astype(str)).sort_values("date")
    tenors = [c for c in curve.columns if c not in ("Date", "date")]
    rates = _carried()

    def lowest(day):
        return float(pd.to_numeric(curve.loc[curve.date.le(day)].iloc[-1][tenors], errors="coerce").min())

    under_curve = sum(rate < lowest(day) for rate, day in zip(rates.rate_pct, rates.end_date))
    assert under_curve > 0.75 * case.analysis()["below_treasury_count"]


# --- alternative readings move outputs ---------------------------------------

def test_absolute_interest_reading_adds_a_reconciliation():
    joined = _reconciled()
    signed = (joined[("payments", "value")] - joined[("interest", "value")]).eq(joined[("liability", "value")])
    unsigned = (joined[("payments", "value")] - joined[("interest", "value")].abs()).eq(joined[("liability", "value")])
    assert int(unsigned.sum()) == int(signed.sum()) + case.analysis()["negative_interest_count"]


def test_ten_year_comparison_changes_the_count(monkeypatch, restore):
    base = case.analysis()["below_treasury_count"]
    monkeypatch.setattr(case, "TENOR", "10 Yr")
    _clear()
    assert case.analysis()["below_treasury_count"] != base


def test_latest_report_reading_moves_outputs(restore):
    facts = load_facts()
    rows = facts.loc[
        facts.concept.astype(str).isin(list(case.CONCEPTS))
        & facts.form.astype(str).eq("10-K")
        & facts.start_date.isna()
        & facts.value.notna()
    ].copy()
    rows["item"] = rows.concept.astype(str).map(case.CONCEPTS)
    rows = rows.sort_values(["filed_date", "accession"]).drop_duplicates(["cik", "item", "end_date"], keep="last")
    rows["year"] = rows.end_date.str[:4].astype(int)
    rows = rows.sort_values("end_date").drop_duplicates(["cik", "item", "year"], keep="last")
    base = case.analysis()
    original = case._balances
    case._balances = lambda: rows[["cik", "item", "year", "end_date", "value"]]
    try:
        latest = case.analysis.__wrapped__()
    finally:
        case._balances = original
    assert latest["schedule_exact_count"] != base["schedule_exact_count"]
    assert latest["below_treasury_count"] != base["below_treasury_count"]


# --- validator ---------------------------------------------------------------

def test_validator_accepts_truth_and_rejects_a_shift():
    truth = dict(zip((v.name for v in case.variables), case.ground_truth()))
    assert case.validate(truth).success
    for name, precision in case.PRECISION.items():
        shifted = dict(truth)
        shifted[name] = truth[name] + (1 if precision == 0 else 10 ** -precision)
        assert not case.validate(shifted).success, name
