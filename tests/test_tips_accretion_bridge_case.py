"""Case-local gate for tips_accretion_stock_versus_daily_increment.

The financial point has to keep holding: the monthly identity must keep closing,
the maturity adjustment must keep collapsing the gap, and the two conventions the
query pins -- detail rows rather than class totals, and redemptions as negative
amounts -- must each keep changing an answer. Every assertion can fail.
"""

import json

import pandas as pd
import pytest

from config import PROJECT_ROOT
from core.data import load_expansion_table
from core.validation import numeric_equal
from evals.calculation_reconciliation import tips_accretion_stock_versus_daily_increment as case


SPEC = json.loads(
    (PROJECT_ROOT
     / "evals/calculation_reconciliation/tips_accretion_stock_versus_daily_increment.json"
     ).read_text()
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


@pytest.fixture(scope="module")
def detail():
    return case._tips_detail()


@pytest.fixture(scope="module")
def increments():
    return case._daily_increments()


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["tips_month_end_count"] == 84
    assert truth["tips_issue_row_count"] == 134
    assert truth["tips_cusip_count"] == 53
    assert truth["tips_outstanding_usd_billions"] == 2133.881769
    assert truth["accretion_share_of_outstanding_pct"] == 18.072992
    assert truth["cumulative_issued_usd_billions"] == 1757.996692
    assert truth["month_ends_with_a_departing_security_count"] == 22
    assert truth["departing_accretion_usd_billions"] == 204.679361
    assert truth["increment_day_count"] == 1757
    assert truth["negative_increment_day_count"] == 323
    assert truth["dts_increment_total_usd_billions"] == 457.434000
    assert truth["months_compared_count"] == 83
    assert truth["opposite_direction_month_count"] == 7
    assert truth["unadjusted_mean_absolute_gap_usd_millions"] == 2512.193653
    assert truth["adjusted_mean_absolute_gap_usd_millions"] == 48.235140
    assert truth["largest_adjusted_gap_usd_millions"] == 271.452971


def test_the_monthly_identity_closes_on_every_month_end(detail):
    """issued + accrued + redeemed = outstanding, in aggregate, every month."""
    grouped = detail.groupby("month", observed=True).agg(
        issued=("issued", "sum"), accrued=("accrued", "sum"),
        redeemed=("redeemed", "sum"), outstanding=("outstanding", "sum"),
    )
    residual = grouped.outstanding - (grouped.issued + grouped.accrued + grouped.redeemed)
    assert len(grouped) == _truth()["tips_month_end_count"]
    assert float(residual.abs().max()) < 1e-6, float(residual.abs().max())
    assert int(grouped.redeemed.gt(0).sum()) == 0, "redemptions are no longer stored as negatives"
    assert int(grouped.redeemed.lt(0).sum()) > 0


def test_the_maturity_adjustment_is_what_closes_the_bridge(detail, increments):
    """Without it the comparison is fifty times worse and reverses sign."""
    truth = _truth()
    by_security = detail.groupby(["month", "security"], observed=True).accrued.sum()
    stock = by_security.groupby(level="month").sum()
    months = sorted(stock.index)
    changes, departed = {}, {}
    for earlier, later in zip(months, months[1:]):
        before = by_security.loc[earlier]
        gone = before.loc[~before.index.isin(by_security.loc[later].index)]
        changes[later] = float(stock[later] - stock[earlier])
        departed[later] = float(gone.sum())
    flow = increments.groupby(increments.day.dt.to_period("M")).amount.sum()
    frame = pd.DataFrame({"change": pd.Series(changes), "departed": pd.Series(departed),
                          "flow": flow}).dropna()
    unadjusted = (frame.change - frame.flow).abs().mean()
    adjusted = (frame.change - (frame.flow - frame.departed)).abs().mean()
    assert round(float(unadjusted), 6) == truth["unadjusted_mean_absolute_gap_usd_millions"]
    assert round(float(adjusted), 6) == truth["adjusted_mean_absolute_gap_usd_millions"]
    assert unadjusted / adjusted > 50, "the adjustment no longer dominates the discrepancy"
    assert not numeric_equal(round(float(unadjusted), 6), truth["adjusted_mean_absolute_gap_usd_millions"], decimals=6)

    # July 2025 is the example the judge reference quotes.
    july = pd.Period("2025-07", freq="M")
    assert round(frame.change[july], 3) == -10197.547
    assert round(frame.flow[july], 0) == 4387
    assert round(frame.departed[july], 3) == 14438.975


def test_the_total_rows_are_present_and_would_double_count():
    """The query says to leave the class totals aside; they exist and they matter."""
    statement = load_expansion_table("treasury_marketable_securities")
    label = statement.security_identifier_or_total_label.astype(str)
    tips = statement.security_class.astype(str).eq(case.TIPS_CLASS)
    totals = tips & label.str.contains("otal", case=False, na=False)
    assert int(totals.sum()) > 0, (
        "the file no longer carries inflation-protected class totals, so the query's "
        "exclusion has gone idle and should be reconsidered"
    )
    outstanding = pd.to_numeric(statement.outstanding_million_usd, errors="coerce")
    month = pd.to_datetime(statement.record_date, errors="coerce").dt.to_period("M")
    final = month.max()
    with_totals = float(outstanding[tips & month.eq(final)].sum()) / 1000.0
    assert round(with_totals, 6) != _truth()["tips_outstanding_usd_billions"]
    assert "published class totals aside" in _turns()[0]["query"]


def test_the_row_and_security_counts_really_differ(detail):
    truth = _truth()
    final = detail.month.max()
    last = detail.loc[detail.month.eq(final)]
    assert truth["tips_issue_row_count"] > truth["tips_cusip_count"], (
        "every security is now listed once, so the first turn's required claim is empty"
    )
    assert int(last.outstanding.notna().sum()) == truth["tips_cusip_count"]
    repeated = last.groupby("security").size()
    assert int(repeated.gt(1).sum()) > 0


def test_the_increment_line_falls_as_well_as_rises(increments):
    truth = _truth()
    assert len(increments) == truth["increment_day_count"]
    assert int(increments.amount.lt(0).sum()) == truth["negative_increment_day_count"] > 0, (
        "no increment is negative any more, so the third turn's required claim is unsupported"
    )
    assert float(increments.amount.min()) < 0 < float(increments.amount.max())


@pytest.mark.parametrize("removed", ["treasury_marketable_securities", "treasury_dts_public_debt"])
def test_both_sources_are_load_bearing(monkeypatch, removed):
    original = case.load_expansion_table

    def without(name):
        if name == removed:
            raise FileNotFoundError("removed source")
        return original(name)

    case.ground_truth.cache_clear()
    try:
        with monkeypatch.context() as ctx:
            ctx.setattr(case, "load_expansion_table", without)
            with pytest.raises(FileNotFoundError):
                case.ground_truth()
    finally:
        case.ground_truth.cache_clear()


def test_stores_partition_the_module_variables():
    declared = [name for turn in _turns() for name in turn["stores"]]
    assert declared == [variable.name for variable in case.variables]
    assert len(declared) == len(set(declared))


def test_every_turn_validator_rejects_a_wrong_value():
    truth = _truth()
    for turn in _turns():
        checker = case.validators[turn["validator"]]
        outputs = {name: truth[name] for name in turn["stores"]}
        assert case._validate_subset(outputs, turn["stores"]).success
        assert checker is not None
        for name in turn["stores"]:
            broken = dict(outputs)
            broken[name] = float(broken[name]) + 1.0
            assert not case._validate_subset(broken, turn["stores"]).success, (turn["validator"], name)


def test_claim_contract_is_well_formed():
    for turn in _turns():
        assert turn["claims"]
        for claim in turn["claims"]:
            if claim["mode"] == "conditional":
                assert claim["description"].startswith("If ")
            if claim["mode"] == "required":
                assert " does not " not in claim["description"].lower()
            if "overlap_waiver" in claim:
                assert claim["overlap_waiver"].strip()


def test_judge_reference_quotes_recomputed_figures():
    truth = _truth()
    facts = " ".join(SPEC["judge_reference"]["facts"])
    assert f"{truth['tips_outstanding_usd_billions']:,.6f}" in facts
    assert f"{truth['accretion_share_of_outstanding_pct']:.6f}" in facts
    assert f"{truth['unadjusted_mean_absolute_gap_usd_millions']:,.6f}" in facts
    assert f"{truth['adjusted_mean_absolute_gap_usd_millions']:.6f}" in facts
    assert "14,438.975" in facts
    assert set(SPEC["judge_reference"]) == {"source", "facts", "known_overclaims"}
    assert SPEC["judge_reference"]["known_overclaims"]


def test_every_departing_security_has_actually_matured(detail):
    """The required claim says securities take their accrued principal away at maturity.

    That is an assertion about why they leave, so it is verified against the
    maturity date rather than inferred from the disappearance.
    """
    statement = load_expansion_table("treasury_marketable_securities")
    maturity = pd.to_datetime(statement.maturity_date, errors="coerce")
    frame = detail.copy()
    frame["matures"] = maturity.loc[frame.index]
    by_security = frame.groupby(["month", "security"], observed=True).agg(
        accrued=("accrued", "sum"), matures=("matures", "max")
    )
    months = sorted(frame.month.unique())
    departures, matured = 0, 0
    for earlier, later in zip(months, months[1:]):
        before = by_security.loc[earlier]
        gone = before.loc[~before.index.isin(by_security.loc[later].index)]
        for _, row in gone.iterrows():
            departures += 1
            if pd.notna(row.matures) and pd.Period(row.matures, freq="M") <= later:
                matured += 1
    assert departures == 23
    assert matured == departures, (
        "a security now leaves the statement without having matured, so the fourth "
        "turn's required claim no longer describes what is happening"
    )


def test_only_one_reading_of_the_scope_reproduces_the_bridge(detail, increments):
    """Detail rows versus class totals, adjusted versus not: four readings, one answer."""
    truth = _truth()
    by_security = detail.groupby(["month", "security"], observed=True).accrued.sum()
    stock = by_security.groupby(level="month").sum()
    months = sorted(stock.index)
    changes, departed = {}, {}
    for earlier, later in zip(months, months[1:]):
        before = by_security.loc[earlier]
        gone = before.loc[~before.index.isin(by_security.loc[later].index)]
        changes[later] = float(stock[later] - stock[earlier])
        departed[later] = float(gone.sum())
    flow = increments.groupby(increments.day.dt.to_period("M")).amount.sum()

    statement = load_expansion_table("treasury_marketable_securities")
    tips = statement.security_class.astype(str).eq(case.TIPS_CLASS)
    accrued = pd.to_numeric(statement.inflation_adjustment_million_usd, errors="coerce")
    month = pd.to_datetime(statement.record_date, errors="coerce").dt.to_period("M")
    with_totals = accrued[tips].groupby(month[tips]).sum().diff().dropna()

    readings = {}
    for label, series in (("detail", pd.Series(changes)), ("totals", with_totals)):
        frame = pd.DataFrame({"change": series, "departed": pd.Series(departed),
                              "flow": flow}).dropna()
        readings[(label, "adjusted")] = round(
            float((frame.change - (frame.flow - frame.departed)).abs().mean()), 6)
        readings[(label, "raw")] = round(float((frame.change - frame.flow).abs().mean()), 6)

    assert readings[("detail", "adjusted")] == truth["adjusted_mean_absolute_gap_usd_millions"]
    assert readings[("detail", "raw")] == truth["unadjusted_mean_absolute_gap_usd_millions"]
    assert readings[("totals", "adjusted")] == 6558.365136
    assert readings[("totals", "raw")] == 8256.006105
    reproducing = sum(v == truth["adjusted_mean_absolute_gap_usd_millions"] for v in readings.values())
    assert reproducing == 1, readings


def test_the_two_files_cover_the_same_months(detail, increments):
    """83 compared months is 84 minus the first, not a coverage gap."""
    truth = _truth()
    statement_months = set(detail.month.unique())
    daily_months = set(increments.day.dt.to_period("M").unique())
    assert statement_months == daily_months
    assert len(statement_months) == truth["tips_month_end_count"]
    assert truth["months_compared_count"] == truth["tips_month_end_count"] - 1


def test_no_security_leaves_and_returns(detail):
    """A reporting gap would make the departure adjustment count the same accrual twice."""
    by_security = detail.groupby(["month", "security"], observed=True).accrued.sum()
    months = sorted(detail.month.unique())
    last_seen, returns = {}, 0
    for index, month in enumerate(months):
        for security in by_security.loc[month].index:
            if security in last_seen and last_seen[security] != months[index - 1]:
                returns += 1
            last_seen[security] = month
    assert returns == 0, "a security now reappears after a gap, so departures double count"
