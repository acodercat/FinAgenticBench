"""Case-local gate for fund_redemption_month_alignment_and_sign.

Two traps have to keep working. The sign of the redemption field must stay
filer-chosen and widespread, and the three reported months must stay ordered
earliest-first, with the market able to settle which is which. Both wrong paths
are recomputed here against the tolerance, and every prompt surface -- the queries
and the variable descriptions alike -- is scanned for the distinction that would
turn either trap into a baseline.
"""

import json

import numpy as np
import pandas as pd
import pytest

from config import PROJECT_ROOT
from core.validation import numeric_equal
from evals.risk_screening import fund_redemption_month_alignment_and_sign as case
from scripts.audit_case_design import has_closed_text_representation


SPEC = json.loads(
    (PROJECT_ROOT / "evals/risk_screening/fund_redemption_month_alignment_and_sign.json").read_text()
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["retained_filing_count"] == 290709
    assert truth["distinct_series_count"] == 16698
    assert truth["lowest_return_month_end_date"] == "2020-03-31"
    assert truth["filings_covering_that_month"] == 11597
    assert truth["filings_redeeming_over_one_tenth_count"] == 2431
    assert truth["period_net_inflow_despite_stress_count"] == 503
    assert truth["first_reported_month_offset"] == 2
    assert round(truth["total_gross_redeemed_usd_billions"], 4) == 53220.9308
    assert round(truth["month_gross_redeemed_usd_billions"], 4) == 1552.7608
    assert round(truth["month_net_share_flow_usd_billions"], 4) == -392.7647


def _restrict_to_month(filings):
    """The module's covering population: one filing per series, earliest period end."""
    selected = pd.Period(_truth()["lowest_return_month_end_date"], "M")
    slot = case._slot_for(filings, selected)
    covering = filings.loc[slot.notna()].copy()
    covering["reported_slot"] = slot[slot.notna()]
    covering = (covering.sort_values(["series_id", "period_end"])
                .drop_duplicates(subset=["series_id"], keep="first")
                .reset_index(drop=True))
    return covering, covering.reported_slot


def _covering():
    return _restrict_to_month(case._filings())


def _signed_filings():
    """The same retained population, but with the redemption field left as filed."""
    raw = case.load_expansion_table("sec_nport")
    frame = raw.loc[:, [
        "accession", "filing_date", "report_date", "series_id", "net_assets_usd",
        *[f"{f}_month_{m}_usd"
          for f in ("sales", "redemptions", "realized_gain", "unrealized_change")
          for m in case.MONTHS],
    ]].copy()
    del raw
    frame["filed_on"] = pd.to_datetime(frame.filing_date)
    frame["period_end"] = pd.to_datetime(frame.report_date)
    frame = frame.loc[frame.series_id.notna()]
    frame = (frame.sort_values(["filed_on", "accession"])
             .drop_duplicates(subset=["series_id", "period_end"], keep="last")
             .reset_index(drop=True))
    built = {}
    for m in case.MONTHS:
        built[f"sold_{m}"] = pd.to_numeric(frame[f"sales_month_{m}_usd"], errors="coerce")
        built[f"redeemed_{m}"] = pd.to_numeric(frame[f"redemptions_month_{m}_usd"], errors="coerce")
        built[f"gain_{m}"] = pd.to_numeric(frame[f"realized_gain_month_{m}_usd"], errors="coerce") \
            + pd.to_numeric(frame[f"unrealized_change_month_{m}_usd"], errors="coerce")
    built["net_assets"] = pd.to_numeric(frame.net_assets_usd, errors="coerce")
    frame = frame.assign(**built).dropna(subset=list(built))
    frame = frame.loc[frame.net_assets.gt(0)].reset_index(drop=True)
    frame["period_month"] = frame.period_end.dt.to_period("M")
    return frame


def test_the_sign_trap_still_understates_by_a_third():
    """Recomputed from the file, not asserted from the docstring."""
    truth = _truth()
    signed = _signed_filings()
    assert len(signed) == truth["retained_filing_count"], "the two populations diverged"

    observations = pd.concat([signed[f"redeemed_{m}"] for m in case.MONTHS])
    negative = int(observations.lt(0).sum())
    assert negative > 100_000, negative
    assert observations.lt(0).mean() > 0.15

    naive = float(sum(signed[f"redeemed_{m}"] for m in case.MONTHS).sum() / case.BILLION)
    correct = truth["total_gross_redeemed_usd_billions"]
    assert round(naive, 4) == 35417.3573, naive
    assert round(100 * (naive / correct - 1), 4) == -33.4522
    assert not numeric_equal(naive, correct, decimals=4)
    # Sales are summed as reported: a negative sale reverses one, so only the
    # redemption field is read as a magnitude.
    sold = float(sum(signed[f"sold_{m}"] for m in case.MONTHS).sum() / case.BILLION)
    assert numeric_equal(sold, truth["total_gross_sold_usd_billions"], decimals=4)
    magnitude = float(sum(signed[f"sold_{m}"].abs() for m in case.MONTHS).sum() / case.BILLION)
    assert not numeric_equal(magnitude, truth["total_gross_sold_usd_billions"], decimals=4)


def test_both_wrong_paths_taken_together_are_also_excluded():
    """The combined path is recomputed, and it is the furthest of the three."""
    truth = _truth()
    signed, signed_slot = _restrict_to_month(_signed_filings())
    reversed_slot = signed_slot.map({1: 3, 2: 2, 3: 1})
    combined = float(case._pick(signed, "redeemed", reversed_slot).sum() / case.BILLION)
    assert round(combined, 4) == 396.2804, combined
    assert not numeric_equal(combined, truth["month_gross_redeemed_usd_billions"], decimals=4)
    assert combined < 660.5342 < 888.2351 < truth["month_gross_redeemed_usd_billions"]


def test_both_population_rules_change_the_answer_and_are_stated():
    """A rule the query spends words on has to earn them; a rule it omits must be idle."""
    truth = _truth()
    raw = case.load_expansion_table("sec_nport")
    fields = [f"{f}_month_{m}_usd" for f in
              ("sales", "redemptions", "realized_gain", "unrealized_change")
              for m in case.MONTHS]
    frame = raw.loc[:, ["series_id", "report_date", "filing_date", "accession",
                        "net_assets_usd", *fields]].copy()
    del raw

    # 1. Filings without a series identifier are a real population, not an empty rule.
    without = int(frame.series_id.isna().sum())
    assert without > 1000, without
    assert "report a fund series identifier" in _turns()[0]["query"]

    # 2. Taking the surviving filing whole differs from a column-wise retention.
    frame["filed_on"] = pd.to_datetime(frame.filing_date)
    frame["period_end"] = pd.to_datetime(frame.report_date)
    kept = frame.loc[frame.series_id.notna()].sort_values(["filed_on", "accession"])
    whole = kept.drop_duplicates(subset=["series_id", "period_end"], keep="last")
    spliced = kept.groupby(["series_id", "period_end"], as_index=False).last()
    assert len(whole) == len(spliced)
    ordered = ["series_id", "period_end"]
    left = whole.sort_values(ordered).reset_index(drop=True)
    right = spliced.sort_values(ordered).reset_index(drop=True)
    differing = sum(
        int((left[field].isna().to_numpy() != right[field].isna().to_numpy()).sum())
        for field in fields
    )
    assert differing > 0, "a column-wise retention no longer splices, so the clause is idle"
    assert "taking that filing whole" in _turns()[0]["query"]
    assert truth["retained_filing_count"] == len(case._filings())


def test_the_slot_trap_still_reverses_the_three_reported_months():
    truth = _truth()
    covering, slot = _covering()
    correct = case._pick(covering, "redeemed", slot)
    reversed_slot = slot.map({1: 3, 2: 2, 3: 1})
    wrong = case._pick(covering, "redeemed", reversed_slot)

    assert round(float(wrong.sum() / case.BILLION), 4) == 888.2351
    assert round(100 * float(wrong.sum() / correct.sum() - 1), 4) == -42.7964
    assert not numeric_equal(
        float(wrong.sum() / case.BILLION), truth["month_gross_redeemed_usd_billions"], decimals=4
    )
    # Not a coverage artifact: both readings draw on exactly the same filings.
    assert len(correct) == len(wrong) == truth["filings_covering_that_month"]
    assert slot.value_counts().sort_index().tolist() == (
        reversed_slot.value_counts().sort_index().tolist()[::-1]
    )


def test_the_market_settles_the_ordering_and_the_off_diagonal_stays_small():
    truth = _truth()
    filings = case._filings()
    returns = case._market_month_returns()
    table = {}
    for month in case.MONTHS:
        rate = filings[f"gain_{month}"] / filings.net_assets
        for offset in case.OFFSETS:
            aligned = (filings.period_month - offset).dt.to_timestamp("M").map(returns)
            table[(month, offset)] = float(rate.corr(aligned, method="spearman"))
    diagonal = {1: 2, 2: 1, 3: 0}
    for month, offset in diagonal.items():
        assert table[(month, offset)] > 0.6, (month, offset, table[(month, offset)])
        others = [table[(month, o)] for o in case.OFFSETS if o != offset]
        assert max(others) < 0.1, (month, others)
    assert round(table[(1, 2)], 4) == round(truth["first_reported_month_rank_correlation"], 4)
    assert round(table[(3, 0)], 4) == round(truth["third_reported_month_rank_correlation"], 4)


def test_the_rank_statistic_is_the_reason_the_diagnostic_survives():
    """A product-moment correlation on the same untrimmed data shows nothing."""
    filings = case._filings()
    returns = case._market_month_returns()
    rate = filings["gain_1"] / filings.net_assets
    aligned = (filings.period_month - 2).dt.to_timestamp("M").map(returns)
    keep = rate.notna() & aligned.notna()
    pearson = float(np.corrcoef(rate[keep], aligned[keep])[0, 1])
    assert abs(pearson) < 0.2, pearson
    spearman = float(rate.corr(aligned, method="spearman"))
    assert spearman > 0.6
    assert "Spearman" in _turns()[3]["query"]


def test_the_accession_tie_rule_changes_the_answer():
    """406 pairs share a filing date and 17 of them disagree, so the rule is not decoration."""
    truth = _truth()
    signed = _signed_filings()
    raw = case.load_expansion_table("sec_nport")
    fields = [f"{f}_month_{m}_usd" for f in
              ("sales", "redemptions", "realized_gain", "unrealized_change") for m in case.MONTHS]
    frame = raw.loc[raw.series_id.notna(),
                    ["series_id", "report_date", "filing_date", "accession",
                     "net_assets_usd", *fields]].copy()
    del raw
    frame["filed_on"] = pd.to_datetime(frame.filing_date)
    frame["period_end"] = pd.to_datetime(frame.report_date)

    grouped = frame.groupby(["series_id", "report_date"])
    shared = grouped.size()[grouped.size().gt(1) & grouped.filing_date.nunique().lt(grouped.size())]
    assert len(shared) > 100, "filings no longer share a date within a series period"

    def total(accession_ascending):
        kept = frame.sort_values(
            ["filed_on", "accession"], ascending=[True, accession_ascending]
        ).drop_duplicates(subset=["series_id", "period_end"], keep="last")
        keep = pd.to_numeric(kept.net_assets_usd, errors="coerce").gt(0)
        for field in fields:
            keep &= pd.to_numeric(kept[field], errors="coerce").notna()
        kept = kept.loc[keep]
        return float(sum(
            pd.to_numeric(kept[f"redemptions_month_{m}_usd"], errors="coerce").abs()
            for m in case.MONTHS
        ).sum() / case.BILLION)

    later, earlier = total(True), total(False)
    assert round(later, 4) == round(truth["total_gross_redeemed_usd_billions"], 4)
    assert not numeric_equal(earlier, later, decimals=4), (earlier, later)
    assert "breaking a tie by the later accession number" in _turns()[0]["query"]


def test_the_rank_tie_rule_is_load_bearing_and_pinned():
    """Ties dominate one side of the pairing, so the ranking method changes the answer."""
    truth = _truth()
    filings = case._filings()
    returns = case._market_month_returns()
    rate = filings["gain_1"] / filings.net_assets
    aligned = (filings.period_month - 2).dt.to_timestamp("M").map(returns)
    keep = rate.notna() & aligned.notna()
    rate, aligned = rate[keep], aligned[keep]
    assert aligned.nunique() < 200, "the market column no longer carries heavy ties"

    pinned = truth["first_reported_month_rank_correlation"]
    assert abs(float(rate.corr(aligned, method="spearman")) - pinned) < 1e-12
    for method in ("min", "max", "first", "dense"):
        alternative = float(
            np.corrcoef(rate.rank(method=method), aligned.rank(method=method))[0, 1]
        )
        assert not numeric_equal(alternative, pinned, decimals=4), method
    assert "tied values taking their average rank" in _turns()[3]["query"]


def test_the_retention_rule_is_not_idle():
    raw = case.load_expansion_table("sec_nport")
    frame = raw.loc[:, ["series_id", "report_date", "redemptions_month_3_usd"]].copy()
    del raw
    counts = frame.groupby(["series_id", "report_date"]).size()
    assert int(counts.gt(1).sum()) > 1000, "no series period is filed more than once any more"
    repeated = counts[counts.gt(1)].index
    disagreeing = (
        frame.set_index(["series_id", "report_date"])
        .loc[repeated]
        .groupby(level=[0, 1])
        .redemptions_month_3_usd.nunique()
    )
    assert int(disagreeing.gt(1).sum()) > 0, "repeated filings now always agree"


def test_no_prompt_surface_states_either_avoidance():
    """P5 names the query and the variable descriptions alike.

    The reveal turn may name the offsets it asks about; nothing before it may say
    which reported month is which, and nothing anywhere may say how to read the
    redemption field's sign.
    """
    sign_giveaways = ("absolute value", "absolute", "magnitude", "regardless of sign",
                      "sign convention", "negative", "unsigned", "ignore the sign")
    order_giveaways = ("most recent month", "two months before", "two months earlier",
                       "ends the period", "earliest of the three", "third reported month is",
                       "first reported month is")
    surfaces = [turn["query"] for turn in _turns()]
    surfaces += [variable.description for variable in case.variables]
    for surface in surfaces:
        lowered = surface.lower()
        for giveaway in sign_giveaways:
            assert giveaway not in lowered, (giveaway, surface)
    for turn in _turns()[:3]:
        lowered = turn["query"].lower()
        for giveaway in order_giveaways:
            assert giveaway not in lowered, (giveaway, turn["validator"])
    assert SPEC["case_type"] == "trap"


def test_every_selection_has_a_real_margin():
    truth = _truth()
    returns = case._market_month_returns()
    filings = case._filings()
    covered = [
        period for period in case._covered_months(filings)
        if period.to_timestamp("M") in returns.index
    ]
    monthly = returns.reindex([period.to_timestamp("M") for period in covered]).sort_values()
    assert round(float(monthly.iloc[1] - monthly.iloc[0]), 4) == 3.7683
    covering, slot = _covering()
    ranked = case._pick(covering, "redeemed", slot).sort_values(ascending=False)
    assert round(100 * float(ranked.iloc[0] / ranked.iloc[1] - 1), 4) == 88.9458
    assert round(float(ranked.iloc[0] / case.BILLION), 4) == round(
        truth["largest_filing_month_redemption_usd_billions"], 4
    )


def test_the_financial_finding_survives():
    """Aggregation hides the stress. If it stopped, the case would measure nothing."""
    truth = _truth()
    assert 0.3 < truth["median_month_share_of_period"] < 0.7
    assert truth["period_net_inflow_despite_stress_count"] > 0
    assert truth["period_net_inflow_despite_stress_count"] < truth[
        "filings_redeeming_over_one_tenth_count"
    ]
    assert truth["month_net_share_flow_usd_billions"] < 0, (
        "the selected month no longer shows a net outflow, so the sign trap's "
        "contradiction with the episode is gone"
    )


def test_summed_money_outputs_are_stable_under_summation_order():
    covering, slot = _covering()
    array = case._pick(covering, "redeemed", slot).to_numpy(dtype=float)
    generator = np.random.default_rng(0)
    sums = [float(np.sum(array[generator.permutation(len(array))])) for _ in range(8)]
    sums.append(float(np.sort(array).sum()))
    assert (max(sums) - min(sums)) / case.BILLION < 0.6e-4 / 10


def test_every_text_output_declares_a_closed_representation():
    truth = _truth()
    by_name = {variable.name: variable for variable in case.variables}
    text_outputs = [name for name, value in truth.items() if isinstance(value, str)]
    assert text_outputs
    for name in text_outputs:
        assert has_closed_text_representation(by_name[name].description), name


def test_stores_partition_the_module_variables():
    declared = [name for turn in _turns() for name in turn["stores"]]
    assert declared == [variable.name for variable in case.variables]
    assert len(declared) == len(set(declared))


def test_every_turn_validator_rejects_a_wrong_value():
    truth = _truth()
    for turn in _turns():
        checker = getattr(case, turn["validator"])
        outputs = {name: truth[name] for name in turn["stores"]}
        assert checker(dict(outputs)).success
        broken = dict(outputs)
        first = turn["stores"][0]
        broken[first] = (
            "__WRONG__" if isinstance(broken[first], str) else float(broken[first]) + 1_000_000.0
        )
        assert not checker(broken).success, turn["validator"]


def test_claim_contract_is_well_formed():
    for turn in _turns():
        for claim in turn["claims"]:
            if claim["mode"] == "conditional":
                assert claim["description"].startswith("If ")
            if claim["mode"] == "required":
                assert " does not " not in claim["description"].lower()
            if "overlap_waiver" in claim:
                assert claim["overlap_waiver"].strip()


def test_judge_reference_carries_the_full_contract():
    reference = SPEC["judge_reference"]
    assert set(reference) == {"source", "facts", "known_overclaims"}
    assert reference["source"].strip() and reference["facts"]
    assert len(reference["known_overclaims"]) >= 4
    facts = " ".join(reference["facts"])
    truth = _truth()
    assert f"{truth['month_gross_redeemed_usd_billions']:,.4f}" in facts
    assert "35,417.3573" in facts
    assert "888.2351" in facts
    assert truth["lowest_return_month_end_date"] in facts


def test_the_factor_source_is_load_bearing():
    """Without it there is no month to select and no way to settle the ordering."""
    returns = case._market_month_returns()
    assert isinstance(returns, pd.Series) and len(returns) > 100
    filings = case._filings()
    assert not {"total", "mkt_rf_pct"} & set(filings.columns)


def _clear_case_caches():
    for name in ("_filings", "_market_month_returns", "ground_truth"):
        getattr(case, name).cache_clear()


@pytest.mark.parametrize("removed", ["sec_nport", "ken_french_daily_factors"])
def test_both_sources_are_load_bearing(monkeypatch, removed):
    """Enforce the contract by removing the source, rather than asserting around it."""
    original_expansion = case.load_expansion_table
    original_french = case.load_ken_french_table

    def expansion(name):
        if name == removed:
            raise FileNotFoundError("removed source")
        return original_expansion(name)

    def french(name):
        if removed == "ken_french_daily_factors":
            raise FileNotFoundError("removed source")
        return original_french(name)

    _clear_case_caches()
    try:
        with monkeypatch.context() as ctx:
            ctx.setattr(case, "load_expansion_table", expansion)
            ctx.setattr(case, "load_ken_french_table", french)
            with pytest.raises(FileNotFoundError):
                case.ground_truth()
    finally:
        _clear_case_caches()
