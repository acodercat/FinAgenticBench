"""Case-local gate for fails_to_deliver_exposure_measure_divergence.

This is a trap, so the gate has to prove three things a baseline would not need:
the naive path still produces the documented wrong number, the validator's
tolerance still excludes it, and the query still withholds the distinction that
avoids it. It also pins the financial content — that the quantity and value
rankings name different securities — because a data refresh that collapsed them
would leave a case that looks fine and measures nothing.
"""

import json

import numpy as np
import pandas as pd

from config import PROJECT_ROOT
from core.validation import numeric_equal
from evals.risk_screening import fails_to_deliver_exposure_measure_divergence as case
from scripts.audit_case_design import has_closed_text_representation


SPEC = json.loads(
    (PROJECT_ROOT / "evals/risk_screening/fails_to_deliver_exposure_measure_divergence.json").read_text()
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


def _cohort_balances():
    dates, later, cohort = case._cohort()
    return dates, later.loc[later.symbol.isin(cohort)]


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["recurring_symbol_count"] == 3694
    assert truth["later_half_settlement_date_count"] == 126
    assert truth["first_half_settlement_date_count"] == 124
    assert truth["largest_average_balance_symbol"] == "SIRI"
    assert round(truth["largest_average_balance_shares"], 4) == 3226336.2778
    assert round(truth["largest_average_balance_margin_pct"], 4) == 17.4323
    assert truth["dollar_exposure_leader_symbol"] == "QQQ"
    assert truth["dollar_exposure_runner_up_symbol"] == "SPY"
    assert truth["share_leader_dollar_exposure_rank"] == 15
    assert truth["cohort_symbols_with_share_count"] == 511
    assert truth["leading_dollar_symbols_with_share_count"] == 4


def test_the_naive_denominator_still_produces_the_documented_wrong_numbers():
    """The trap: a frame-level mean divides by dates present, not by the window's dates."""
    dates, held = _cohort_balances()
    naive = held.groupby("symbol").balance.mean().sort_values(ascending=False)
    correct = (held.groupby("symbol").balance.sum() / dates).sort_values(ascending=False)
    truth = _truth()

    leader = truth["largest_average_balance_symbol"]
    assert round(float(naive[leader]), 4) == 3946780.3010
    assert round(100.0 * (float(naive.iloc[0]) / float(naive.iloc[1]) - 1.0), 4) == 1.4707
    assert round(float(correct[leader]), 4) == round(truth["largest_average_balance_shares"], 4)

    # The tolerance must exclude both wrong figures rather than merely differ from them.
    assert not numeric_equal(float(naive[leader]), truth["largest_average_balance_shares"], decimals=4)
    assert not numeric_equal(1.4707, truth["largest_average_balance_margin_pct"], decimals=4)


def test_the_trap_is_confined_to_the_first_turn():
    """Later turns rest on sums, which the denominator cannot touch."""
    dates, held = _cohort_balances()
    by_sum = held.groupby("symbol").balance.sum()
    by_value = held.groupby("symbol").valued.sum()
    # Dividing by a different denominator rescales every average by the same shape
    # only within a symbol, so the summed quantities used downstream are untouched.
    assert by_sum.equals(held.groupby("symbol").balance.sum())
    assert by_value.notna().all()
    truth = _truth()
    assert round(float(by_value.max()) / 1e6, 4) == round(truth["dollar_exposure_leader_usd_millions"], 4)


def test_the_query_withholds_the_distinction_that_avoids_the_trap():
    """P5: a trap whose query states the avoidance is a baseline."""
    joined = " ".join(turn["query"] for turn in _turns()).lower()
    for giveaway in ("denominator", "divide by", "dates present", "absent", "zero balance",
                     "every settlement date in the window"):
        assert giveaway not in joined, giveaway
    # The first turn must not hand the denominator over as one of its own outputs.
    assert "later_half_settlement_date_count" not in _turns()[0]["stores"]
    assert "how many settlement dates the window holds" not in _turns()[0]["query"]
    assert SPEC["case_type"] == "trap"
    # The avoidance lives in a conditional claim, which the model never sees.
    conditional = [
        claim for turn in _turns() for claim in turn["claims"]
        if claim["id"] == "absent_date_treatment"
    ]
    assert conditional and conditional[0]["mode"] == "conditional"


def test_quantity_and_value_name_different_securities():
    """The financial point. If these ever coincide the case stops measuring anything."""
    truth = _truth()
    assert truth["largest_average_balance_symbol"] != truth["dollar_exposure_leader_symbol"]
    assert truth["share_leader_dollar_exposure_rank"] > 5


def test_the_share_count_denominator_is_missing_where_exposure_concentrates():
    truth = _truth()
    assert (
        truth["dollar_weighted_share_count_coverage_pct"]
        < truth["share_weighted_share_count_coverage_pct"]
    ), "coverage no longer falls as exposure rises"
    assert truth["leading_dollar_symbols_with_share_count"] < 10


def test_no_rank_the_case_uses_is_tied():
    dates, held = _cohort_balances()
    averages = (held.groupby("symbol").balance.sum() / dates).sort_values(ascending=False)
    values = held.groupby("symbol").valued.sum().sort_values(ascending=False)
    assert float(averages.iloc[0]) != float(averages.iloc[1])
    assert float(values.iloc[0]) != float(values.iloc[1])
    assert float(values.iloc[case.LEADING_DOLLAR_COUNT - 1]) != float(values.iloc[case.LEADING_DOLLAR_COUNT])


def test_absent_prices_are_held_the_way_the_query_says():
    """A balance stands without a price; only its valuation needs one."""
    dates, held = _cohort_balances()
    unpriced = held.reported_price.isna()
    assert unpriced.any(), "the absent-price reading is no longer load bearing"
    # Dropping those rows would discard real balances and move the reported margin.
    kept = held.loc[~unpriced]
    moved = (kept.groupby("symbol").balance.sum() / dates).sort_values(ascending=False)
    assert round(100.0 * (float(moved.iloc[0]) / float(moved.iloc[1]) - 1.0), 4) == 74.8327
    assert "whether or not the file reports a price" in _turns()[0]["query"]


def test_reported_averages_are_stable_under_summation_order():
    dates, held = _cohort_balances()
    values = held.loc[held.symbol.eq(_truth()["largest_average_balance_symbol"]), "balance"]
    array = values.to_numpy(dtype=float)
    generator = np.random.default_rng(0)
    sums = [float(np.sum(array[generator.permutation(len(array))])) for _ in range(8)]
    sums.append(float(np.sort(array).sum()))
    spread = (max(sums) - min(sums)) / dates
    assert spread < 0.6e-4 / 10, spread


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


def test_judge_reference_quotes_recomputed_figures():
    truth = _truth()
    facts = " ".join(SPEC["judge_reference"]["facts"])
    assert f"{truth['largest_average_balance_shares']:,.4f}" in facts
    assert f"{truth['dollar_exposure_leader_usd_millions']:,.4f}" in facts
    assert "3,946,780.3010" in facts
    assert truth["dollar_exposure_leader_symbol"] in facts


def test_conventions_the_query_leaves_out_change_no_output():
    """Whatever the query withholds must be provably idle, or it belongs in the query."""
    from core.data import load_financial_statements

    statements = load_financial_statements()
    frame = statements.loc[:, ["ticker", "form", "filed_date", "CommonStockSharesOutstanding"]].copy()
    del statements
    frame["ticker"] = frame.ticker.astype(str).str.upper()
    frame["filed_on"] = pd.to_datetime(frame.filed_date, errors="coerce")
    frame["shares"] = pd.to_numeric(frame.CommonStockSharesOutstanding, errors="coerce")
    frame = frame.loc[
        frame.shares.gt(0)
        & frame.filed_on.notna()
        & frame.filed_on.le(case.WINDOW_END)
        & ~frame.ticker.isin(("", "NAN", "NONE"))
    ]
    originals = frame.loc[~frame.form.astype(str).str.endswith("/A")]
    assert set(originals.ticker) != set(frame.ticker), "amendments no longer add any ticker"

    dates, later, cohort = case._cohort()
    held = later.loc[later.symbol.isin(cohort)]
    balances = held.groupby("symbol").balance.sum()
    values = case._ranked(held.groupby("symbol").valued.sum())

    def coverage(carried):
        with_count = sorted(symbol for symbol in cohort if symbol.upper() in carried)
        return (
            len(with_count),
            int(sum(1 for s in values.head(case.LEADING_DOLLAR_COUNT).index if s.upper() in carried)),
            round(100.0 * float(balances.reindex(with_count).sum()) / float(balances.sum()), 8),
            round(100.0 * float(values.reindex(with_count).fillna(0.0).sum()) / float(values.sum()), 8),
        )

    assert coverage(set(frame.ticker)) == coverage(set(originals.ticker))


def test_share_count_universe_respects_the_stated_vintage():
    universe = case._share_count_universe()
    assert universe
    from core.data import load_financial_statements

    statements = load_financial_statements()
    filed = pd.to_datetime(statements.filed_date, errors="coerce")
    assert filed.gt(case.WINDOW_END).any(), "no later filings exist, so the vintage rule is vacuous"

def test_no_variable_description_states_the_avoidance():
    """P5 names variable descriptions alongside the query as a prompt surface.

    A distinction stated in a description makes the case a baseline just as surely
    as one stated in the query, and the original guard only scanned the query.
    """
    trapped = {name for name in _turns()[0]["stores"]}
    for variable in case.variables:
        if variable.name not in trapped:
            continue
        lowered = variable.description.lower()
        for giveaway in ("a date the file omits", "omitted date", "nil balance", "126 dates",
                     "dates the file leaves out", "every settlement date in the window"):
            assert giveaway not in lowered, (variable.name, giveaway)
