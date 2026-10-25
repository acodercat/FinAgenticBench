"""Case-local gate for z1_corporate_equity_issuance_versus_market_value.

A trap needs three things proved: the naive paths still produce the documented
wrong numbers, the tolerance still excludes them, and the query still withholds
the distinction that avoids them. The financial content is pinned too — that the
sector retired equity while the market value grew — because a refresh that
reversed that would leave a case which looks fine and measures nothing.
"""

import json

import numpy as np
import pandas as pd

from config import PROJECT_ROOT
from core.validation import numeric_equal
from evals.calculation_reconciliation import (
    z1_corporate_equity_issuance_versus_market_value as case,
)
from scripts.audit_case_design import has_closed_text_representation


SPEC = json.loads(
    (PROJECT_ROOT / "evals/calculation_reconciliation/z1_corporate_equity_issuance_versus_market_value.json").read_text()
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["covered_quarter_count"] == 41
    assert truth["net_retirement_quarter_count"] == 33
    assert truth["identity_closing_quarter_count"] == 41
    assert truth["opposite_sign_quarter_count"] == 21
    assert truth["revaluation_exceeds_issuance_quarter_count"] == 37
    assert truth["matched_quarter_count"] == 41
    assert truth["sign_agreement_quarter_count"] == 39
    assert round(truth["summed_net_issuance_usd_millions"], 4) == -1050755.0
    assert round(truth["annual_rate_to_quarterly_ratio"], 4) == 4.0
    assert truth["largest_revaluation_quarter_end"] == "2022-06-30"


def test_both_naive_paths_still_produce_the_documented_wrong_numbers():
    changes = case._changes()
    truth = _truth()
    correct = truth["summed_net_issuance_usd_millions"]

    annual_rate_path = float(changes[case.ANNUAL_RATE_PREFIX].sum())
    level_change_path = float(
        changes[case.LEVEL_PREFIX].iloc[-1] - changes.opening.iloc[0]
    )
    assert round(annual_rate_path, 4) == -4203020.0
    assert round(level_change_path, 4) == 1689557.0

    # Exactly four times, and the other way round in sign.
    assert round(annual_rate_path / correct, 6) == 4.0
    assert np.sign(level_change_path) != np.sign(correct)

    # The tolerance must exclude both, not merely differ from them.
    assert not numeric_equal(annual_rate_path, correct, decimals=4)
    assert not numeric_equal(level_change_path, correct, decimals=4)


def test_the_annual_rate_series_is_exactly_four_times_the_quarterly_one():
    changes = case._changes()
    ratio = changes[case.ANNUAL_RATE_PREFIX] / changes[case.FLOW_PREFIX]
    assert ratio.round(8).eq(4.0).all(), ratio.describe()


def test_the_decomposition_closes_on_every_quarter():
    changes = case._changes()
    assert changes.residual.abs().max() == 0.0
    assert len(changes) == _truth()["identity_closing_quarter_count"]


def test_the_sector_retired_equity_while_the_market_value_grew():
    """The financial point. If this reverses the case stops measuring anything."""
    truth = _truth()
    changes = case._changes()
    assert truth["summed_net_issuance_usd_millions"] < 0
    assert float(changes[case.LEVEL_PREFIX].iloc[-1]) > float(changes.opening.iloc[0])
    assert truth["opposite_sign_quarter_count"] > len(changes) / 3


def test_the_revaluation_leg_tracks_the_market():
    truth = _truth()
    assert truth["revaluation_market_return_correlation"] > 0.95
    assert truth["median_absolute_gap_percentage_points"] < 2.0
    assert truth["sign_agreement_quarter_count"] >= 35


def test_the_query_withholds_the_distinction_that_avoids_the_trap():
    """P5: a trap whose query states the avoidance is a baseline."""
    opening = _turns()[0]["query"].lower()
    for giveaway in ("annual rate", "annualis", "annualiz", "four times",
                     "not the change in", "revaluation", "holding gain"):
        assert giveaway not in opening, giveaway
    assert SPEC["case_type"] == "trap"
    # The first turn names the four codes neutrally and says which is which nowhere.
    assert "103064103" in _turns()[0]["query"]
    conditional = [
        claim for turn in _turns() for claim in turn["claims"]
        if claim["id"] == "quarterly_flow_selection"
    ]
    assert conditional and conditional[0]["mode"] == "conditional"


def test_pinning_the_newest_release_changes_the_answer():
    """Cross-release differencing is a different question, owned by four other cases.

    The pin only means something if the releases disagree. They do, so a module
    that silently mixed them would report different numbers.
    """
    from core.data import load_expansion_table

    accounts = load_expansion_table("fed_z1_vintages")
    frame = accounts.loc[accounts.series.astype(str).str.slice(2).eq(case.SERIES_STEM)].copy()
    del accounts
    frame["vintage"] = frame.vintage.astype(str)
    frame["prefix"] = frame.series.astype(str).str.slice(0, 2)
    frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    frame["quarter_end"] = pd.to_datetime(frame.date)
    wide = frame.pivot_table(
        index=["prefix", "quarter_end"], columns="vintage", values="value", aggfunc="first"
    )
    releases = sorted(wide.columns)
    assert len(releases) > 1, "there is no longer more than one release to confuse"
    newest = releases[-1]
    disagreements = [
        int((wide[[earlier, newest]].dropna().diff(axis=1).iloc[:, 1] != 0).sum())
        for earlier in releases[:-1]
    ]
    assert min(disagreements) > 0, "the releases now agree, so the pin is decorative"

    # And the module reads that newest release, not an average or a mixture.
    legs = case._series()
    newest_only = wide.xs(case.LEVEL_PREFIX)[newest].dropna()
    assert legs[case.LEVEL_PREFIX].reindex(newest_only.index).equals(newest_only)


def test_repeated_statement_tables_carry_one_value():
    """The level is published in several tables at once; aggfunc=first must be safe."""
    from core.data import load_expansion_table

    accounts = load_expansion_table("fed_z1_vintages")
    frame = accounts.loc[
        accounts.vintage.astype(str).eq(accounts.vintage.astype(str).max())
        & accounts.series.astype(str).str.slice(2).eq(case.SERIES_STEM)
    ].copy()
    del accounts
    distinct = frame.groupby([frame.series.astype(str), frame.date.astype(str)]).value.nunique()
    assert distinct.max() == 1, "a series now reports two different values for one quarter"
    assert distinct.size > 0


def test_selection_margin_is_recorded_and_real():
    changes = case._changes()
    ranked = changes[case.REVALUATION_PREFIX].abs().sort_values(ascending=False)
    assert float(ranked.iloc[0]) != float(ranked.iloc[1]), "the largest revaluation is tied"
    margin = 100.0 * (ranked.iloc[0] / ranked.iloc[1] - 1.0)
    assert round(margin, 2) == 5.82, margin
    # The query states a tie rule; assert it is currently idle rather than silent.
    assert not ranked.duplicated().any()
    assert "breaking a tie by the earlier quarter" in _turns()[3]["query"]


def test_summed_money_outputs_are_stable_under_summation_order():
    changes = case._changes()
    array = changes[case.FLOW_PREFIX].to_numpy(dtype=float)
    generator = np.random.default_rng(0)
    sums = [float(np.sum(array[generator.permutation(len(array))])) for _ in range(8)]
    sums.append(float(np.sort(array).sum()))
    assert max(sums) - min(sums) < 0.6e-4 / 10


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
    assert f"{truth['summed_net_issuance_usd_millions']:,.4f}" in facts
    assert f"{truth['revaluation_market_return_correlation']:,.4f}" in facts
    assert "-4,203,020.0000" in facts
    assert truth["largest_revaluation_quarter_end"] in facts


def test_the_market_return_source_is_load_bearing():
    """Removing the factor file makes the third turn impossible, not merely thinner."""
    returns = case._quarterly_market_return()
    assert isinstance(returns, pd.Series) and not returns.empty
    accounts_only = set(case._changes().columns)
    assert not {"market_return"} & accounts_only

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
        for giveaway in ("annual rate", "annualis", "annualiz", "four times",
                     "not the change in", "revaluation", "holding gain"):
            assert giveaway not in lowered, (variable.name, giveaway)
