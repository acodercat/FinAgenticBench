"""Case-local gate for sba_variable_rate_ceiling_passthrough.

What has to keep holding: the rate file has to stay the only way to form a
spread, the variable group has to stay far more concentrated than the fixed one,
the two tiers have to stay several points apart, and each convention the query
spends words on has to keep moving an answer.
"""

import json

import pandas as pd
import pytest

from config import PROJECT_ROOT
from core.validation import numeric_equal
from evals.comparison_ranking import sba_variable_rate_ceiling_passthrough as case
from scripts.audit_case_design import has_closed_text_representation


SPEC = json.loads(
    (PROJECT_ROOT / "evals/comparison_ranking/sba_variable_rate_ceiling_passthrough.json").read_text()
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["matched_loan_count"] == 281056
    assert truth["variable_rate_loan_count"] == 239767
    assert truth["fixed_rate_loan_count"] == 41289
    assert truth["smallest_tier_loan_count"] == 60285
    assert truth["covered_year_count"] == 5
    assert round(truth["variable_modal_spread_pct"], 2) == 5.92
    assert round(truth["smallest_tier_modal_spread_pct"], 2) == 9.67
    assert round(truth["largest_tier_modal_spread_pct"], 2) == 5.92
    assert round(truth["variable_top_five_spread_share_pct"], 4) == 46.3004
    assert round(truth["fixed_top_five_spread_share_pct"], 4) == 15.0960
    assert round(truth["modal_rate_range_pp"], 2) == 5.25
    assert round(truth["policy_rate_range_pp"], 4) == 5.2800


@pytest.mark.parametrize("removed", ["sba_7a", "nyfed_reference_rates"])
def test_removing_either_source_makes_the_case_impossible(removed, monkeypatch):
    """Strong source removal, enforced rather than asserted."""
    original = case.load_expansion_table

    def blocked(name):
        if name == removed:
            raise FileNotFoundError(f"removed source {name}")
        return original(name)

    for cached in (case._matched, case._policy_rates, case.ground_truth):
        cached.cache_clear()
    try:
        with monkeypatch.context() as context:
            context.setattr(case, "load_expansion_table", blocked)
            with pytest.raises(FileNotFoundError):
                case.ground_truth()
    finally:
        for cached in (case._matched, case._policy_rates, case.ground_truth):
            cached.cache_clear()


def test_the_variable_group_stacks_and_the_fixed_one_does_not():
    """The case's finding. If the two converged it would measure nothing."""
    truth = _truth()
    assert truth["variable_top_five_spread_share_pct"] > 2.5 * truth[
        "fixed_top_five_spread_share_pct"
    ]
    matched = case._matched()
    variable = matched.loc[matched.kind.eq(case.VARIABLE)]
    fixed = matched.loc[matched.kind.eq(case.FIXED)]
    # Fewer distinct spreads in absolute terms, on 5.8 times the rows. A multiple
    # of the fixed count would pass even if the variable group were far more spread.
    assert variable.spread.nunique() < fixed.spread.nunique(), (
        variable.spread.nunique(), fixed.spread.nunique())
    assert len(variable) > 4 * len(fixed)
    # Concentration, not merely a different centre.
    assert float(variable.spread.value_counts(normalize=True).iloc[0]) > 3 * float(
        fixed.spread.value_counts(normalize=True).iloc[0]
    )


def test_the_fixed_mode_is_too_close_to_select_on():
    """Why the case reports five-value coverage instead of the fixed mode."""
    matched = case._matched()
    fixed = matched.loc[matched.kind.eq(case.FIXED)]
    counts = fixed.spread.value_counts()
    margin = 100.0 * float(counts.iloc[0] - counts.iloc[1]) / float(len(fixed))
    assert margin < 0.5, margin
    stores = [name for turn in _turns() for name in turn["stores"]]
    assert not any("fixed_modal" in name for name in stores)


def test_the_two_tiers_stay_apart_and_each_mode_has_a_margin():
    truth = _truth()
    matched = case._matched()
    variable = matched.loc[matched.kind.eq(case.VARIABLE)]
    edges = (0.0,) + case.TIER_EDGES + (float("inf"),)
    tier = pd.cut(variable.approval, bins=edges, labels=False, right=True)
    smallest = variable.loc[tier.eq(0)]
    largest = variable.loc[tier.eq(len(case.TIER_EDGES))]
    assert abs(truth["smallest_tier_modal_spread_pct"]
               - truth["largest_tier_modal_spread_pct"]) > 3.0
    for group, label in ((smallest, "smallest"), (largest, "largest")):
        counts = group.spread.value_counts()
        margin = 100.0 * float(counts.iloc[0] - counts.iloc[1]) / float(len(group))
        assert margin > 1.0, (label, margin)


def test_every_tie_rule_is_stated_and_currently_idle():
    """A rule that never binds still has to be written, and its idleness asserted."""
    matched = case._matched()
    variable = matched.loc[matched.kind.eq(case.VARIABLE)]
    fixed = matched.loc[matched.kind.eq(case.FIXED)]
    edges = (0.0,) + case.TIER_EDGES + (float("inf"),)
    tier = pd.cut(variable.approval, bins=edges, labels=False, right=True)

    groups = {
        "variable": variable, "fixed": fixed,
        "smallest": variable.loc[tier.eq(0)],
        "largest": variable.loc[tier.eq(len(case.TIER_EDGES))],
    }
    for label, group in groups.items():
        counts = group.spread.value_counts()
        assert int(counts.iloc[0]) != int(counts.iloc[1]), f"{label}: the mode is tied"
        assert int(counts.iloc[case.TOP_VALUES - 1]) != int(counts.iloc[case.TOP_VALUES]), (
            f"{label}: the fifth place is tied")
    for _, group in variable.groupby(variable.approved_on.dt.year):
        if len(group) < case.MIN_YEAR_LOANS:
            continue
        counts = group.rate.round(case.SPREAD_DECIMALS).value_counts()
        assert int(counts.iloc[0]) != int(counts.iloc[1]), "a yearly modal rate is tied"

    assert "breaking a tie by the lower spread" in _turns()[1]["query"]
    assert "admitting the lower spread first where a tie decides" in _turns()[1]["query"]
    assert "breaking any tie by the lower spread" in _turns()[2]["query"]
    assert "breaking a tie by the lower rate" in _turns()[3]["query"]


def test_the_spread_rounding_rule_changes_the_answer():
    """The query spends words on two decimals, so unrounded must differ."""
    matched = case._matched()
    variable = matched.loc[matched.kind.eq(case.VARIABLE)]
    raw = (variable.rate - variable.policy_rate)
    unrounded_top = 100.0 * float(raw.value_counts(normalize=True).iloc[0])
    rounded_top = _truth()["variable_modal_spread_share_pct"]
    assert not numeric_equal(unrounded_top, rounded_top, decimals=4), (
        unrounded_top, rounded_top
    )
    assert unrounded_top < rounded_top, "rounding no longer merges neighbouring values"
    assert "round the difference to two decimals" in _turns()[0]["query"]


def test_the_same_day_rule_removes_a_real_population():
    """An approval on a day the rate file skips is dropped, and such days exist."""
    truth = _truth()
    loans = case.load_expansion_table("sba_7a")
    frame = loans.loc[:, ["ApprovalDate", "InitialInterestRate",
                          "FixedorVariableInterestInd", "GrossApproval"]].copy()
    del loans
    frame["approved_on"] = pd.to_datetime(frame.ApprovalDate, errors="coerce")
    frame["rate"] = pd.to_numeric(frame.InitialInterestRate, errors="coerce")
    frame["approval"] = pd.to_numeric(frame.GrossApproval, errors="coerce")
    frame["kind"] = frame.FixedorVariableInterestInd.astype(str).str.upper().str[0]
    eligible = frame.loc[frame.kind.isin((case.VARIABLE, case.FIXED))].dropna(
        subset=["approved_on", "rate", "approval"])
    rates = case._policy_rates()
    inside = eligible.loc[eligible.approved_on.between(rates.index.min(), rates.index.max())]
    dropped = len(inside) - truth["matched_loan_count"]
    assert dropped > 0, "every approval inside the window now has a same-day rate"
    assert truth["matched_loan_count"] < len(inside)


def test_the_yearly_modal_rate_tracks_policy_while_the_spread_holds():
    truth = _truth()
    matched = case._matched()
    variable = matched.loc[matched.kind.eq(case.VARIABLE)]
    spreads = set()
    rates = []
    for _, group in variable.groupby(variable.approved_on.dt.year):
        if len(group) < case.MIN_YEAR_LOANS:
            continue
        spreads.add(float(group.spread.value_counts().index[0]))
        rates.append(float(group.rate.round(case.SPREAD_DECIMALS).value_counts().index[0]))
    assert len(rates) == truth["covered_year_count"]
    # One spread across every year, while the rate itself travels with policy.
    assert len(spreads) == 1, spreads
    assert spreads.pop() == truth["variable_modal_spread_pct"]
    assert truth["modal_rate_range_pp"] > 0.8 * truth["policy_rate_range_pp"]


def test_no_output_is_a_mechanical_zero():
    truth = _truth()
    numeric = [v for v in truth.values() if isinstance(v, (int, float))]
    assert all(value != 0 for value in numeric), truth


def test_every_text_output_declares_a_closed_representation():
    truth = _truth()
    by_name = {variable.name: variable for variable in case.variables}
    # This case reports no label, so the closed-representation rule has nothing to
    # bite on. If an output ever becomes a string the loop below starts checking it.
    text_outputs = [name for name, value in truth.items() if isinstance(value, str)]
    assert text_outputs == [], text_outputs
    for name in text_outputs:  # pragma: no cover - empty by the assertion above
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
        broken[first] = float(broken[first]) + 1_000.0
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
    assert reference["source"].strip() and len(reference["known_overclaims"]) >= 5
    facts = " ".join(reference["facts"])
    truth = _truth()
    assert f"{truth['matched_loan_count']:,}" in facts
    assert "46.3004" in facts and "15.0960" in facts and "9.67" in facts
    joined = " ".join(reference["known_overclaims"]).lower()
    assert "statutory base rate" in joined
