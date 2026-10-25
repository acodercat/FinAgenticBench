"""Case-local gate for rate_futures_spread_leg_and_benchmark_succession.

The trap has to keep working: adding the spread leg must still reverse the sign
of the net position, and the exact-equality contract on a contract count must
still exclude it. The three financial properties the case rests on are pinned
too — that one category's legs can add past open interest, that the retired
contract stops rather than empties, and that the benchmark compounds.
"""

import json

import numpy as np
import pandas as pd

from config import PROJECT_ROOT
from core.validation import numeric_equal
from evals.temporal_path import rate_futures_spread_leg_and_benchmark_succession as case
from scripts.audit_case_design import has_closed_text_representation


SPEC = json.loads(
    (PROJECT_ROOT / "evals/temporal_path/rate_futures_spread_leg_and_benchmark_succession.json").read_text()
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["successor_report_date_count"] == 365
    assert truth["final_open_interest_contracts"] == 12419866
    assert truth["final_long_contracts"] == 1683620
    assert truth["final_net_contracts"] == -476567
    assert truth["final_spread_contracts"] == 4094368
    assert truth["three_legs_exceed_open_interest_date_count"] == 27
    assert truth["spread_exceeds_directional_date_count"] == 89
    assert truth["predecessor_report_date_count"] == 232
    assert truth["predecessor_final_report_date"] == "2023-06-13"
    assert truth["succession_window_calendar_days"] == 931


def test_adding_the_spread_leg_still_reverses_the_net():
    """The trap, on the pinned date and across the whole table."""
    truth = _truth()
    successor = case._contract(case.SUCCESSOR_CODE)
    final = successor.iloc[-1]

    naive = int(final.long - final.short + final.spread)
    assert naive == 3617801
    assert np.sign(naive) != np.sign(truth["final_net_contracts"])
    # Counts are compared exactly, so any difference is excluded.
    assert not numeric_equal(naive, truth["final_net_contracts"], decimals=0)
    assert abs(naive - truth["final_net_contracts"]) == truth["final_spread_contracts"]

    positions = case._positions()
    directional = positions.long - positions.short
    with_spread = directional + positions.spread
    live = directional.ne(0)
    flips = int((np.sign(directional[live]) * np.sign(with_spread[live])).lt(0).sum())
    assert len(positions) == 597
    assert flips == 229, flips
    relative = ((with_spread[live] / directional[live]) - 1.0).abs()
    assert round(100 * float(relative.median()), 2) == 304.44


def test_one_category_can_add_past_open_interest():
    truth = _truth()
    assert truth["final_three_leg_share_of_open_interest_pct"] > 50
    assert truth["three_legs_exceed_open_interest_date_count"] > 0, (
        "the legs no longer add past open interest, so the non-partition point is gone"
    )
    successor = case._contract(case.SUCCESSOR_CODE)
    assert (successor.long <= successor.open_interest).all()
    assert (successor.short <= successor.open_interest).all()


def test_the_predecessor_stops_rather_than_empties():
    truth = _truth()
    successor = case._contract(case.SUCCESSOR_CODE)
    predecessor = case._contract(case.PREDECESSOR_CODE)
    assert truth["predecessor_final_open_interest_contracts"] > 0, (
        "the contract now ends at zero, which removes the distinction"
    )
    # Every predecessor date is a successor date, so absence is delisting, not a gap.
    assert set(predecessor.reported_on) <= set(successor.reported_on)
    assert predecessor.reported_on.max() < successor.reported_on.max()
    assert (predecessor.open_interest > 0).all()


def test_the_two_contracts_share_the_crossover_report_date():
    successor = case._contract(case.SUCCESSOR_CODE)
    predecessor = case._contract(case.PREDECESSOR_CODE)
    crossover = pd.Timestamp(f"{case.MIGRATION_YEAR}-12-31")
    assert (
        successor.loc[successor.reported_on.le(crossover)].reported_on.max()
        == predecessor.loc[predecessor.reported_on.le(crossover)].reported_on.max()
    ), "the query says 'the last report date of 2022'; the two contracts must agree on it"


def test_the_endpoint_rule_for_the_mean_is_load_bearing():
    """Including the start date moves the answer past the tolerance, so the query pins it."""
    panel = case._overnight_rates()
    successor = case._contract(case.SUCCESSOR_CODE)
    predecessor = case._contract(case.PREDECESSOR_CODE)
    start, end = predecessor.reported_on.max(), successor.reported_on.max()
    pinned = float(panel.rate.loc[(panel.index > start) & (panel.index <= end)].mean())
    inclusive = float(panel.rate.loc[(panel.index >= start) & (panel.index <= end)].mean())
    assert abs(pinned - inclusive) > 0.6e-4, "the endpoint rule no longer changes the answer"
    assert round(pinned, 4) == round(_truth()["arithmetic_mean_overnight_rate_pct"], 4)
    assert "published after the start date through the end date" in _turns()[3]["query"]


def test_the_benchmark_compounds_and_the_published_average_proves_it():
    truth = _truth()
    assert truth["compounded_overnight_rate_pct"] > truth["arithmetic_mean_overnight_rate_pct"]
    assert round(truth["compounding_gap_percentage_points"], 4) == 0.3100
    assert not numeric_equal(
        truth["arithmetic_mean_overnight_rate_pct"],
        truth["compounded_overnight_rate_pct"],
        decimals=4,
    )

    # The source's own ninety-day average settles which convention is meant.
    from core.data import load_expansion_table

    rates = load_expansion_table("nyfed_reference_rates")
    frame = rates.loc[rates["Rate Type"].astype(str).eq(case.INDEX_RATE_TYPE)].copy()
    del rates
    frame["observed_on"] = pd.to_datetime(frame["Effective Date"])
    frame["published"] = pd.to_numeric(frame["90-Day Average SOFR"], errors="coerce")
    assert frame.published.notna().sum() > 1000, "the published averages are no longer populated"

    panel = case._overnight_rates()
    end = panel.index.max()
    start = panel.index[panel.index <= end - pd.Timedelta(days=90)].max()
    days = (end - start).days
    compounded = (
        float(panel.accrual_index[end]) / float(panel.accrual_index[start]) - 1.0
    ) * case.DAY_COUNT_BASIS / days * 100.0
    published = float(frame.loc[frame.observed_on.eq(end), "published"].iloc[0])
    plain = float(panel.rate.loc[(panel.index > start) & (panel.index <= end)].mean())
    assert abs(published - compounded) < abs(published - plain) / 100


def test_the_accrual_index_reconstructs_the_daily_rate():
    panel = case._overnight_rates().copy()
    panel["days"] = panel.index.to_series().diff().dt.days
    implied = panel.accrual_index / panel.accrual_index.shift(1) - 1.0
    accrued = panel.rate.shift(1) / 100.0 * panel.days / case.DAY_COUNT_BASIS
    gap = (implied - accrued).abs().dropna()
    assert gap.max() < 1e-7, gap.max()


def test_the_query_withholds_the_distinction_that_avoids_the_trap():
    """P5: a trap whose query states the avoidance is a baseline."""
    opening = _turns()[0]["query"].lower()
    for giveaway in ("spread", "offsetting", "two directional", "long minus short",
                     "exclude", "three legs"):
        assert giveaway not in opening, giveaway
    assert SPEC["case_type"] == "trap"
    conditional = [
        claim for turn in _turns() for claim in turn["claims"]
        if claim["id"] == "net_position_composition"
    ]
    assert conditional and conditional[0]["mode"] == "conditional"


def test_every_anchor_the_case_selects_is_a_last_date():
    """No selection can tie, because each is the end of a stated series or year."""
    joined = " ".join(turn["query"] for turn in _turns())
    assert "on the last of them" in joined
    assert "its last report date" in joined
    assert "the last report date of 2022" in joined
    assert "last report date" in joined


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
    assert f"{truth['final_net_contracts']:,}" in facts
    assert f"{truth['final_spread_contracts']:,}" in facts
    assert "3,617,801" in facts
    assert truth["predecessor_final_report_date"] in facts


def test_the_rate_source_is_load_bearing():
    """The fourth turn is impossible from the positioning file alone."""
    panel = case._overnight_rates()
    assert {"rate", "accrual_index"} <= set(panel.columns)
    positions = case._positions()
    assert not {"rate", "accrual_index"} & set(positions.columns)

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
        for giveaway in ("spread", "offsetting", "two directional", "long minus short",
                     "exclude", "three legs"):
            assert giveaway not in lowered, (variable.name, giveaway)
