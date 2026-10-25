import importlib
import json
from pathlib import Path

import pytest

from core.dataset_layers import load_runtime_table


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "pension_funding_vintage_curve_scenario": (
        ROOT / "evals/scenario_analysis/pension_funding_vintage_curve_scenario.json"
    ),
}


@pytest.mark.runtime_data
@pytest.mark.parametrize("case_name", CASES)
def test_new_cases_are_four_turn_conversations_with_partitioned_outputs(case_name):
    spec = json.loads(CASES[case_name].read_text())
    module = importlib.import_module(spec["module"])
    turns = spec["conversations"][0]["turns"]
    assert len(turns) == 4
    stored = [name for turn in turns for name in turn["stores"]]
    declared = [variable.name for variable in module.variables]
    assert len(stored) == len(set(stored))
    assert set(stored) == set(declared)

    truth = dict(zip(declared, module.ground_truth()))
    for turn in turns:
        outputs = {name: truth[name] for name in turn["stores"]}
        check = getattr(module, turn["validator"].replace("validate_", "validate_turn_"), None)
        if check is None:
            check = {
                "validate_screen": module.validate_turn_1,
                "validate_identity": module.validate_turn_2,
                "validate_bridge": module.validate_turn_3,
                "validate_scenario": module.validate_turn_4,
                "validate_early_screen": module.validate_turn_1,
                "validate_updated_screen": module.validate_turn_2,
                "validate_filing_reconciliation": module.validate_turn_3,
                "validate_curve_scenario": module.validate_turn_4,
            }[turn["validator"]]
        assert check(outputs).success


@pytest.mark.runtime_data
def test_pension_vintage_case_independent_key_results_and_data_contracts():
    module = importlib.import_module(
        "evals.scenario_analysis.pension_funding_vintage_curve_scenario"
    )
    result = dict(zip((variable.name for variable in module.variables), module.ground_truth()))
    assert result["early_underfunded_plan_count"] == 4
    assert result["early_shortfall_leader_plan_name"] == "LUMEN COMBINED PENSION PLAN"
    assert result["early_shortfall_leader_plan_number"] == "001"
    assert result["updated_underfunded_plan_count"] == 58
    assert result["updated_shortfall_leader_plan_name"].startswith("LOCKHEED MARTIN")
    assert result["updated_shortfall_leader_plan_number"] == "001"
    assert result["early_leader_rank_at_updated_cutoff"] == 26
    assert result["updated_leader_net_asset_rollforward_residual_usd_millions"] == 0
    assert result["treasury_curve_start_observation_date"] == "2023-12-29"
    assert result["treasury_thirty_year_yield_change_basis_points"] == pytest.approx(75)
    assert result["hypothetical_stressed_actuarial_funded_pct"] == pytest.approx(
        95.2070099526
    )

    financials = load_runtime_table("dol_form5500.plan_financials")
    schedules = load_runtime_table("dol_form5500.schedule_sb")
    assert not financials.filing_id.duplicated().any()
    assert not schedules.filing_id.duplicated().any()
    assert financials.plan_number.dropna().str.fullmatch(r"\d{3}").all()
    assert schedules.plan_number.dropna().str.fullmatch(r"\d{3}").all()
    assert financials.sponsor_ein.dropna().str.fullmatch(r"\d{9}").all()
    assert schedules.sponsor_ein.dropna().str.fullmatch(r"\d{9}").all()
    assert financials.sponsor_ein.str.startswith("0", na=False).any()
    assert schedules.sponsor_ein.str.startswith("0", na=False).any()
    selected = result["updated_shortfall_leader_filing_id"]
    assert len(financials.loc[financials.filing_id.eq(selected)]) == 1
    assert len(schedules.loc[schedules.filing_id.eq(selected)]) == 1
    curve = load_runtime_table("treasury_rates.yield_curve")
    assert not curve.Date.duplicated().any()
    assert not curve.Date.eq("2024-01-01").any()
    assert curve.Date.eq("2023-12-29").sum() == 1
