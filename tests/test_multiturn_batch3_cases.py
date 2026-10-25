import importlib
import json
from pathlib import Path

import pytest

from core.dataset_layers import load_runtime_table


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "bis_eer_credit_service_screen": "risk_screening",
    "treasury_debt_marketability_auction_path": "calculation_reconciliation",
    "nfip_county_bank_deposit_sensitivity": "risk_screening",
}


@pytest.mark.runtime_data
@pytest.mark.parametrize("case_id,family", CASES.items())
def test_batch3_cases_are_partitioned_four_turn_contracts(case_id, family):
    spec = json.loads((ROOT / f"evals/{family}/{case_id}.json").read_text())
    module = importlib.import_module(spec["module"])
    turns = spec["conversations"][0]["turns"]
    assert len(turns) == 4
    stored = [name for turn in turns for name in turn["stores"]]
    declared = [variable.name for variable in module.variables]
    partition = [
        name for names in (
            module.TURN_1_NAMES, module.TURN_2_NAMES,
            module.TURN_3_NAMES, module.TURN_4_NAMES,
        ) for name in names
    ]
    assert stored == partition
    assert len(stored) == len(set(stored))
    assert set(stored) == set(declared)
    truth = dict(zip(declared, module.ground_truth()))
    assert module.validate(truth).success
    for index, turn in enumerate(turns, start=1):
        assert getattr(module, f"validate_turn_{index}")(
            {name: truth[name] for name in turn["stores"]}
        ).success




@pytest.mark.runtime_data
def test_bis_screen_anchors_and_source_populations():
    module = importlib.import_module("evals.risk_screening.bis_eer_credit_service_screen")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["joint_economy_count"] == 17
    assert result["screened_economy_count"] == 3
    assert result["selected_economy"] == "Japan"
    assert result["selected_credit_gap_rank"] == 1
    assert result["selected_nominal_eer_decline_rank"] == 2
    assert result["selected_real_eer_decline_rank"] == 1
    credit = load_runtime_table("bis.credit_gap")
    dsr = load_runtime_table("bis.debt_service_ratios")
    eer = load_runtime_table("bis.effective_exchange_rates")
    assert len(credit) and len(dsr) and len(eer)



@pytest.mark.runtime_data
def test_treasury_debt_marketable_anchors_and_stock_flow_gap():
    module = importlib.import_module("evals.calculation_reconciliation.treasury_debt_marketability_auction_path")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["aligned_mspd_snapshot_count"] == 24
    assert result["selected_interval_end_mspd_date"] == "2025-07-31"
    assert result["largest_marketable_change_class"] == "Bills"
    assert result["selected_class_issue_event_count"] == 33
    assert result["largest_selected_class_term_by_offering"] == "13-Week"
    assert result["selected_class_offering_to_stock_change_ratio"] == pytest.approx(10.87093888)


@pytest.mark.runtime_data
def test_nfip_bank_sensitivity_anchors_and_policy_grain():
    module = importlib.import_module("evals.risk_screening.nfip_county_bank_deposit_sensitivity")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["eligible_nfip_bank_county_count"] == 17
    assert result["selected_county_fips"] == "37021"
    assert result["selected_county_claim_row_count"] == 506
    assert result["selected_county_represented_policy_count"] == 513
    assert result["selected_county_deposit_leader_certificate"] == "3511"
    assert result["adjusted_selected_county_leader_certificate"] == "9846"
    assert result["original_leader_adjusted_rank"] == 2
    claims = load_runtime_table("fema_nfip.claims")
    assert claims.policy_count.gt(1).any()
