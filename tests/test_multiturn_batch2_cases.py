import importlib
import json
from pathlib import Path

import pytest

from core.dataset_layers import load_runtime_table


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "disaster_credit_union_delinquency_concentration": "risk_screening",
    "reg_cf_amendment_reg_d_market_context": "temporal_path",
}


@pytest.mark.runtime_data
@pytest.mark.parametrize("case_id,family", CASES.items())
def test_batch2_cases_are_partitioned_four_turn_contracts(case_id, family):
    spec = json.loads((ROOT / f"evals/{family}/{case_id}.json").read_text())
    module = importlib.import_module(spec["module"])
    turns = spec["conversations"][0]["turns"]
    assert len(turns) == 4
    stored = [name for turn in turns for name in turn["stores"]]
    declared = [variable.name for variable in module.variables]
    partition = [
        name
        for names in (
            module.TURN_1_NAMES, module.TURN_2_NAMES,
            module.TURN_3_NAMES, module.TURN_4_NAMES,
        )
        for name in names
    ]
    assert stored == partition
    assert len(stored) == len(set(stored))
    assert set(stored) == set(declared)
    truth = dict(zip(declared, module.ground_truth()))
    for index, turn in enumerate(turns, start=1):
        assert getattr(module, f"validate_turn_{index}")(
            {name: truth[name] for name in turn["stores"]}
        ).success


@pytest.mark.runtime_data
def test_disaster_credit_union_anchors_and_county_grain():
    module = importlib.import_module(
        "evals.risk_screening.disaster_credit_union_delinquency_concentration"
    )
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["eligible_disaster_screened_state_count"] == 11
    assert result["delinquency_leader_state"] == "VA"
    assert result["dominant_credit_union_number"] == "5536"
    assert result["dominant_credit_union_state_delinquent_share_pct"] == pytest.approx(77.7456088)
    assert result["adjusted_leader_state_rank"] == 3
    assert result["adjusted_screen_leader_state"] == "ND"
    counties = load_runtime_table("census_geo_crosswalk.counties_2024")
    assert not counties.GEOID.duplicated().any()
    disasters = load_runtime_table("fema_disasters.declaration_areas")
    assert disasters.county_fips.duplicated().any()



@pytest.mark.runtime_data
def test_reg_cf_reg_d_anchors_and_indefinite_amount_contract():
    module = importlib.import_module(
        "evals.temporal_path.reg_cf_amendment_reg_d_market_context"
    )
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["eligible_reg_cf_lineage_count"] == 399
    assert result["largest_increase_file_number"] == "020-35641"
    assert result["latest_form_c_a_filing_date"] == "2025-11-20"
    assert result["initial_form_d_filing_count"] == 3057
    assert result["indefinite_initial_form_d_filing_count"] == 900
    assert result["selected_maximum_form_d_inclusive_percentile_pct"] == pytest.approx(58.5999073)
    form_d = load_runtime_table("sec_form_d.filings")
    assert form_d.TOTALOFFERINGAMOUNT.eq("Indefinite").any()


def test_batch2_auxiliary_identifier_fidelity():
    rates = load_runtime_table("nyfed.reference_rates")
    footnotes = rates["Footnote ID"].dropna()
    assert footnotes.str.fullmatch(r"\d+").all()
    form_c = load_runtime_table("sec_form_c.filings")
    assert form_c.CIK.dropna().str.fullmatch(r"\d{10}").all()
    assert form_c.PERIOD.dropna().str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    form_d = load_runtime_table("sec_form_d.filings")
    assert form_d.CIK.dropna().str.fullmatch(r"\d{10}").all()
