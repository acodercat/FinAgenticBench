import importlib
import json
from pathlib import Path

import pytest

from core.data import runtime_variables
from core.dataset_layers import load_runtime_table


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "apple_commitment_curve_discount_stress": "scenario_analysis",
    "tga_drawdown_debt_transaction_bridge": "calculation_reconciliation",
    "storm_damage_payroll_outlier_sensitivity": "risk_screening",
}


@pytest.mark.runtime_data
@pytest.mark.parametrize("case_id,family", CASES.items())
def test_batch5_cases_are_partitioned_four_turn_contracts(case_id, family):
    spec = json.loads((ROOT / f"evals/{family}/{case_id}.json").read_text())
    module = importlib.import_module(spec["module"])
    turns = spec["conversations"][0]["turns"]
    assert len(turns) == 4
    stored = [name for turn in turns for name in turn["stores"]]
    partition = [name for names in (module.T1, module.T2, module.T3, module.T4) for name in names]
    declared = [variable.name for variable in module.variables]
    assert stored == partition == declared
    assert len(stored) == len(set(stored))
    truth = dict(zip(declared, module.ground_truth()))
    assert module.validate(truth).success
    for index, turn in enumerate(turns, start=1):
        assert getattr(module, f"validate_turn_{index}")(
            {name: truth[name] for name in turn["stores"]}
        ).success


def test_batch5_later_turns_explicitly_consume_prior_state():
    markers = {
        "scenario_analysis/apple_commitment_curve_discount_stress.json": ["turn-1", "turn-2", "turn-3"],
        "calculation_reconciliation/tga_drawdown_debt_transaction_bridge.json": ["turn-1", "same ending date", "turn-1"],
        "risk_screening/storm_damage_payroll_outlier_sensitivity.json": ["turn-1", "turn-1", "turn-2"],
    }
    for relative, expected in markers.items():
        turns = json.loads((ROOT / "evals" / relative).read_text())["conversations"][0]["turns"]
        for turn, marker in zip(turns[1:], expected, strict=True):
            assert marker in turn["query"].lower()


@pytest.mark.runtime_data
def test_batch5_new_tic_and_dts_runtime_contracts():
    tic = load_runtime_table("treasury_tic.major_foreign_holders")
    # The table now spans every TIC year block; the 2025 contract is asserted on that subset.
    tic = tic.loc[tic.observation_month.astype(str).str.startswith("2025")]
    assert len(tic) == 456
    assert tic.loc[tic.row_type.eq("country")].country_name.nunique() == 36
    assert tic.groupby("country_name").observation_month.nunique().eq(12).all()
    assert set(tic.row_type) == {"country", "aggregate"}

    # DTS tables now start 2019-01-01; the 2024-2025 contract is asserted on that subset.
    cash = load_runtime_table("treasury_dts.operating_cash_balance")
    cash = cash.loc[cash.record_date.astype(str).ge("2024-01-01")]
    assert len(cash) == 2004
    assert cash.groupby("record_date").account_type.nunique().eq(4).all()
    assert cash.today_amount_million_usd.notna().all()
    debt = load_runtime_table("treasury_dts.public_debt_transactions")
    debt = debt.loc[debt.record_date.astype(str).ge("2024-01-01")]
    assert len(debt) == 11523
    assert debt.transaction_today_amount_million_usd.notna().all()
    assert debt.transaction_today_amount_million_usd.lt(0).any()


def test_batch5_treasury_dts_source_family_exposes_both_tables():
    names = {variable.name for variable in runtime_variables(["treasury_dts"])}
    assert {"treasury_dts_operating_cash_df", "treasury_dts_public_debt_transactions_df"} <= names


@pytest.mark.runtime_data
def test_batch5_material_numerical_anchors():
    apple = importlib.import_module("evals.scenario_analysis.apple_commitment_curve_discount_stress")
    a = dict(zip((v.name for v in apple.variables), apple.ground_truth()))
    assert a["gross_debt_fact_physical_row_count"] == 4
    assert a["gross_debt_distinct_undimensioned_value_count"] == 1
    assert a["maturity_sum_minus_gross_debt_usd_millions"] == pytest.approx(0)
    assert a["shocked_combined_present_value_usd_billions"] < a["base_combined_present_value_usd_billions"]

    tga = importlib.import_module("evals.calculation_reconciliation.tga_drawdown_debt_transaction_bridge")
    g = dict(zip((v.name for v in tga.variables), tga.ground_truth()))
    assert g["largest_tga_decline_end_date"] == "2025-08-01"
    assert abs(g["dts_net_issues_minus_total_debt_change_usd_millions"]) < 0.5
    assert g["marketable_net_minus_public_change_usd_billions"] == pytest.approx(
        -g["nonmarketable_net_minus_intragovernmental_change_usd_billions"], abs=0.001
    )

    storm = importlib.import_module("evals.risk_screening.storm_damage_payroll_outlier_sensitivity")
    s = dict(zip((v.name for v in storm.variables), storm.ground_truth()))
    assert s["absolute_damage_leader_state_usps"] == "NC"
    assert s["damage_per_employee_leader_state_usps"] == "NM"
    assert s["top_event_type_rows_excluding_largest_event_count"] == 366
    assert s["top_event_type_damage_retained_pct"] == pytest.approx(84.4922009987)
    assert s["top_type_remainder_exceeds_all_other_types"] is True


@pytest.mark.parametrize("case_id,family", CASES.items())
def test_batch5_declared_sources_are_runtime_visible(case_id, family):
    spec = json.loads((ROOT / f"evals/{family}/{case_id}.json").read_text())
    exposed = runtime_variables(spec["data_sources"])
    assert len(exposed) >= len(set(spec["data_sources"]))
