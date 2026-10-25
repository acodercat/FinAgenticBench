import importlib
import json
from pathlib import Path

import pytest

from core.dataset_layers import load_runtime_table


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "gdp_revision_curve_positioning": "temporal_path",
    "money_fund_redemption_repo_recovery": "risk_screening",
    "sba_lending_deposit_geography_capital_sensitivity": "scenario_analysis",
    "wti_negative_spot_stress_recovery": "temporal_path",
}


@pytest.mark.runtime_data
@pytest.mark.parametrize("case_id,family", CASES.items())
def test_batch1_cases_are_genuine_four_turn_contracts(case_id, family):
    spec = json.loads((ROOT / f"evals/{family}/{case_id}.json").read_text())
    module = importlib.import_module(spec["module"])
    turns = spec["conversations"][0]["turns"]
    assert len(turns) == 4
    stored = [name for turn in turns for name in turn["stores"]]
    declared = [variable.name for variable in module.variables]
    assert len(stored) == len(set(stored))
    assert stored == [name for names in (
        module.TURN_1_NAMES, module.TURN_2_NAMES,
        module.TURN_3_NAMES, module.TURN_4_NAMES,
    ) for name in names]
    assert set(stored) == set(declared)
    truth = dict(zip(declared, module.ground_truth()))
    for index, turn in enumerate(turns, start=1):
        check = getattr(module, f"validate_turn_{index}")
        assert check({name: truth[name] for name in turn["stores"]}).success


@pytest.mark.runtime_data
def test_gdp_revision_case_anchors_and_source_contracts():
    module = importlib.import_module("evals.temporal_path.gdp_revision_curve_positioning")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["largest_revision_quarter"] == "2025-Q2"
    assert result["largest_adjacent_revision_stage"] == "second-to-third"
    assert result["leader_current_real_gdp_pct"] - result["leader_first_release_real_gdp_pct"] == pytest.approx(0.8698)
    assert result["curve_end_two_ten_spread_bp"] - result["curve_start_two_ten_spread_bp"] == pytest.approx(18)
    assert result["two_year_net_contract_change"] == -221718
    assert result["ten_year_net_contract_change"] == -358539
    revisions = load_runtime_table("rtdsm.revisions")
    complete = revisions.loc[
        revisions.series.eq("real_gdp_annualized_pct")
        & revisions.period.between("2024:Q1", "2025:Q2")
    ].dropna(subset=["first_release", "second_release", "third_release", "most_recent"])
    assert len(complete) == 6


@pytest.mark.runtime_data
def test_money_fund_case_anchors_and_missing_liquidity_is_not_silently_zero():
    module = importlib.import_module("evals.risk_screening.money_fund_redemption_repo_recovery")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["eligible_large_fund_report_observation_count"] == 1248
    assert result["redemption_pressure_event_date"] == "2025-10-31"
    assert result["redemption_pressure_leader_series_id"] == "S000085371"
    assert result["leader_redemption_liquidity_ratio_pct"] == pytest.approx(151.28038667)
    assert result["one_month_decline_recovered_pct"] == pytest.approx(49.48204872)
    liquidity = load_runtime_table("sec_nmfp.daily_liquidity")
    event = liquidity.loc[liquidity.liquidity_date.eq("2025-10-31")]
    assert event.daily_liquid_assets_usd.isna().any()
    assert event.daily_liquid_assets_usd.eq(0).any()
    spec = json.loads(
        (ROOT / "evals/risk_screening/money_fund_redemption_repo_recovery.json").read_text()
    )
    assert "ratio above 100%" not in spec["conversations"][0]["turns"][0]["query"]
    assert "report date selected in the first turn" in spec["conversations"][0]["turns"][2]["query"]
    declared = [
        name
        for turn in spec["conversations"][0]["turns"]
        for name in turn["stores"]
    ]
    assert "event_selected_accession" not in declared
    assert len(module.variables) == 32
    second_turn_claims = {
        claim["id"]: claim["mode"]
        for claim in spec["conversations"][0]["turns"][1]["claims"]
    }
    assert second_turn_claims["maturity_path_divergence_result"] == "required"
    assert second_turn_claims["reported_snapshot_path_boundary"] == "conditional"


@pytest.mark.runtime_data
def test_sba_geography_case_anchors_and_identifier_contracts():
    module = importlib.import_module("evals.scenario_analysis.sba_lending_deposit_geography_capital_sensitivity")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["eligible_sba_sod_lender_count"] == 19
    assert result["geography_mismatch_leader_fdic_certificate"] == "19690"
    assert result["leader_geography_tvd_pct"] == pytest.approx(99.74397405)
    assert result["negative_private_employment_jurisdiction_count"] == 13
    assert result["selected_bank_rssd_id"] == "468806"
    assert result["hypothetical_equity_reduction_pct"] == pytest.approx(0.39292042)
    sba = load_runtime_table("sba_7a_504.seven_a_loans")
    nonmissing = sba.BankFDICNumber.dropna()
    assert nonmissing.str.fullmatch(r"\d{5}").all()
    assert sba.LocationID.dropna().str.fullmatch(r"\d{7}").all()
    assert sba.fdic_certificate.dropna().str.fullmatch(r"[1-9]\d*").all()
    assert (
        sba.BankFDICNumber.dropna().str.lstrip("0")
        == sba.loc[sba.BankFDICNumber.notna(), "fdic_certificate"]
    ).all()
    spec = json.loads(
        (ROOT / "evals/scenario_analysis/sba_lending_deposit_geography_capital_sensitivity.json").read_text()
    )
    assert "census_geo_crosswalk" not in spec["data_sources"]
    first_turn_claims = {
        claim["id"]: claim["mode"]
        for claim in spec["conversations"][0]["turns"][0]["claims"]
    }
    assert first_turn_claims["approval_deposit_distribution_distinction"] == "required"
    assert first_turn_claims["loan_depositor_tracing_boundary"] == "conditional"
    assert first_turn_claims["funding_liquidity_boundary"] == "conditional"
    call = load_runtime_table("ffiec_call_reports.bank_balance_sheets")
    selected = call.loc[call.fdic_certificate.eq("19690") & call.report_date.eq("2024-12-31")]
    assert len(selected) == 1
    assert selected.consolidated_total_equity_thousand_usd.isna().all()
    assert selected.domestic_office_total_equity_thousand_usd.notna().all()


@pytest.mark.runtime_data
def test_wti_recovery_case_anchors_and_reported_date_grid():
    module = importlib.import_module("evals.temporal_path.wti_negative_spot_stress_recovery")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["joint_wti_ofr_observation_count"] == 6514
    assert result["selloff_window_end_date"] == "2020-04-20"
    assert result["selloff_end_wti_usd_per_barrel"] == pytest.approx(-36.98)
    assert result["wti_recovery_date"] == "2020-05-04"
    assert result["wti_recovery_later_joint_observation_count"] == 10
    wti = load_runtime_table("eia_bulk.wti_daily_spot_prices")
    assert not wti.observation_date.duplicated().any()
    assert not wti.observation_date.eq("2020-04-18").any()
    assert not wti.observation_date.eq("2020-04-19").any()
    spec = json.loads(
        (ROOT / "evals/temporal_path/wti_negative_spot_stress_recovery.json").read_text()
    )
    claims_by_turn = [
        {claim["id"]: claim["mode"] for claim in turn.get("claims", [])}
        for turn in spec["conversations"][0]["turns"]
    ]
    assert "wti_spot_measure_identity" not in claims_by_turn[0]
    assert claims_by_turn[0]["retrospective_snapshot_boundary"] == "conditional"
    assert claims_by_turn[2]["oil_stress_causality_boundary"] == "conditional"
    assert "oil_stress_causality_boundary" not in claims_by_turn[3]
