import importlib
import json
from pathlib import Path

import pytest

from config import DATASETS_DIR
from core.data import runtime_variables
from core.dataset_layers import load_runtime_table


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "scf_age_wealth_duration_sensitivity": "scenario_analysis",
    "money_fund_outflow_sofr_positioning": "risk_screening",
}


@pytest.mark.runtime_data
@pytest.mark.parametrize("case_id,family", CASES.items())
def test_batch4_cases_are_partitioned_four_turn_contracts(case_id, family):
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




def test_batch4_later_turns_consume_prior_state():
    checks = {
        "evals/scenario_analysis/scf_age_wealth_duration_sensitivity.json": {
            1: "selected age band", 2: "turn-2", 3: "turn-3",
        },
        "evals/risk_screening/money_fund_outflow_sofr_positioning.json": {
            1: "selected above", 2: "selected liquidity-step", 3: "turn-1",
        },
    }
    for relative, required in checks.items():
        turns = json.loads((ROOT / relative).read_text())["conversations"][0]["turns"]
        for turn_index, marker in required.items():
            assert marker in turns[turn_index]["query"].lower()


@pytest.mark.runtime_data
def test_batch4_scf_anchors_and_implicate_integrity():
    module = importlib.import_module("evals.scenario_analysis.scf_age_wealth_duration_sensitivity")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["eligible_scf_household_count"] == 3977
    assert result["selected_age_band"] == "65-74"
    assert result["selected_age_band_implicate_row_count"] == 4330
    assert result["survey_weighted_household_count_millions"] == pytest.approx(109.60309638346928)
    assert result["selected_cohort_aggregate_duration_loss_usd_billions"] == pytest.approx(
        419.0364106995732
    )
    assert result["ten_year_drawup_start_date"] == "2025-04-04"
    assert result["ten_year_drawup_end_date"] == "2025-05-21"
    rows = load_runtime_table("fed_scf.household_summary")
    assert rows.groupby("household_id").implicate.nunique().eq(5).all()
    description = next(
        variable.description for variable in runtime_variables(["fed_scf"])
        if variable.name == "scf_summary_df"
    )
    assert "X42001 divided by five" in description


@pytest.mark.runtime_data
def test_batch4_money_fund_anchors_and_regression_path():
    module = importlib.import_module("evals.risk_screening.money_fund_outflow_sofr_positioning")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["liquidity_drawdown_leader_accession"] == "0001145549-25-006380"
    assert result["liquidity_drawdown_start_date"] == "2025-01-30"
    assert result["liquidity_drawdown_end_date"] == "2025-01-31"
    assert result["prior_cftc_report_date"] == "2025-01-28"
    assert result["next_cftc_report_date"] == "2025-02-04"
    assert result["net_outflow_to_weekly_liquidity_decline_pct"] < 50
    assert result["weekly_liquidity_decomposition_residual_usd_millions"] == pytest.approx(0)
    spec = json.loads((ROOT / "evals/risk_screening/money_fund_outflow_sofr_positioning.json").read_text())
    assert spec["case_type"] == "baseline"
    turns = spec["conversations"][0]["turns"]
    assert "decompose" in turns[3]["query"].lower()
    assert next(
        claim for claim in turns[2]["claims"]
        if claim["id"] == "fund_positioning_causality_boundary"
    )["mode"] == "required"


def test_batch4_call_report_source_family_exposes_both_governed_tables():
    exposed = {variable.name for variable in runtime_variables(["fdic_bankfind", "ffiec_call_reports", "fhfa_hpi"])}
    assert "ffiec_bank_balance_sheets_df" in exposed
    assert "ffiec_bank_capital_df" in exposed


@pytest.mark.runtime_data
def test_batch4_repaired_source_integrity_contracts():
    filings = load_runtime_table("sec_nmfp.series_filings")
    # Re-pinned on 2026-08-28: the full-population N-MFP build extends the
    # filing history back to 2019 (3,633 rows before the rebuild).
    assert len(filings) == 20385
    assert filings.cik.str.fullmatch(r"\d{10}").all()
    assert filings.cik.str.startswith("0").all()

    flows = load_runtime_table("sec_nmfp.daily_shareholder_flows")
    assert len(flows) == 247400
    assert not flows.duplicated(["source_archive", "source_row_number"]).any()
    business = [
        "accession", "class_id", "gross_subscriptions_usd",
        "gross_redemptions_usd", "flow_date",
    ]
    duplicated = flows.loc[flows.duplicated(business, keep=False), business]
    assert len(duplicated) == 2
    assert duplicated.gross_subscriptions_usd.eq(0).all()
    assert duplicated.gross_redemptions_usd.eq(0).all()

    capital = load_runtime_table("ffiec_call_reports.bank_capital")
    balance = load_runtime_table("ffiec_call_reports.bank_balance_sheets")
    assert len(capital) == len(balance) == 31763
    assert len(capital.loc[capital.bank_id.eq("37")]) == 7
    assert capital.filing_type.str.fullmatch(r"\d{3}").all()

    fdic_schema = json.loads((DATASETS_DIR / "fdic_bankfind/schema.json").read_text())
    fdic_columns = {
        column["name"]: column for column in
        fdic_schema["tables"]["fdic.bank_financials"]["columns"]
    }
    assert fdic_columns["RBCRWAJ"]["unit"] == "percent"

    bis_schema = json.loads((DATASETS_DIR / "bis/schema.json").read_text())
    bis_columns = {
        column["name"]: column for column in
        bis_schema["tables"]["bis.global_liquidity"]["columns"]
    }
    assert "UNIT_MULT" not in bis_columns["OBS_VALUE:Observation Value"]["unit"]


@pytest.mark.parametrize("case_id,family", CASES.items())
def test_batch4_declared_sources_are_runtime_visible(case_id, family):
    spec = json.loads((ROOT / f"evals/{family}/{case_id}.json").read_text())
    exposed = runtime_variables(spec["data_sources"])
    assert len(exposed) >= len(set(spec["data_sources"]))
