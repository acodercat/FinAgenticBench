"""Contracts for the three cases authored on the 2026-08-27 population tables."""

import importlib
import json
from pathlib import Path

import pytest

from core.dataset_layers import load_runtime_table
from core.evaluator import load_specs

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "cblr_election_leverage_capital_comparability": ("comparison_ranking", "cblr_capital_comparability"),
    "county_flood_loss_deposit_leadership_shift": ("temporal_path", "county_flood_deposit_shift"),
    "institutional_holding_fails_market_quality_path": ("temporal_path", "institutional_fails_market_quality"),
}


def _spec(case_id):
    family = CASES[case_id][0]
    return json.loads((ROOT / f"evals/{family}/{case_id}.json").read_text())


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_registered_four_turn_contracts_partition_the_outputs(case_id):
    spec = _spec(case_id)
    registry = json.loads((ROOT / "benchmarks.json").read_text())["cases"]
    assert registry[case_id] == f"evals/{CASES[case_id][0]}/{case_id}.json"
    module = importlib.import_module(spec["module"])
    declared = [variable.name for variable in module.variables]
    turns = spec["conversations"][0]["turns"]
    assert len(turns) == 4
    stored = [name for turn in turns for name in turn["stores"]]
    assert stored == declared, "turn stores must partition the module variables in order"
    assert len(module.DECIMALS) == len(declared)
    assert {turn["validator"] for turn in turns} <= set(module.validators)
    assert all(claim["mode"] in {"required", "conditional"} for turn in turns for claim in turn["claims"])
    assert [s for s in load_specs(ROOT / "benchmarks.json") if s["name"] == case_id]


@pytest.mark.runtime_data
def test_cblr_capital_anchors():
    module = importlib.import_module("evals.comparison_ranking.cblr_election_leverage_capital_comparability")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["linked_bank_count"] == 4265
    assert result["band_bank_count"] == 859
    assert result["cblr_elector_count"] == 173
    assert result["cblr_median_tier1_leverage_pct"] == pytest.approx(11.3685)
    assert result["non_elector_median_tier1_leverage_pct"] == pytest.approx(10.1835)
    assert result["cblr_below_threshold_certificate"] == "26451"
    assert result["cblr_cet1_reported_count"] == 0
    assert result["lowest_cet1_certificate"] == "34997"
    assert result["lowest_cet1_ratio_pct"] == pytest.approx(result["lowest_cet1_recomputed_ratio_pct"], abs=1e-4)
    assert result["book_below_leverage_bank_count"] == 582
    assert result["largest_negative_divergence_certificate"] == "12013"
    assert result["largest_negative_divergence_book_equity_to_assets_pct"] < 0 < result["largest_negative_divergence_tier1_leverage_pct"]
    capital = load_runtime_table("ffiec_call_reports.bank_capital")
    snapshot = capital.loc[capital.report_date.astype(str).eq("2025-09-30")]
    # Every missing RWA is a CBLR elector: the coalesced capital table has no other gaps.
    assert not (snapshot.standardized_risk_weighted_assets_thousand_usd.isna() & ~snapshot.cblr_election.astype("boolean").fillna(False)).any()
    assert module.validate(result).success


@pytest.mark.runtime_data
def test_county_flood_deposit_anchors():
    module = importlib.import_module("evals.temporal_path.county_flood_loss_deposit_leadership_shift")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert (result["leader_county_fips"], result["leader_loss_year"]) == ("12071", "2022")
    assert (result["runner_up_county_fips"], result["runner_up_loss_year"]) == ("12103", "2024")
    assert result["leader_net_payment_usd_millions"] == pytest.approx(3391.645328, abs=1e-5)
    assert result["leader_margin_usd_millions"] == pytest.approx(14.684433, abs=1e-5)
    assert result["pre_event_leader_certificate"] == "3510" and result["post_event_leader_certificate"] == "3511"
    assert result["earliest_major_disaster_number"] == 4673
    assert result["days_from_incident_begin_to_declaration"] == 6
    assert result["claim_record_rank_of_leader"] == 1
    claims = load_runtime_table("fema_nfip.claims")
    years = claims.date_of_loss.astype(str).str.slice(0, 4)
    assert set(years.unique()) == {str(y) for y in range(2019, 2026)}
    assert claims.county_code.astype("string").str.fullmatch(r"\d{5}", na=False).groupby(years).mean().gt(0.99).all()
    assert module.validate(result).success


@pytest.mark.runtime_data
def test_institutional_fails_market_quality_anchors():
    module = importlib.import_module("evals.temporal_path.institutional_holding_fails_market_quality_path")
    result = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert result["leader_cusip"] == "67066G104" and result["leader_symbol"] == "NVDA"
    assert result["leader_filer_count"] == 5234
    assert result["window_settlement_date_count"] == 62 and result["leader_fail_date_count"] == 50
    assert result["leader_peak_fail_settlement_date"] == "2025-12-18"
    assert result["cohort_symbol_count"] == 17
    assert result["leader_odd_lot_share_rank"] == 14
    assert result["filer_count_change"] == 661
    holdings = load_runtime_table("sec_13f.holdings_by_cusip")
    holdings = holdings.loc[holdings.cusip.astype(str).eq("67066G104")]
    by_period = holdings.set_index(holdings.period_of_report.astype(str)).share_value_usd_total
    # Unit inference: NVIDIA's aggregate must be of the order of its market
    # capitalization on both sides of the 2023 unit change (a filing-date rule gave
    # $3.2T for 2022-09-30). 13F sums can exceed capitalization because affiliated
    # managers file overlapping holdings, so the upper bound is loose.
    assert 50e9 < by_period["2022-09-30"] < 400e9
    assert 400e9 < by_period["2023-06-30"] < 2.0e12
    assert module.validate(result).success
