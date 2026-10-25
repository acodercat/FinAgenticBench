import importlib
import json

from config import DATASETS_DIR, PROJECT_ROOT
from core.data import load_expansion_table, runtime_variables


def _expected_outputs(module):
    return {
        variable.name: value
        for variable, value in zip(module.variables, module.ground_truth())
    }


def _turn(spec):
    return spec["conversations"][0]["turns"][0]


def _claims(spec):
    """Every claim in the case, in turn order: a case may spread them over turns."""
    return [claim for turn in spec["conversations"][0]["turns"] for claim in turn["claims"]]


def test_phase6_build_report_and_governed_tables_are_complete():
    report = json.loads(
        (PROJECT_ROOT / "metadata/build_reports/phase6_cross_source_table_build.json").read_text()
    )
    assert report["fdic_sod_branches"] == {
        "rows": 76120,
        "report_date": "2025-06-30",
        "output": "fdic_sod/branches_2025.csv.gz",
    }
    assert report["finra_april_short_volume"] == {
        "rows": 219470,
        "trading_dates": 21,
        "start_date": "2025-04-01",
        "end_date": "2025-04-30",
        "output": "finra_short_volume/april_2025.csv.gz",
        "trailer_rule": "one-field trailer equals data-row count",
    }

    # The governed tables now hold every annual SOD snapshot 2019-2025 and every
    # FINRA trading day 2024-2025. The phase-6 slices survive as subsets, so the
    # original population checks are asserted on those subsets.
    branches = load_expansion_table("fdic_sod")
    assert not branches.duplicated(["report_date", "branch_id"]).any()
    assert sorted(branches.report_date.unique()) == [f"{y}-06-30" for y in range(2019, 2026)]
    snapshot_2025 = branches.loc[branches.report_date.eq("2025-06-30")]
    assert len(snapshot_2025) == 76097
    county_fips = branches.branch_county_fips.dropna()
    assert county_fips.str.fullmatch(r"\d{5}").all()

    finra = load_expansion_table("finra_short_volume")
    assert not finra.duplicated(["date", "symbol"]).any()
    assert finra.date.min() == "2024-01-02" and finra.date.max() == "2025-12-31"
    april = finra.loc[finra.date.str.startswith("2025-04")]
    assert len(april) == 219470
    assert april.date.nunique() == 21
    assert april.symbol.eq("NAN").sum() == 21


def test_phase6_runtime_exposes_only_requested_local_tables():
    variables = runtime_variables(
        {
            "fdic_bankfind",
            "fdic_sod",
            "sec_midas_security_exchange",
            "finra_short_volume",
        }
    )
    assert [variable.name for variable in variables] == [
        "fdic_bank_financials_df",
        "fdic_sod_branches_df",
        "finra_short_volume_df",
        "midas_security_exchange_df",
    ]


def test_jpm_branch_deposit_geography_anchor_values_and_label_validation():
    module = importlib.import_module(
        "evals.calculation_reconciliation.jpm_branch_deposit_geography_bridge"
    )
    truth = module.ground_truth()
    assert truth[:2] == (5000, 49)
    assert tuple(round(value, 6) for value in truth[2:7]) == (
        2669.161000,
        2132.981000,
        536.180000,
        2132.981000,
        0.000000,
    )
    assert truth[7:9] == ("NY", 577)
    assert round(truth[9], 4) == 11.54
    assert round(truth[10], 6) == 933.644201
    assert round(truth[11], 4) == 43.7718
    assert round(truth[12], 4) == 2364.9822
    expected = _expected_outputs(module)
    assert module.validate(
        {**expected, "largest_branch_deposit_state_usps": "  ny "}
    ).success
    assert not module.validate(
        {**expected, "largest_branch_deposit_state_usps": "CA"}
    ).success


def test_midas_finra_leader_anchor_values_are_stable():
    module = importlib.import_module(
        "evals.comparison_ranking.april_odd_lot_short_volume_leaders"
    )
    truth = module.ground_truth()
    assert truth[:4] == (
        21,
        6516,
        ["NVDA", "TSLA", "SPY", "PLTR", "AAPL"],
        "NVDA",
    )
    assert round(truth[4], 6) == 2.493175
    assert round(truth[5], 6) == 0.466144
    assert round(truth[6], 4) == 18.6968
    assert truth[7:9] == ("PLTR", 4)
    assert round(truth[9], 6) == 0.194595
    assert round(truth[10], 4) == 22.0100
    assert tuple(round(value, 6) for value in truth[11:14]) == (
        0.738257,
        0.010968,
        1.237594,
    )
    assert round(truth[14], 4) == 59.6526
    assert module.validate(_expected_outputs(module)).success
    assert not module.validate(
        {**_expected_outputs(module), "full_coverage_common_ticker_count": 6502}
    ).success


def test_phase6_case_source_removal_contract_is_explicit():
    registry = json.loads((PROJECT_ROOT / "benchmarks.json").read_text())
    paths = registry["cases"]
    assert paths["jpm_branch_deposit_geography_bridge"] == (
        "evals/calculation_reconciliation/jpm_branch_deposit_geography_bridge.json"
    )
    assert paths["april_odd_lot_short_volume_leaders"] == (
        "evals/comparison_ranking/april_odd_lot_short_volume_leaders.json"
    )
    jpm = json.loads(
        (PROJECT_ROOT / "evals/calculation_reconciliation/jpm_branch_deposit_geography_bridge.json").read_text()
    )
    market = json.loads(
        (PROJECT_ROOT / "evals/comparison_ranking/april_odd_lot_short_volume_leaders.json").read_text()
    )
    assert jpm["data_sources"] == ["fdic_bankfind", "fdic_sod"]
    assert market["data_sources"] == [
        "sec_midas_security_exchange",
        "finra_short_volume",
    ]


def test_phase6_review_fixes_keep_output_and_claim_contracts_aligned():
    jpm_module = importlib.import_module(
        "evals.calculation_reconciliation.jpm_branch_deposit_geography_bridge"
    )
    assert "report_date" not in {variable.name for variable in jpm_module.variables}

    jpm = json.loads(
        (PROJECT_ROOT / "evals/calculation_reconciliation/jpm_branch_deposit_geography_bridge.json").read_text()
    )
    jpm_claim_modes = {claim["id"]: claim["mode"] for claim in _claims(jpm)}
    assert jpm_claim_modes["within_bank_state_concentration_meaning"] == "required"
    assert "48 states plus the District of Columbia" in " ".join(
        jpm["judge_reference"]["facts"]
    )

    market = json.loads(
        (PROJECT_ROOT / "evals/comparison_ranking/april_odd_lot_short_volume_leaders.json").read_text()
    )
    market_turn = _turn(market)
    assert "dot-form share-class ticker" in market_turn["query"]
    assert "period separator with a slash" in market_turn["query"]
    assert "not adding it again to the numerator" not in market_turn["query"]
    assert "excluding short-exempt volume from the numerator" not in market_turn["query"]
    market_claim_modes = {claim["id"]: claim["mode"] for claim in _claims(market)}
    assert market_claim_modes["short_exempt_subset_semantics"] == "required"
    assert market_claim_modes["cross_venue_coverage_boundary"] == "required"
    assert market_claim_modes["market_quality_causality_boundary"] == "conditional"


def test_finra_short_volume_semantics_are_recorded_in_runtime_and_provenance():
    finra_variable = next(
        variable
        for variable in runtime_variables({"finra_short_volume"})
        if variable.name == "finra_short_volume_df"
    )
    assert "ShortExemptVolume is the separately reported exempt subset" in finra_variable.description

    manifest = json.loads(
        (DATASETS_DIR / "finra_short_volume/sources.json").read_text()
    )
    assert manifest["field_documentation_url"].endswith(
        "DailyShortSaleVolumeFileLayout.pdf"
    )
