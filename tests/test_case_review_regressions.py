import importlib
import json

from config import DATASETS_DIR, PROJECT_ROOT
from core.data import load_expansion_table, runtime_variables


def _expected_outputs(module):
    return {
        variable.name: value
        for variable, value in zip(module.variables, module.ground_truth())
    }


def _case_spec(relative_path):
    return json.loads((PROJECT_ROOT / relative_path).read_text())


def _primary_turn(spec):
    return spec["conversations"][0]["turns"][0]


def _claims(spec):
    """Every claim in the case, in turn order: a case may spread them over turns."""
    return [claim for turn in spec["conversations"][0]["turns"] for claim in turn["claims"]]


def test_reviewed_fhfa_population_and_claim_contract_is_explicit_without_join_recipe():
    spec = _case_spec(
        "evals/comparison_ranking/state_hpi_level_growth_leaders.json"
    )
    turn = _primary_turn(spec)
    assert "Across the 50 states and the District of Columbia" in turn["query"]
    assert "reducing it to one distinct" not in turn["query"]
    assert "before the join" not in turn["query"]
    claims = {claim["id"]: claim for claim in _claims(spec)}
    assert "time_grain_boundary" not in claims
    assert claims["observation_timing_inference"]["mode"] == "required"
    assert "both the 2025-Q4 FHFA HPI and the 2025 annual QCEW" in claims[
        "release_vintage_boundary"
    ]["description"]
    assert claims["qcew_finality_boundary"]["mode"] == "conditional"
    assert "preliminary annual vintage" in claims["qcew_finality_boundary"][
        "description"
    ]

    manifest = json.loads((DATASETS_DIR / "bls_qcew/sources.json").read_text())
    annual_2025 = next(
        item for item in manifest["raw_files"]
        if item["file"] == "raw/2025_annual_singlefile.zip"
    )
    assert "preliminary annual vintage" in annual_2025["note"]
    assert "2026-08-28" in annual_2025["note"]

    module = importlib.import_module(spec["module"])
    assert module.variables[0].description == (
        "Store the matched population count as an integer."
    )


def test_reviewed_treasury_claims_do_not_score_prompt_stock_flow_paraphrase():
    spec = _case_spec(
        "evals/comparison_ranking/notes_bonds_weighted_coupon.json"
    )
    claim_ids = {claim["id"] for claim in _claims(spec)}
    assert "stock_flow_distinction" not in claim_ids
    assert claim_ids == {
        "stock_weighted_coupon_basis",
        "coupon_yield_distinction",
        "reopening_lineage_meaning",
        "borrowing_cost_boundary",
    }


def test_pimco_curve_case_accepts_common_one_year_tenor_labels_only():
    module = importlib.import_module(
        "evals.scenario_analysis.pimco_usd_dv01_treasury_curve_scenario"
    )
    expected = _expected_outputs(module)
    for label in ("1 year", "1-year", "1 Yr", "one year"):
        assert module.validate(
            {**expected, "largest_absolute_contribution_tenor": label}
        ).success
    assert not module.validate(
        {**expected, "largest_absolute_contribution_tenor": "10 year"}
    ).success


def test_cross_source_case_anchor_values_are_stable():
    dv01 = importlib.import_module(
        "evals.scenario_analysis.pimco_usd_dv01_treasury_curve_scenario"
    )
    dv01_truth = dv01.ground_truth()
    assert tuple(round(value, 1) for value in dv01_truth[:5]) == (
        -35.0,
        -20.0,
        -1.0,
        2.0,
        11.0,
    )
    assert round(dv01_truth[5], 3) == 443.016
    assert round(dv01_truth[6], 3) == 496.750
    assert dv01_truth[7] == "1 year"
    assert round(dv01_truth[8], 3) == 405.324
    assert round(dv01_truth[9], 4) == 0.2187

    hpi = importlib.import_module(
        "evals.comparison_ranking.state_hpi_level_growth_leaders"
    )
    hpi_truth = hpi.ground_truth()
    assert hpi_truth[0] == 51
    assert (hpi_truth[1], hpi_truth[6], hpi_truth[11]) == (
        "Montana",
        "Delaware",
        "Idaho",
    )

    treasury = importlib.import_module(
        "evals.comparison_ranking.notes_bonds_weighted_coupon"
    )
    treasury_truth = treasury.ground_truth()
    assert (treasury_truth[4], treasury_truth[5]) == (60, 9)
    assert (treasury_truth[8], treasury_truth[9]) == (24, 16)

    sba = importlib.import_module(
        "evals.calculation_reconciliation.five_o_four_financing_mix"
    )
    sba_truth = sba.ground_truth()
    assert sba_truth[:2] == (51, 2135)
    assert sba_truth[7:10] == ("SD", 30, 38496)
    assert round(sba_truth[5], 4) == 43.0992
    assert round(sba_truth[10], 4) == 7.7930

    redemption = importlib.import_module(
        "evals.risk_screening.fedfund_peak_redemption_liquidity"
    )
    redemption_truth = redemption.ground_truth()
    assert redemption_truth[:2] == ("2025-04-09", 15)
    assert redemption_truth[6:9] == (1.934, 2.579, -0.6450000000000001)
    assert redemption_truth[9:13] == (22, "2025-04-07", 2.743, 3)


def test_new_cross_source_cases_expose_all_required_local_runtime_tables():
    variables = {
        variable.name: variable.value
        for variable in runtime_variables(
            {
                "sba_504",
                "census_geo_crosswalk",
                "bls_qcew",
                "sec_nmfp",
                "ofr_market_stress",
            }
        )
    }
    assert {
        "sba_five_o_four_loans_df",
        "census_counties_df",
        "bls_qcew_state_annual_df",
        "nmfp_daily_flows_df",
        "nmfp_daily_liquidity_df",
        "ofr_financial_stress_index_df",
    }.issubset(variables)
    assert variables["ofr_financial_stress_index_df"].date.is_unique
    assert variables["ofr_financial_stress_index_df"].date.max() == "2025-12-31"
    assert load_expansion_table("sba_504").equals(
        variables["sba_five_o_four_loans_df"]
    )


def test_sba_state_label_validation_normalizes_case_and_whitespace():
    module = importlib.import_module(
        "evals.calculation_reconciliation.five_o_four_financing_mix"
    )
    expected = _expected_outputs(module)
    assert module.validate(
        {**expected, "highest_intensity_state_usps": "  sd  "}
    ).success
    assert not module.validate(
        {**expected, "highest_intensity_state_usps": "UT"}
    ).success


def test_thirteen_f_excludes_restatements_filed_under_the_holdings_submission_type():
    """The submission type and the amendment flag disagree; both must be honoured.

    One restatement per period carries submission type 13F-HR with its own
    amendment flag set, so a filter on the type alone silently admits it.
    """
    module = importlib.import_module(
        "evals.calculation_reconciliation.thirteen_f_reporting_unit_break_ownership"
    )
    truth = _expected_outputs(module)
    assert truth["pre_break_filing_count"] == 6745
    assert truth["post_break_filing_count"] == 7066
    assert truth["post_break_thousand_holding_convention_count"] == 1308
    assert truth["post_break_thousand_cover_scale_count"] == 1267
    assert truth["filings_including_other_managers"] == 974

    filings = load_expansion_table("sec_13f_filings")
    admitted_by_type_alone = filings.loc[
        filings.submissiontype.astype(str).eq("13F-HR")
        & filings.isamendment.astype(str).str.upper().eq("Y")
        & filings.periodofreport.astype(str).isin(
            (module.PRE_BREAK_PERIOD, module.POST_BREAK_PERIOD)
        )
    ]
    assert not admitted_by_type_alone.empty, "the conflicting-field case is gone from the data"

    spec = _case_spec(
        "evals/calculation_reconciliation/thirteen_f_reporting_unit_break_ownership.json"
    )
    query = _primary_turn(spec)["query"]
    assert "Use Form 13F-HR submissions, including combination reports" in query
    assert "exclude notices and every amendment" in query
