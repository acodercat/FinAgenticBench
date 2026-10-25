from pathlib import Path

import pytest

from config import DATASETS_DIR
from core.dataset_layers import RAW_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = PROJECT_ROOT / "tests"
RUNTIME_TEST_FILES = {
    "test_reg_cf_offering_cap_case.py",
    "test_operating_lease_discount_rate_case.py",
    "test_pension_discount_rate_case.py",
    "test_tips_reference_cpi_case.py",
    "test_settlement_cycle_fails_case.py",
    "test_treasury_coupon_setting_case.py",
    "test_sofr_average_compounding_case.py",
    "test_cash_flow_restricted_cash_case.py",
    "test_treasury_bill_rate_basis_case.py",
    "test_volatility_time_scaling_case.py",
    "test_13f_option_underlier_boundary_case.py",
    "test_diluted_antidilution_eps_case.py",
    "test_leverage_definition_goodwill_case.py",
    "test_money_fund_class_flow_yield_path.py",
    "test_money_fund_liquidity_stock_flow_path.py",
    "test_bdc_commitment_mark_maturity_path.py",
    "test_effective_tax_rate_bridge_case.py",
    "test_hmda_sba_jobs_batch.py",
    "test_bank_htm_capital_recognition_case.py",
    "test_sba_strip_sale_lender_funding_case.py",
    "test_stress_tax_yield_knowledge_batch.py",
    "test_hmda_dti_encoding_case.py",
    "test_disaster_assistance_flood_loss_case.py",
    "test_sba_status_withheld_denominator_case.py",
    "test_nport_at_risk_knowledge_batch.py",
    "test_buyback_compensation_screen_case.py",
    "test_annual_fourth_quarter_derivation_case.py",
    "test_bdc_fiscal_label_alignment_case.py",
    "test_treasury_auction_dispersion_case.py",
    "test_insider_holdings_disclosure_case.py",
    "test_rcl_rcn_financial_knowledge_batch.py",
    "test_rcb_nport_lending_batch.py",
    "test_rcs_nport_repo_batch.py",
    "test_rce_time_deposit_case.py",
    "test_13f_amendment_public_position_case.py",
    "test_sba_bdc_financial_knowledge_batch.py",
    "test_hpi_estimation_error_case.py",
    "test_treasury_bill_repo_carry_advance_limit.py",
    "test_disaster_county_mortgage_collateral_paydown.py",
    "test_tips_accretion_bridge_case.py",
    "test_form_d_indefinite_offering_case.py",
    "test_sba_guarantee_retention_case.py",
    "test_branch_runoff_reg_cf_batch.py",
    "test_nfip_coverage_exhaustion_case.py",
    "test_new_case_batch_139_140.py",
    "test_new_case_batch_141_142.py",
    "test_new_case_batch_143_144.py",
    "test_new_case_batch_145_146.py",
    "test_new_case_batch_147_148.py",
    "test_new_case_batch_137_138.py",
    "test_new_case_batch_135_136.py",
    "test_new_case_batch_129_130.py",
    "test_new_case_batch_131_132.py",
    "test_new_case_batch_133_134.py",
    "test_capital_forecast_batch.py",
    "test_cash_payout_tips_batch.py",
    "test_issuance_lineage_batch.py",
    "test_disclosure_mortgage_candidates.py",
    "test_fund_risk_candidate.py",
    "test_group_divestiture_candidate.py",
    "test_state_preservation_group3.py",
    "test_state_preservation_group2.py",
    "test_state_preservation_group1.py",
    "test_raw_certification_quant_cases.py",
    "test_advanced_quant_v3_group4.py",
    "test_advanced_quant_v3_group3.py",
    "test_advanced_quant_v3_group2.py",
    "test_advanced_quant_v3_group1.py",
    "test_cox_smith_wilson.py",
    "test_strict_review_pilot.py",
    "test_new_case_batch_20260908.py",
    "test_advanced_quant_v2_group5.py",
    "test_advanced_quant_v2_group4.py",
    "test_advanced_quant_v2_group3.py",
    "test_advanced_quant_v2_group2.py",
    "test_advanced_quant_v2_group1.py",
    "test_bis_sba_multisource_cases.py",
    "test_case_review_regressions.py",
    "test_cases.py",
    "test_crosswalk_economic_sources.py",
    "test_dataset.py",
    "test_dataset_layers.py",
    "test_disaster_trade_multisource_cases.py",
    "test_eia_fema_multisource_cases.py",
    "test_fails_exposure_divergence_case.py",
    "test_ffiec_ofr_multisource_cases.py",
    "test_flood_metro_grain_case.py",
    "test_intermediate_rounding.py",
    "test_near_miss_probes.py",
    "test_form5500_ffiec_treasury_sources.py",
    "test_midas_fdic_multisource_cases.py",
    "test_multiturn_batch1_cases.py",
    "test_multiturn_batch2_cases.py",
    "test_multiturn_batch3_cases.py",
    "test_multiturn_batch4_cases.py",
    "test_multiturn_batch5_cases.py",
    "test_multiturn_expansion_cases.py",
    "test_population_cases_2026_08_27.py",
    "test_quantitative_finance_cases.py",
    "test_advanced_quantitative_cases.py",
    "test_advanced_quantitative_group2.py",
    "test_advanced_quantitative_group3.py",
    "test_advanced_quantitative_group4.py",
    "test_advanced_quantitative_group5.py",
    "test_rate_futures_succession_case.py",
    "test_treasury_cash_control_band_case.py",
    "test_fund_redemption_alignment_case.py",
    "test_industry_rate_beta_case.py",
    "test_mortgage_originator_retention_case.py",
    "test_sba_rate_ceiling_case.py",
    "test_allowance_working_capital_batch.py",
    "test_reinsurance_bdc_deleveraging_batch.py",
    "test_stress_trade_multisource_cases.py",
    "test_temporal_coverage.py",
    "test_z1_equity_decomposition_case.py",
    "test_verdict_baseline.py",
    "test_new_case_batch_137_138.py",
    "test_new_case_batch_135_136.py",
    "test_new_case_batch_129_130.py",
    "test_new_case_batch_131_132.py",
    "test_new_case_batch_133_134.py",
}


def _requires_runtime(item) -> bool:
    """Tests that read the governed runtime tables, named explicitly.

    The suite is a flat directory, so membership is by file name rather than by
    parent folder.  A new runtime-dependent test file must be added here or it
    will run, and fail, in a code-only checkout.
    """
    return Path(item.path).name in RUNTIME_TEST_FILES


def pytest_collection_modifyitems(config, items):
    runtime_materialized = (
        DATASETS_DIR / "sec/runtime/company_facts.csv.gz"
    ).is_file()
    raw_materialized = any(
        path.is_file()
        for path in RAW_DIR.glob("*/**/*")
    )
    for item in items:
        if _requires_runtime(item):
            item.add_marker(pytest.mark.runtime_data)
        if "runtime_data" in item.keywords and not runtime_materialized:
            item.add_marker(pytest.mark.skip(
                reason="governed runtime tables are not materialized"
            ))
        if "raw_data" in item.keywords and not raw_materialized:
            item.add_marker(pytest.mark.skip(
                reason="source-native raw archive is not materialized"
            ))
