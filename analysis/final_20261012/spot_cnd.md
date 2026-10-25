# Computed-but-not-delivered detector: manual spot check (2026-10-13)

Detector (final rule, `turn_flags.py`): for a wrong numeric output whose reference has at least four significant digits, any of the reference's printed forms (as stored, `%g`, 2/3/4/6-decimal roundings for non-integers, thousands-separated for |x| >= 1000) occurs in the tool output of the same turn, delimited so that no digit, sign or decimal point touches it on the left and no digit on the right. Only tool returns are searched; the query, system prompt, harness continuation prompts and the model's own text are not.

Result: 438 / wrong numeric outputs 10634. The earlier whole-session substring rule (kept as `n_cnd_loose`, not reported) flagged 1,175 of the same outputs.

## Sample A (first rule, before the significant-digit fix): 20 hits

19 true positives; 1 false positive (`eligible_disaster_screened_state_count`, expected 11, matched the padded form `11.000000` of an unrelated coverage percentage). The fix counts significant digits after dropping leading zeros and no longer pads integer references into decimal forms, so this hit no longer fires. Among the 20 non-hits of sample A, no value with four or more significant digits appeared in the turn's tool output at three-significant-digit precision; the three near matches (95.43 for 95.3703; `41,` for 41; `34` for 34) are either a different value or a short integer the rule excludes by design.

## Sample B (final rule): 20 hits, seed 20261013

19 true positives, 1 false positive. Precision on the two samples combined: 38/40 = 95%.

- `five_o_four_financing_mix` turn 1 `aggregate_third_party_dollars_usd_billions` (qwen3.8-flash-nonthinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `flood_occurrence_reinstatement_aggregate_cover` turn 2 `helene_occurrence_recovery_m` (qwen3.8-27b-nonthinking-20261008): true positive: the value is printed as a computed candidate for this quantity
- `bank_acquisition_goodwill_nci_capital_bridge` turn 3 `new_loan_rwa_capacity_mn` (deepseek-v4-pro-0813-thinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `bank_run_liquidity_equity_frontier` turn 5 `third_ending_equity_assets_pct` (qwen3.8-flash-nonthinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `volatility_time_scaling_autocorrelation_regimes` turn 4 `early_monthly_sharpe` (qwen3.8-flash-nonthinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `xbrl_dimensional_revenue_double_count` turn 1 `dimensioned_filings` (gemini-3.8-flash-thinking-20261007): true positive: the value is printed as a computed candidate for this quantity
- `industry_beta_crisis_prediction_failure` turn 1 `lowest_beta` (qwen3.8-27b-nonthinking-20261008): true positive: the value is printed as a computed candidate for this quantity
- `apple_commitment_curve_discount_stress` turn 1 `gross_long_term_debt_usd_billions` (deepseek-v4-flash-thinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `money_market_share_class_fee_wedge` turn 1 `class_rows` (gpt-6-luna-nonthinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `fund_redemption_month_alignment_and_sign` turn 1 `total_gross_redeemed_usd_billions` (qwen3.8-flash-nonthinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `flood_parametric_basis_risk_frozen_contract` turn 5 `revised_county_2_benefit_mn` (qwen3.8-flash-nonthinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `april_odd_lot_short_volume_leaders` turn 1 `full_coverage_common_ticker_count` (qwen3.8-flash-thinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `bis_var_generalized_connectedness` turn 4 `girf_shock_size` (qwen3.8-flash-nonthinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `flood_occurrence_reinstatement_aggregate_cover` turn 2 `helene_occurrence_recovery_m` (deepseek-v4-flash-nonthinking-20261006): FALSE POSITIVE: 1500 appears as the cap parameter inside min(above, 1500, capacity), not as a computed value
- `sba_collateral_seniority_guarantee_loss_frontier` turn 4 `stochastic_504_mean_loss_mn` (qwen3.8-flash-thinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `bdc_fiscal_label_cross_section_alignment` turn 3 `label_fair_value_mn` (qwen3.8-flash-thinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `sba_guaranteed_strip_sale_and_lender_funding` turn 3 `large_bank_sold_share_pct` (deepseek-v4-pro-0813-thinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `treasury_delivery_basket_ctd_switch` turn 3 `base_boundary_quote` (qwen3.8-flash-thinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `bis_gaussian_copula_joint_stress` turn 4 `scenario_common_stress_score` (qwen3.8-flash-nonthinking-20261006): true positive: the value is printed as a computed candidate for this quantity
- `hmda_origination_vs_mortgage_transactions` turn 2 `cashout_refinance_bn` (deepseek-v4-flash-nonthinking-20261006): true positive: the value is printed as a computed candidate for this quantity

## Sample B: 20 non-hits

Listed in `spot_cnd_sample.md`; none has a four-significant-digit form of the reference in the turn's tool output (checked at three significant digits as in sample A). Recall is therefore limited mainly by the four-digit requirement, which excludes short counts and dates; the paper reports the detector as a lower bound on wrong values whose correct value the agent had printed.
