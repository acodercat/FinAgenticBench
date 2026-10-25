"""Reconcile Schedule SB FTAP and test sensitivity to asset and balance choices.

Schedule SB's funding target attainment percentage (FTAP) generally subtracts the
prefunding and funding standard carryover balances from actuarial assets before
dividing by the funding target. In the selected 2023 population, the reconstructed
formula closely matches the filed FTAP and materially differs from an unadjusted
assets-to-target ratio.

The 80 and 60 percent counts in this case are mechanical FTAP sensitivity screens,
not determinations that IRC section 436 restrictions applied. Those restrictions
use the adjusted funding target attainment percentage (AFTAP), whose certification
can include annuity-purchase and other adjustments, timing rules and presumptions
not present in the two declared tables.

Current-value assets provide a second counterfactual, not an alternative statutory
restriction measure. Comparing them with actuarial assets quantifies smoothing
sensitivity. Most valuations are dated at the beginning of the plan year, and the
Schedule SB current value aligns much more closely with beginning-than ending-year
financial-statement assets, but the aggregate match does not isolate investment
return, contributions or other causes of the change.

Measured rejected alternatives include
subtracting only one funding balance, substituting current-value assets, retaining
multiple filings per plan, broadening filing status and changing the reconciliation
tolerance. The query pins these choices, so they are baseline instruction-following
probes rather than a live hidden-method trap. This is a hard baseline case.

Boundaries: four filed FTAP values use the 999.99 reporting cap, one record reports
a prefunding balance larger than actuarial assets, and the reconstructed FTAP does
not exactly match every certification. The case reports the mechanically defined
population and does not diagnose those records or infer actual benefit restrictions.
Numeric validation uses the shared rounding tolerance; counts match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


PLAN_YEAR_END = "2023-12-31"
VALUATION_FIRST_DAY = "2023-01-01"
RECEIVED_STATUS = "FILING_RECEIVED"
RESTRICTION_THRESHOLD = 80.0
ACCRUAL_THRESHOLD = 60.0
CERTIFICATION_TOLERANCE = 0.05
ASSET_MATCH_TOLERANCE = 0.01


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["plan_count", "credit_balance_plans", "credit_balance_share_pct",
                "median_credit_balance_usd_millions", "total_credit_balance_usd_billions",
                "median_unadjusted_ratio_pct"]
TURN_2_NAMES = ["median_statutory_ratio_pct", "median_overstatement_pp", "unadjusted_agreement_pct",
                "unadjusted_median_gap_pp", "statutory_agreement_pct", "statutory_median_gap_pp"]
TURN_3_NAMES = ["unadjusted_below_eighty", "statutory_below_eighty", "wrongly_cleared_eighty",
                "unadjusted_below_sixty", "statutory_below_sixty", "wrongly_cleared_sixty"]
TURN_4_NAMES = ["median_market_ratio_pct", "smoothing_gap_pp", "market_below_eighty",
    "actuarial_above_eighty_market_below", "first_day_valuations", "matched_plans",
                "beginning_asset_agreement_pct", "median_ending_asset_ratio"]

variables = [
    _v(TURN_1_NAMES[0], "Store how many plans the population holds as an integer."),
    _v(TURN_1_NAMES[1], "Store how many carry a positive credit balance as an integer."),
    _v(TURN_1_NAMES[2], "Store that as a percent of the population, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3], "Store the median positive credit balance in USD millions, rounded to 4 decimals."),
    _v(TURN_1_NAMES[4], "Store the total credit balance across the population in USD billions, rounded to 4 decimals."),
    _v(TURN_1_NAMES[5], "Store the median ratio of actuarial assets to funding target, in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the median statutory attainment percentage, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store the median difference between the two ratios in percentage points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the percent of plans where the unadjusted ratio reproduces the certified figure, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store the median absolute gap of the unadjusted ratio in percentage points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store the percent where the statutory ratio reproduces it, rounded to 4 decimals."),
    _v(TURN_2_NAMES[5], "Store its median absolute gap in percentage points, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store how many plans the unadjusted ratio places below 80 percent as an integer."),
    _v(TURN_3_NAMES[1], "Store how many the statutory ratio places there as an integer."),
    _v(TURN_3_NAMES[2], "Store how many the unadjusted ratio places at or above 80 percent while the reconstructed FTAP is below it, as an integer."),
    _v(TURN_3_NAMES[3], "Store how many the unadjusted ratio places below 60 percent as an integer."),
    _v(TURN_3_NAMES[4], "Store how many the statutory ratio places there as an integer."),
    _v(TURN_3_NAMES[5], "Store how many the unadjusted ratio places at or above 60 percent while the reconstructed FTAP is below it, as an integer."),
    _v(TURN_4_NAMES[0], "Store the median ratio built on current value assets, in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[1], "Store the median difference from the statutory ratio in percentage points, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store how many plans fall below 80 percent on current value as an integer."),
    _v(TURN_4_NAMES[3], "Store how many are at least 80 percent on actuarial-value FTAP but below it on current value, as an integer."),
    _v(TURN_4_NAMES[4], "Store how many plans carry a valuation date of the first day of the plan year as an integer."),
    _v(TURN_4_NAMES[5], "Store how many plans match a plan financial statement with positive assets at both ends as an integer."),
    _v(TURN_4_NAMES[6], "Store the percent whose current value assets equal beginning net assets within one percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[7], "Store the median ratio of current value assets to ending net assets, rounded to 4 decimals."),
]

DECIMALS = [0, 0, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4,
            0, 0, 0, 0, 0, 0,
            4, 4, 0, 0, 0, 0, 4, 4]

NUMERIC_COLUMNS = (
    "current_value_assets_usd", "actuarial_value_assets_usd", "total_funding_target_usd",
    "funding_standard_carryover_balance_usd", "prefunding_balance_usd", "reported_ftap_pct",
)


@lru_cache(maxsize=1)
def _plans():
    frame = load_expansion_table("dol_form5500_schedule_sb").copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    kept = frame.loc[
        frame.plan_year_end.astype(str).eq(PLAN_YEAR_END)
        & frame.filing_status.astype(str).eq(RECEIVED_STATUS)
        & frame.total_funding_target_usd.gt(0)
        & frame.actuarial_value_assets_usd.notna()
        & frame.reported_ftap_pct.notna()
    ].copy()
    kept = kept.sort_values(["sponsor_ein", "plan_number", "date_received", "filing_id"])
    kept = kept.drop_duplicates(["sponsor_ein", "plan_number"], keep="last")
    kept["credit"] = (
        kept.prefunding_balance_usd.fillna(0) + kept.funding_standard_carryover_balance_usd.fillna(0)
    )
    kept["unadjusted"] = kept.actuarial_value_assets_usd / kept.total_funding_target_usd * 100
    kept["statutory"] = (kept.actuarial_value_assets_usd - kept.credit) / kept.total_funding_target_usd * 100
    kept["market"] = (kept.current_value_assets_usd - kept.credit) / kept.total_funding_target_usd * 100
    kept["valuation"] = pd.to_datetime(kept.valuation_date, errors="coerce")
    return kept


@lru_cache(maxsize=1)
def _matched():
    financials = load_expansion_table("dol_form5500_financials").copy()
    for column in ("net_assets_beginning_usd", "net_assets_ending_usd"):
        financials[column] = pd.to_numeric(financials[column], errors="coerce")
    joined = _plans().merge(
        financials[["filing_id", "net_assets_beginning_usd", "net_assets_ending_usd"]],
        on="filing_id", how="left",
    )
    return joined.loc[
        joined.net_assets_beginning_usd.gt(0)
        & joined.net_assets_ending_usd.gt(0)
        & joined.current_value_assets_usd.gt(0)
    ].copy()


@lru_cache(maxsize=1)
def ground_truth():
    plans = _plans()
    matched = _matched()
    holders = plans.loc[plans.credit.gt(0)]
    certified = plans.reported_ftap_pct

    def agreement(series):
        gap = (series - certified).abs()
        return float(gap.lt(CERTIFICATION_TOLERANCE).mean()) * 100, float(gap.median())

    unadjusted_agreement, unadjusted_gap = agreement(plans.unadjusted)
    statutory_agreement, statutory_gap = agreement(plans.statutory)

    beginning_ratio = matched.current_value_assets_usd / matched.net_assets_beginning_usd
    ending_ratio = matched.current_value_assets_usd / matched.net_assets_ending_usd

    return (
        int(len(plans)), int(len(holders)), float(plans.credit.gt(0).mean()) * 100,
        float(holders.credit.median()) / 1e6, float(plans.credit.sum()) / 1e9,
        float(plans.unadjusted.median()),
        float(plans.statutory.median()), float((plans.unadjusted - plans.statutory).median()),
        unadjusted_agreement, unadjusted_gap, statutory_agreement, statutory_gap,
        int(plans.unadjusted.lt(RESTRICTION_THRESHOLD).sum()),
        int(plans.statutory.lt(RESTRICTION_THRESHOLD).sum()),
        int((plans.unadjusted.ge(RESTRICTION_THRESHOLD) & plans.statutory.lt(RESTRICTION_THRESHOLD)).sum()),
        int(plans.unadjusted.lt(ACCRUAL_THRESHOLD).sum()),
        int(plans.statutory.lt(ACCRUAL_THRESHOLD).sum()),
        int((plans.unadjusted.ge(ACCRUAL_THRESHOLD) & plans.statutory.lt(ACCRUAL_THRESHOLD)).sum()),
        float(plans.market.median()), float((plans.statutory - plans.market).median()),
        int(plans.market.lt(RESTRICTION_THRESHOLD).sum()),
        int((plans.statutory.ge(RESTRICTION_THRESHOLD) & plans.market.lt(RESTRICTION_THRESHOLD)).sum()),
        int(plans.valuation.eq(pd.Timestamp(VALUATION_FIRST_DAY)).sum()),
        int(len(matched)),
        float(beginning_ratio.sub(1).abs().lt(ASSET_MATCH_TOLERANCE).mean()) * 100,
        float(ending_ratio.median()),
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_credit_balance_scale": turn_validator(validate_turn_1),
    "validate_statutory_formula": turn_validator(validate_turn_2),
    "validate_restriction_thresholds": turn_validator(validate_turn_3),
    "validate_asset_measure_choices": turn_validator(validate_turn_4),
}
