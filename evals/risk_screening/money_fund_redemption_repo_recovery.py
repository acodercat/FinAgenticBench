"""Screen a money-fund redemption event and trace its one-month recovery.

Among 1,248 original 2025 N-MFP3 series-report observations with at least $1
billion net assets, a same-date flow, positive same-date daily liquid assets and
all three selected New York Fed benchmarks on the report date plus complete
adjacent-month original filings, TCW Central Cash Fund on 31 October has the
largest aggregate net-redemption-to-daily-liquid-assets ratio at 151.2804%.

The query pins original filings, series aggregation, positive denominator and
month-end dates, so largest-class, amended-filing, net-assets-denominator and
missing-as-zero alternatives are regression paths.  The case is therefore a
hard baseline rather than a live trap.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
RATE_TYPES = ("TGCR", "BGCR", "SOFR")

TURN_1_NAMES = [
    "eligible_large_fund_report_observation_count", "redemption_pressure_event_date",
    "redemption_pressure_leader_series_name", "redemption_pressure_leader_series_id",
    "redemption_pressure_leader_accession", "leader_flow_share_class_count",
    "leader_gross_subscriptions_usd_billions", "leader_gross_redemptions_usd_billions",
    "leader_daily_liquid_assets_usd_billions", "leader_redemption_liquidity_ratio_pct",
    "redemption_liquidity_ratio_leader_margin_pp",
]
TURN_2_NAMES = [
    "previous_month_end_selected_accession", "previous_month_end_net_assets_usd_billions",
    "previous_month_end_wam_days", "previous_month_end_wal_days",
    "event_net_assets_usd_billions", "event_wam_days", "event_wal_days",
    "following_month_end_selected_accession", "following_month_end_net_assets_usd_billions",
    "following_month_end_wam_days", "following_month_end_wal_days",
]
TURN_3_NAMES = [
    "event_day_tgcr_pct", "event_day_bgcr_pct", "event_day_sofr_pct",
    "event_day_sofr_first_percentile_pct", "event_day_sofr_ninety_ninth_percentile_pct",
]
TURN_4_NAMES = [
    "previous_to_event_net_asset_decline_pct", "event_to_following_net_asset_rebound_pct",
    "one_month_decline_recovered_pct", "event_to_following_wam_change_days",
    "event_to_following_wal_change_days",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("eligible_large_fund_report_observation_count", "Store the eligible series-report observation count as an integer."),
    _v("redemption_pressure_event_date", "Store the selected report date as an ISO YYYY-MM-DD string."),
    _v("redemption_pressure_leader_series_name", "Store the selected series name exactly as reported."),
    _v("redemption_pressure_leader_series_id", "Store the selected SEC series identifier as a string."),
    _v("redemption_pressure_leader_accession", "Store the selected event accession number as a string."),
    _v("leader_flow_share_class_count", "Store the contributing share-class count as an integer."),
    _v("leader_gross_subscriptions_usd_billions", "Store gross subscriptions in USD billions, rounded to 6 decimals."),
    _v("leader_gross_redemptions_usd_billions", "Store gross redemptions in USD billions, rounded to 6 decimals."),
    _v("leader_daily_liquid_assets_usd_billions", "Store same-date daily liquid assets in USD billions, rounded to 6 decimals."),
    _v("leader_redemption_liquidity_ratio_pct", "Store net redemption divided by daily liquid assets in percent, rounded to 4 decimals."),
    _v("redemption_liquidity_ratio_leader_margin_pp", "Store the leader's margin over the runner-up ratio in percentage points, rounded to 4 decimals."),
    _v("previous_month_end_selected_accession", "Store the preceding month-end accession number as a string."),
    _v("previous_month_end_net_assets_usd_billions", "Store preceding month-end net assets in USD billions, rounded to 4 decimals."),
    _v("previous_month_end_wam_days", "Store preceding month-end weighted-average maturity in days as an integer."),
    _v("previous_month_end_wal_days", "Store preceding month-end weighted-average life in days as an integer."),
    _v("event_net_assets_usd_billions", "Store selected-event net assets in USD billions, rounded to 4 decimals."),
    _v("event_wam_days", "Store selected-event weighted-average maturity in days as an integer."),
    _v("event_wal_days", "Store selected-event weighted-average life in days as an integer."),
    _v("following_month_end_selected_accession", "Store the following month-end accession number as a string."),
    _v("following_month_end_net_assets_usd_billions", "Store following month-end net assets in USD billions, rounded to 4 decimals."),
    _v("following_month_end_wam_days", "Store following month-end weighted-average maturity in days as an integer."),
    _v("following_month_end_wal_days", "Store following month-end weighted-average life in days as an integer."),
    _v("event_day_tgcr_pct", "Store the event-day TGCR in percent, rounded to 2 decimals."),
    _v("event_day_bgcr_pct", "Store the event-day BGCR in percent, rounded to 2 decimals."),
    _v("event_day_sofr_pct", "Store the event-day SOFR in percent, rounded to 2 decimals."),
    _v("event_day_sofr_first_percentile_pct", "Store the event-day SOFR first percentile in percent, rounded to 2 decimals."),
    _v("event_day_sofr_ninety_ninth_percentile_pct", "Store the event-day SOFR ninety-ninth percentile in percent, rounded to 2 decimals."),
    _v("previous_to_event_net_asset_decline_pct", "Store the previous-to-event decline as a percent of preceding month-end net assets, rounded to 4 decimals."),
    _v("event_to_following_net_asset_rebound_pct", "Store the event-to-following rebound as a percent of event net assets, rounded to 4 decimals."),
    _v("one_month_decline_recovered_pct", "Store the rebound divided by the prior decline in percent, rounded to 4 decimals."),
    _v("event_to_following_wam_change_days", "Store following month-end minus event weighted-average maturity in days as an integer."),
    _v("event_to_following_wal_change_days", "Store following month-end minus event weighted-average life in days as an integer."),
]


def _one(frame, label):
    if len(frame) != 1:
        raise ValueError(f"expected one {label} row, found {len(frame)}")
    return frame.iloc[0]


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    series = load_expansion_table("sec_nmfp")
    flows = load_expansion_table("sec_nmfp_flows")
    liquidity = load_expansion_table("sec_nmfp_liquidity")
    rates = load_expansion_table("nyfed_reference_rates")
    rate_counts = rates.loc[rates["Rate Type"].isin(RATE_TYPES)].groupby(
        "Effective Date"
    )["Rate Type"].nunique()
    complete_rate_dates = set(rate_counts.loc[rate_counts.eq(len(RATE_TYPES))].index)
    candidates = series.loc[
        series.form.eq("N-MFP3") & series.report_date.between(START_DATE, END_DATE)
        & series.report_date.isin(complete_rate_dates)
        & series.net_assets_usd.ge(1_000_000_000)
    ].copy()
    original_keys = set(zip(
        series.loc[series.form.eq("N-MFP3"), "series_id"],
        series.loc[series.form.eq("N-MFP3"), "report_date"],
    ))
    candidate_periods = pd.to_datetime(candidates.report_date).dt.to_period("M")
    candidates["previous_report_date"] = candidate_periods.map(
        lambda period: (period - 1).end_time.strftime("%Y-%m-%d")
    )
    candidates["following_report_date"] = candidate_periods.map(
        lambda period: (period + 1).end_time.strftime("%Y-%m-%d")
    )
    candidates = candidates.loc[
        candidates.apply(
            lambda row: (row.series_id, row.previous_report_date) in original_keys
            and (row.series_id, row.following_report_date) in original_keys,
            axis=1,
        )
    ].copy()
    flow_summary = flows.groupby(["accession", "flow_date"]).agg(
        gross_subscriptions=("gross_subscriptions_usd", "sum"),
        gross_redemptions=("gross_redemptions_usd", "sum"),
        class_count=("class_id", "nunique"),
    ).reset_index()
    flow_summary["net_redemption"] = flow_summary.gross_redemptions - flow_summary.gross_subscriptions
    liquid = liquidity[["accession", "liquidity_date", "daily_liquid_assets_usd"]]
    cohort = candidates.merge(
        flow_summary,
        left_on=["accession", "report_date"],
        right_on=["accession", "flow_date"],
        how="inner",
    ).merge(
        liquid,
        left_on=["accession", "report_date"],
        right_on=["accession", "liquidity_date"],
        how="inner",
    )
    cohort = cohort.loc[cohort.daily_liquid_assets_usd.gt(0)].copy()
    cohort["pressure_ratio"] = cohort.net_redemption / cohort.daily_liquid_assets_usd * 100
    cohort = cohort.sort_values(
        ["pressure_ratio", "report_date", "series_name", "series_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    leader = cohort.iloc[0]

    event_date = pd.Timestamp(leader.report_date)
    event_period = event_date.to_period("M")
    report_dates = (
        (event_period - 1).end_time.strftime("%Y-%m-%d"),
        event_date.strftime("%Y-%m-%d"),
        (event_period + 1).end_time.strftime("%Y-%m-%d"),
    )
    path = series.loc[
        series.series_id.eq(leader.series_id) & series.form.eq("N-MFP3")
        & series.report_date.isin(report_dates)
    ].sort_values("report_date")
    if list(path.report_date) != list(report_dates):
        raise ValueError("selected series lacks one original monthly filing")
    september, october_row, november = (path.iloc[index] for index in range(3))

    selected_rates = rates.loc[
        rates["Effective Date"].eq(event_date.strftime("%Y-%m-%d"))
    ]
    tgcr = _one(selected_rates.loc[selected_rates["Rate Type"].eq("TGCR")], "TGCR")
    bgcr = _one(selected_rates.loc[selected_rates["Rate Type"].eq("BGCR")], "BGCR")
    sofr = _one(selected_rates.loc[selected_rates["Rate Type"].eq("SOFR")], "SOFR")

    decline = float(september.net_assets_usd - october_row.net_assets_usd)
    rebound = float(november.net_assets_usd - october_row.net_assets_usd)
    return (
        len(cohort),
        event_date.strftime("%Y-%m-%d"),
        str(leader.series_name),
        str(leader.series_id),
        str(leader.accession),
        int(leader.class_count),
        float(leader.gross_subscriptions) / 1e9,
        float(leader.gross_redemptions) / 1e9,
        float(leader.daily_liquid_assets_usd) / 1e9,
        float(leader.pressure_ratio),
        float(leader.pressure_ratio - cohort.iloc[1].pressure_ratio),
        str(september.accession),
        float(september.net_assets_usd) / 1e9,
        int(september.weighted_average_maturity_days),
        int(september.weighted_average_life_days),
        float(october_row.net_assets_usd) / 1e9,
        int(october_row.weighted_average_maturity_days),
        int(october_row.weighted_average_life_days),
        str(november.accession),
        float(november.net_assets_usd) / 1e9,
        int(november.weighted_average_maturity_days),
        int(november.weighted_average_life_days),
        float(tgcr["Rate (%)"]),
        float(bgcr["Rate (%)"]),
        float(sofr["Rate (%)"]),
        float(sofr["1st Percentile (%)"]),
        float(sofr["99th Percentile (%)"]),
        decline / float(september.net_assets_usd) * 100,
        rebound / float(october_row.net_assets_usd) * 100,
        rebound / decline * 100,
        int(november.weighted_average_maturity_days - october_row.weighted_average_maturity_days),
        int(november.weighted_average_life_days - october_row.weighted_average_life_days),
    )


DECIMALS = [
    0, None, None, None, None, 0, 6, 6, 6, 4, 4,
    None, 4, 0, 0, 4, 0, 0, None, 4, 0, 0,
    2, 2, 2, 2, 2,
    4, 4, 4, 0, 0,
]


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    places = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [places[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_pressure_screen": turn_validator(validate_turn_1),
    "validate_monthly_path": turn_validator(validate_turn_2),
    "validate_repo_snapshot": turn_validator(validate_turn_3),
    "validate_recovery": turn_validator(validate_turn_4),
}
