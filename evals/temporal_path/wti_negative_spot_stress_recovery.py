"""Trace the sharpest five-observation WTI spot decline and its recovery.

Across 6,514 joint EIA/OFR dates from 2000 through 2025, the most negative
ending-minus-starting WTI level change over five consecutive joint observations
runs from 14 April to 20 April 2020: $20.15 to -$36.98 per barrel, or -$57.13.
The first later joint observation at or above the starting level is 4 May 2020.

This is a hard baseline.  The query pins joint reported observations, a price-
level change rather than a return, the five-observation window and the recovery
threshold.  Calendar-day windows, synthesized weekends, percentage/log returns
through a nonpositive price and futures-settlement interpretations are measured
regression alternatives or semantic overclaims, not live traps.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


TURN_1_NAMES = [
    "joint_wti_ofr_observation_count", "selloff_window_start_date", "selloff_window_end_date",
    "selloff_start_wti_usd_per_barrel", "selloff_end_wti_usd_per_barrel",
    "selloff_magnitude_margin_usd_per_barrel",
]
TURN_2_NAMES = [
    "selloff_second_observation_date", "selloff_second_wti_usd_per_barrel",
    "selloff_third_observation_date", "selloff_third_wti_usd_per_barrel",
    "selloff_fourth_observation_date", "selloff_fourth_wti_usd_per_barrel",
    "selloff_trough_date", "selloff_trough_wti_usd_per_barrel",
]
TURN_3_NAMES = [
    "selloff_start_ofr_fsi", "selloff_end_ofr_fsi", "selloff_credit_contribution_change",
    "selloff_equity_valuation_contribution_change", "selloff_safe_assets_contribution_change",
    "selloff_funding_contribution_change", "selloff_volatility_contribution_change",
    "largest_positive_contribution_change_category", "largest_positive_contribution_change",
]
TURN_4_NAMES = [
    "wti_recovery_date", "wti_recovery_price_usd_per_barrel", "wti_recovery_calendar_days",
    "wti_recovery_later_joint_observation_count", "recovery_date_ofr_fsi",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("joint_wti_ofr_observation_count", "Store the joint observation count as an integer."),
    _v("selloff_window_start_date", "Store the selected window start date as an ISO YYYY-MM-DD string."),
    _v("selloff_window_end_date", "Store the selected window end date as an ISO YYYY-MM-DD string."),
    _v("selloff_start_wti_usd_per_barrel", "Store the starting WTI spot price in USD per barrel, rounded to 2 decimals."),
    _v("selloff_end_wti_usd_per_barrel", "Store the ending WTI spot price in USD per barrel, rounded to 2 decimals."),
    _v("selloff_magnitude_margin_usd_per_barrel", "Store the selected decline-magnitude margin over the runner-up in USD per barrel, rounded to 2 decimals."),
    _v("selloff_second_observation_date", "Store the second window observation date as an ISO YYYY-MM-DD string."),
    _v("selloff_second_wti_usd_per_barrel", "Store the second WTI observation in USD per barrel, rounded to 2 decimals."),
    _v("selloff_third_observation_date", "Store the third window observation date as an ISO YYYY-MM-DD string."),
    _v("selloff_third_wti_usd_per_barrel", "Store the third WTI observation in USD per barrel, rounded to 2 decimals."),
    _v("selloff_fourth_observation_date", "Store the fourth window observation date as an ISO YYYY-MM-DD string."),
    _v("selloff_fourth_wti_usd_per_barrel", "Store the fourth WTI observation in USD per barrel, rounded to 2 decimals."),
    _v("selloff_trough_date", "Store the window trough date as an ISO YYYY-MM-DD string."),
    _v("selloff_trough_wti_usd_per_barrel", "Store the window trough WTI price in USD per barrel, rounded to 2 decimals."),
    _v("selloff_start_ofr_fsi", "Store starting OFR FSI, rounded to 3 decimals."),
    _v("selloff_end_ofr_fsi", "Store ending OFR FSI, rounded to 3 decimals."),
    _v("selloff_credit_contribution_change", "Store ending minus starting Credit contribution, rounded to 3 decimals."),
    _v("selloff_equity_valuation_contribution_change", "Store ending minus starting Equity valuation contribution, rounded to 3 decimals."),
    _v("selloff_safe_assets_contribution_change", "Store ending minus starting Safe assets contribution, rounded to 3 decimals."),
    _v("selloff_funding_contribution_change", "Store ending minus starting Funding contribution, rounded to 3 decimals."),
    _v("selloff_volatility_contribution_change", "Store ending minus starting Volatility contribution, rounded to 3 decimals."),
    _v("largest_positive_contribution_change_category", "Store the market category with the largest contribution increase."),
    _v("largest_positive_contribution_change", "Store that contribution increase, rounded to 3 decimals."),
    _v("wti_recovery_date", "Store the first qualifying recovery date as an ISO YYYY-MM-DD string."),
    _v("wti_recovery_price_usd_per_barrel", "Store the recovery-date WTI spot price in USD per barrel, rounded to 2 decimals."),
    _v("wti_recovery_calendar_days", "Store elapsed calendar days after the trough as an integer."),
    _v("wti_recovery_later_joint_observation_count", "Store the number of later joint observations through the recovery observation as an integer."),
    _v("recovery_date_ofr_fsi", "Store recovery-date OFR FSI, rounded to 3 decimals."),
]


@lru_cache(maxsize=1)
def _path():
    wti = load_expansion_table("eia_bulk_wti_daily").rename(columns={
        "observation_date": "date", "wti_spot_price_usd_per_barrel": "wti"
    })
    fsi = load_expansion_table("ofr_market_stress")
    joint = wti[["date", "wti"]].merge(fsi, on="date", how="inner")
    joint = joint.loc[joint.date.between("2000-01-01", "2025-12-31")].sort_values("date").reset_index(drop=True)
    joint["five_change"] = joint.wti - joint.wti.shift(4)
    ranking = joint.dropna(subset=["five_change"]).sort_values(["five_change", "date"]).copy()
    end_index = int(ranking.index[0])
    window = joint.loc[end_index - 4:end_index].copy()
    trough = window.sort_values(["wti", "date"]).iloc[0]
    start_price = float(window.iloc[0].wti)
    recovery = joint.loc[(joint.index > trough.name) & joint.wti.ge(start_price)].iloc[0]
    return joint, ranking, window, trough, recovery


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    joint, ranking, window, trough, recovery = _path()
    start = window.iloc[0]
    end = window.iloc[-1]
    category_columns = {
        "Credit": "credit", "Equity valuation": "equity_valuation",
        "Safe assets": "safe_assets", "Funding": "funding", "Volatility": "volatility",
    }
    changes = {label: float(end[column] - start[column]) for label, column in category_columns.items()}
    largest_category = sorted(changes, key=lambda label: (-changes[label], label))[0]
    return (
        len(joint),
        str(start.date),
        str(end.date),
        float(start.wti),
        float(end.wti),
        float(ranking.iloc[1].five_change - ranking.iloc[0].five_change),
        str(window.iloc[1].date),
        float(window.iloc[1].wti),
        str(window.iloc[2].date),
        float(window.iloc[2].wti),
        str(window.iloc[3].date),
        float(window.iloc[3].wti),
        str(trough.date),
        float(trough.wti),
        float(start.ofr_fsi),
        float(end.ofr_fsi),
        changes["Credit"],
        changes["Equity valuation"],
        changes["Safe assets"],
        changes["Funding"],
        changes["Volatility"],
        largest_category,
        changes[largest_category],
        str(recovery.date),
        float(recovery.wti),
        int((pd.Timestamp(recovery.date) - pd.Timestamp(trough.date)).days),
        int(recovery.name - trough.name),
        float(recovery.ofr_fsi),
    )


DECIMALS = [
    0, None, None, 2, 2, 2,
    None, 2, None, 2, None, 2, None, 2,
    3, 3, 3, 3, 3, 3, 3, None, 3,
    None, 2, 0, 0, 3,
]


def _validate_subset(outputs, names):
    by_name = {v.name:v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    places = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [places[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_selloff_screen":turn_validator(validate_turn_1),
    "validate_selloff_path":turn_validator(validate_turn_2),
    "validate_stress_path":turn_validator(validate_turn_3),
    "validate_recovery_path":turn_validator(validate_turn_4),
}
