"""Update a pension-funding screen, reconcile its filing, and shock liabilities.

This four-turn baseline follows a realistic defined-benefit review.  A
2025-09-30 information cutoff selects Lumen from the Form 5500/Schedule SB
intersection; extending the cutoff through 2025-10-15 expands the eligible
underfunded cohort from four to 58 plans and selects Lockheed Martin, while the
unchanged Lumen filing falls to rank 26.  The selected filing's Schedule H net
assets roll forward exactly, but its ending net assets, Schedule SB current-value
assets and actuarial-value assets remain different measures.  Finally, a pinned
12-year first-order liability-duration sensitivity applies the observed 75 bp
rise in the 30-year Treasury par yield from 2023-12-29 to 2024-12-31.

The queries explicitly require re-running the cutoff screen, selecting the
latest successful filing per plan identity, preserving the two asset bases and
using a hypothetical Treasury proxy.  Reusing the September cohort, retaining
the earliest filing, including filing-error rows, substituting current-value
assets, or using the next observation after the holiday valuation date are
therefore measured convention probes rather than live traps.  Numeric validation
uses the shared 0.6 x 10^-N rounding-boundary tolerance.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


PLAN_YEAR_END = "2024-12-31"
EARLY_CUTOFF = "2025-09-30"
UPDATED_CUTOFF = "2025-10-15"
MINIMUM_PARTICIPANTS = 10_000
CURVE_END_DATE = "2024-12-31"
LIABILITY_DURATION = 12.0


TURN_1_NAMES = [
    "early_underfunded_plan_count",
    "early_shortfall_leader_plan_name",
    "early_shortfall_leader_sponsor_ein",
    "early_shortfall_leader_plan_number",
    "early_shortfall_leader_filing_id",
    "early_shortfall_leader_participant_count",
    "early_shortfall_leader_gap_usd_billions",
    "early_shortfall_leader_actuarial_funded_pct",
]
TURN_2_NAMES = [
    "updated_underfunded_plan_count",
    "updated_shortfall_leader_plan_name",
    "updated_shortfall_leader_sponsor_ein",
    "updated_shortfall_leader_plan_number",
    "updated_shortfall_leader_filing_id",
    "updated_shortfall_leader_participant_count",
    "updated_shortfall_leader_gap_usd_billions",
    "updated_shortfall_leader_actuarial_funded_pct",
    "early_leader_rank_at_updated_cutoff",
    "early_leader_gap_at_updated_cutoff_usd_billions",
]
TURN_3_NAMES = [
    "updated_leader_beginning_net_assets_usd_billions",
    "updated_leader_total_income_usd_billions", "updated_leader_total_expenses_usd_billions",
    "updated_leader_net_income_usd_billions", "updated_leader_transfers_to_plan_usd_billions",
    "updated_leader_transfers_from_plan_usd_billions",
    "updated_leader_ending_net_assets_usd_billions",
    "updated_leader_net_asset_rollforward_residual_usd_millions",
    "updated_leader_income_expense_residual_usd_millions",
    "updated_leader_sb_current_value_assets_usd_billions",
    "updated_leader_sb_actuarial_value_assets_usd_billions",
]
TURN_4_NAMES = [
    "treasury_curve_start_observation_date", "treasury_curve_start_thirty_year_yield_pct",
    "treasury_curve_end_thirty_year_yield_pct",
    "treasury_thirty_year_yield_change_basis_points", "first_order_funding_target_change_pct",
    "hypothetical_stressed_funding_target_usd_billions",
    "hypothetical_stressed_actuarial_funded_pct",
]


variables = [
    Variable("early_underfunded_plan_count", None, "Store the eligible underfunded-plan count at the first cutoff as an integer."),
    Variable("early_shortfall_leader_plan_name", None, "Store the first-cutoff leader's plan name exactly as reported."),
    Variable("early_shortfall_leader_sponsor_ein", None, "Store the first-cutoff leader's nine-digit sponsor EIN as a string."),
    Variable("early_shortfall_leader_plan_number", None, "Store the first-cutoff leader's official three-digit plan number as a string."),
    Variable("early_shortfall_leader_filing_id", None, "Store the first-cutoff leader's selected filing ID as a string."),
    Variable("early_shortfall_leader_participant_count", None, "Store the first-cutoff leader's Schedule SB participant count as an integer."),
    Variable("early_shortfall_leader_gap_usd_billions", None, "Store the first-cutoff leader's actuarial funding-target shortfall in USD billions, rounded to 4 decimals."),
    Variable("early_shortfall_leader_actuarial_funded_pct", None, "Store the first-cutoff leader's actuarial assets divided by funding target in percent, rounded to 4 decimals."),
    Variable("updated_underfunded_plan_count", None, "Store the eligible underfunded-plan count at the updated cutoff as an integer."),
    Variable("updated_shortfall_leader_plan_name", None, "Store the updated-cutoff leader's plan name exactly as reported."),
    Variable("updated_shortfall_leader_sponsor_ein", None, "Store the updated-cutoff leader's nine-digit sponsor EIN as a string."),
    Variable("updated_shortfall_leader_plan_number", None, "Store the updated-cutoff leader's official three-digit plan number as a string."),
    Variable("updated_shortfall_leader_filing_id", None, "Store the updated-cutoff leader's selected filing ID as a string."),
    Variable("updated_shortfall_leader_participant_count", None, "Store the updated-cutoff leader's Schedule SB participant count as an integer."),
    Variable("updated_shortfall_leader_gap_usd_billions", None, "Store the updated-cutoff leader's actuarial funding-target shortfall in USD billions, rounded to 4 decimals."),
    Variable("updated_shortfall_leader_actuarial_funded_pct", None, "Store the updated-cutoff leader's actuarial assets divided by funding target in percent, rounded to 4 decimals."),
    Variable("early_leader_rank_at_updated_cutoff", None, "Store the first-cutoff leader's one-based shortfall rank at the updated cutoff as an integer."),
    Variable("early_leader_gap_at_updated_cutoff_usd_billions", None, "Store the first-cutoff leader's shortfall at the updated cutoff in USD billions, rounded to 4 decimals."),
    Variable("updated_leader_beginning_net_assets_usd_billions", None, "Store beginning net assets in USD billions, rounded to 4 decimals."),
    Variable("updated_leader_total_income_usd_billions", None, "Store total income in USD billions, rounded to 4 decimals."),
    Variable("updated_leader_total_expenses_usd_billions", None, "Store total expenses in USD billions, rounded to 4 decimals."),
    Variable("updated_leader_net_income_usd_billions", None, "Store net income in USD billions, rounded to 4 decimals."),
    Variable("updated_leader_transfers_to_plan_usd_billions", None, "Store transfers to the plan in USD billions, rounded to 4 decimals."),
    Variable("updated_leader_transfers_from_plan_usd_billions", None, "Store transfers from the plan in USD billions, rounded to 4 decimals."),
    Variable("updated_leader_ending_net_assets_usd_billions", None, "Store ending net assets in USD billions, rounded to 4 decimals."),
    Variable("updated_leader_net_asset_rollforward_residual_usd_millions", None, "Store the net-asset roll-forward residual in USD millions, rounded to 3 decimals."),
    Variable("updated_leader_income_expense_residual_usd_millions", None, "Store total income minus total expenses minus net income in USD millions, rounded to 3 decimals."),
    Variable("updated_leader_sb_current_value_assets_usd_billions", None, "Store Schedule SB current-value assets in USD billions, rounded to 4 decimals."),
    Variable("updated_leader_sb_actuarial_value_assets_usd_billions", None, "Store Schedule SB actuarial-value assets in USD billions, rounded to 4 decimals."),
    Variable("treasury_curve_start_observation_date", None, "Store the selected starting Treasury observation date as an ISO YYYY-MM-DD string."),
    Variable("treasury_curve_start_thirty_year_yield_pct", None, "Store the starting 30-year Treasury par yield in percent, rounded to 2 decimals."),
    Variable("treasury_curve_end_thirty_year_yield_pct", None, "Store the ending 30-year Treasury par yield in percent, rounded to 2 decimals."),
    Variable("treasury_thirty_year_yield_change_basis_points", None, "Store the ending-minus-starting 30-year yield change in basis points, rounded to 1 decimal."),
    Variable("first_order_funding_target_change_pct", None, "Store the first-order hypothetical funding-target change in percent, rounded to 2 decimals."),
    Variable("hypothetical_stressed_funding_target_usd_billions", None, "Store the hypothetical stressed funding target in USD billions, rounded to 4 decimals."),
    Variable("hypothetical_stressed_actuarial_funded_pct", None, "Store the hypothetical stressed actuarial funded ratio in percent, rounded to 4 decimals."),
]


def _joined_filings() -> pd.DataFrame:
    financials = load_expansion_table("dol_form5500_financials")
    schedules = load_expansion_table("dol_form5500_schedule_sb")
    return financials.merge(schedules, on="filing_id", suffixes=("_financial", "_sb"))


def _snapshot(cutoff: str) -> pd.DataFrame:
    rows = _joined_filings()
    rows = rows.loc[
        rows.plan_year_end_financial.eq(PLAN_YEAR_END)
        & rows.filing_status_financial.eq("FILING_RECEIVED")
        & rows.filing_status_sb.eq("FILING_RECEIVED")
        & rows.date_received_financial.le(cutoff)
    ].copy()
    identity = ["sponsor_ein_financial", "plan_number_financial", "plan_year_end_financial"]
    rows = (
        rows.sort_values(["date_received_financial", "filing_id"])
        .drop_duplicates(identity, keep="last")
    )
    rows = rows.loc[
        rows.schedule_sb_participant_count.ge(MINIMUM_PARTICIPANTS)
        & rows.total_funding_target_usd.gt(0)
        & rows.actuarial_value_assets_usd.notna()
    ].copy()
    rows["funding_gap_usd"] = (
        rows.total_funding_target_usd - rows.actuarial_value_assets_usd
    )
    rows = rows.loc[rows.funding_gap_usd.gt(0)].copy()
    rows["actuarial_funded_pct"] = (
        rows.actuarial_value_assets_usd / rows.total_funding_target_usd * 100
    )
    rows = rows.sort_values(
        ["funding_gap_usd", "plan_name_financial", "filing_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    rows["shortfall_rank"] = range(1, len(rows) + 1)
    return rows


def _zero_if_missing(value) -> float:
    return 0.0 if pd.isna(value) else float(value)


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    early = _snapshot(EARLY_CUTOFF)
    updated = _snapshot(UPDATED_CUTOFF)
    first = early.iloc[0]
    current = updated.iloc[0]
    old_at_updated = updated.loc[
        updated.sponsor_ein_financial.eq(first.sponsor_ein_financial)
        & updated.plan_number_financial.eq(first.plan_number_financial)
        & updated.plan_year_end_financial.eq(first.plan_year_end_financial)
    ]
    if len(old_at_updated) != 1:
        raise ValueError(f"expected one prior leader at updated cutoff, found {len(old_at_updated)}")
    old_at_updated = old_at_updated.iloc[0]

    beginning = float(current.net_assets_beginning_usd)
    income = float(current.total_income_usd)
    expenses = float(current.total_expenses_usd)
    net_income = float(current.net_income_usd)
    transfers_to = _zero_if_missing(current.transfers_to_plan_usd)
    transfers_from = _zero_if_missing(current.transfers_from_plan_usd)
    ending = float(current.net_assets_ending_usd)
    rollforward_residual = ending - (
        beginning + net_income + transfers_to - transfers_from
    )
    income_expense_residual = income - expenses - net_income
    current_assets = float(current.current_value_assets_usd)
    actuarial_assets = float(current.actuarial_value_assets_usd)

    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["Date"] = pd.to_datetime(curve.Date)
    valuation_date = pd.Timestamp(current.valuation_date)
    start = curve.loc[curve.Date.lt(valuation_date)].sort_values("Date").iloc[-1]
    end_rows = curve.loc[curve.Date.eq(CURVE_END_DATE)]
    if len(end_rows) != 1:
        raise ValueError(f"expected one ending Treasury curve row, found {len(end_rows)}")
    end = end_rows.iloc[0]
    start_yield = float(start["30 Yr"])
    end_yield = float(end["30 Yr"])
    yield_change_bps = (end_yield - start_yield) * 100
    target_change_pct = -LIABILITY_DURATION * (yield_change_bps / 10_000) * 100
    target = float(current.total_funding_target_usd)
    stressed_target = target * (1 + target_change_pct / 100)
    base_ratio = actuarial_assets / target * 100
    stressed_ratio = actuarial_assets / stressed_target * 100

    return (
        len(early),
        str(first.plan_name_financial),
        str(first.sponsor_ein_financial),
        str(first.plan_number_financial),
        str(first.filing_id),
        int(first.schedule_sb_participant_count),
        float(first.funding_gap_usd) / 1e9,
        float(first.actuarial_funded_pct),
        len(updated),
        str(current.plan_name_financial),
        str(current.sponsor_ein_financial),
        str(current.plan_number_financial),
        str(current.filing_id),
        int(current.schedule_sb_participant_count),
        float(current.funding_gap_usd) / 1e9,
        float(current.actuarial_funded_pct),
        int(old_at_updated.shortfall_rank),
        float(old_at_updated.funding_gap_usd) / 1e9,
        beginning / 1e9,
        income / 1e9,
        expenses / 1e9,
        net_income / 1e9,
        transfers_to / 1e9,
        transfers_from / 1e9,
        ending / 1e9,
        rollforward_residual / 1e6,
        income_expense_residual / 1e6,
        current_assets / 1e9,
        actuarial_assets / 1e9,
        start.Date.strftime("%Y-%m-%d"),
        start_yield,
        end_yield,
        yield_change_bps,
        target_change_pct,
        stressed_target / 1e9,
        stressed_ratio,
    )


DECIMALS = [
    0, None, None, None, None, 0, 4, 4,
    0, None, None, None, None, 0, 4, 4, 0, 4,
    4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4,
    None, 2, 2, 1, 2, 4, 4,
]


def _validate_subset(outputs: dict, names: list[str]):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip((variable.name for variable in variables), ground_truth()))
    places = dict(zip((variable.name for variable in variables), DECIMALS))
    return validate_ordered_outputs(
        outputs,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [places[name] for name in names],
    )


def validate_turn_1(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_turn_2(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_turn_3(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_turn_4(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_early_screen": turn_validator(validate_turn_1),
    "validate_updated_screen": turn_validator(validate_turn_2),
    "validate_filing_reconciliation": turn_validator(validate_turn_3),
    "validate_curve_scenario": turn_validator(validate_turn_4),
}
