"""Trace a GDP revision-selected quarter into rates and futures positioning.

The canonical path selects 2025-Q2 from six complete 2024-Q1 through 2025-Q2
real-GDP release histories because its first-to-current absolute revision is
0.8698 percentage point, 0.1204 point above 2024-Q2.  It then carries that
quarter into quarter-end Treasury curve observations and the latest prior CFTC
futures-only reports for 2-year and 10-year Treasury-note contracts.

This is a hard baseline.  The query pins the complete-vintage population,
signed changes, last reported curve observation, CFTC scope and report-date
alignment.  Using the third release as current, selecting the largest signed
rather than absolute revision, using calendar-quarter dates absent from a
source, or mixing CFTC scopes produces measured alternatives that the
query excludes explicitly rather than leaving them as a live trap.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


GDP_SERIES = "real_gdp_annualized_pct"
START_PERIOD = "2024:Q1"
END_PERIOD = "2025:Q2"
CONTRACTS = {"2Y": "042601", "10Y": "043602"}

TURN_1_NAMES = [
    "complete_gdp_vintage_quarter_count", "largest_revision_quarter",
    "leader_first_release_real_gdp_pct", "leader_current_real_gdp_pct",
    "absolute_revision_leader_margin_pp",
]
TURN_2_NAMES = [
    "leader_second_release_real_gdp_pct", "leader_third_release_real_gdp_pct",
    "largest_adjacent_revision_stage",
]
TURN_3_NAMES = [
    "curve_start_observation_date", "curve_end_observation_date",
    "curve_start_two_year_yield_pct", "curve_start_ten_year_yield_pct",
    "curve_end_two_year_yield_pct", "curve_end_ten_year_yield_pct",
    "curve_start_two_ten_spread_bp", "curve_end_two_ten_spread_bp",
]
TURN_4_NAMES = [
    "cftc_start_report_date", "cftc_end_report_date",
    "two_year_start_lev_money_net_contracts", "two_year_end_lev_money_net_contracts",
    "two_year_net_contract_change", "two_year_start_net_open_interest_pct",
    "two_year_end_net_open_interest_pct",
    "ten_year_start_lev_money_net_contracts", "ten_year_end_lev_money_net_contracts",
    "ten_year_net_contract_change", "ten_year_start_net_open_interest_pct",
    "ten_year_end_net_open_interest_pct",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("complete_gdp_vintage_quarter_count", "Store the complete-vintage quarter count as an integer."),
    _v("largest_revision_quarter", "Store the selected quarter as a YYYY-Qn string."),
    _v("leader_first_release_real_gdp_pct", "Store the first-release annualized real-GDP growth rate in percent, rounded to 4 decimals."),
    _v("leader_current_real_gdp_pct", "Store the current annualized real-GDP growth rate in percent, rounded to 4 decimals."),
    _v("absolute_revision_leader_margin_pp", "Store the leader's absolute-revision margin over the runner-up in percentage points, rounded to 4 decimals."),
    _v("leader_second_release_real_gdp_pct", "Store the second-release annualized real-GDP growth rate in percent, rounded to 4 decimals."),
    _v("leader_third_release_real_gdp_pct", "Store the third-release annualized real-GDP growth rate in percent, rounded to 4 decimals."),
    _v(
        "largest_adjacent_revision_stage",
        "Store exactly one release-stage label from: first-to-second | "
        "second-to-third | third-to-current.",
    ),
    _v("curve_start_observation_date", "Store the starting Treasury observation date as an ISO YYYY-MM-DD string."),
    _v("curve_end_observation_date", "Store the ending Treasury observation date as an ISO YYYY-MM-DD string."),
    _v("curve_start_two_year_yield_pct", "Store the starting 2-year par yield in percent, rounded to 2 decimals."),
    _v("curve_start_ten_year_yield_pct", "Store the starting 10-year par yield in percent, rounded to 2 decimals."),
    _v("curve_end_two_year_yield_pct", "Store the ending 2-year par yield in percent, rounded to 2 decimals."),
    _v("curve_end_ten_year_yield_pct", "Store the ending 10-year par yield in percent, rounded to 2 decimals."),
    _v("curve_start_two_ten_spread_bp", "Store the starting 10-year-minus-2-year spread in basis points, rounded to 1 decimal."),
    _v("curve_end_two_ten_spread_bp", "Store the ending 10-year-minus-2-year spread in basis points, rounded to 1 decimal."),
    _v("cftc_start_report_date", "Store the starting CFTC report date as an ISO YYYY-MM-DD string."),
    _v("cftc_end_report_date", "Store the ending CFTC report date as an ISO YYYY-MM-DD string."),
    _v("two_year_start_lev_money_net_contracts", "Store starting 2-year leveraged-money long minus short contracts as an integer."),
    _v("two_year_end_lev_money_net_contracts", "Store ending 2-year leveraged-money long minus short contracts as an integer."),
    _v("two_year_net_contract_change", "Store ending minus starting 2-year net contracts as an integer."),
    _v("two_year_start_net_open_interest_pct", "Store starting 2-year leveraged-money net divided by open interest in percent, rounded to 4 decimals."),
    _v("two_year_end_net_open_interest_pct", "Store ending 2-year leveraged-money net divided by open interest in percent, rounded to 4 decimals."),
    _v("ten_year_start_lev_money_net_contracts", "Store starting 10-year leveraged-money long minus short contracts as an integer."),
    _v("ten_year_end_lev_money_net_contracts", "Store ending 10-year leveraged-money long minus short contracts as an integer."),
    _v("ten_year_net_contract_change", "Store ending minus starting 10-year net contracts as an integer."),
    _v("ten_year_start_net_open_interest_pct", "Store starting 10-year leveraged-money net divided by open interest in percent, rounded to 4 decimals."),
    _v("ten_year_end_net_open_interest_pct", "Store ending 10-year leveraged-money net divided by open interest in percent, rounded to 4 decimals."),
]


def _one(frame: pd.DataFrame, label: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"expected one {label} row, found {len(frame)}")
    return frame.iloc[0]


def _quarter_end(period: str) -> pd.Timestamp:
    return pd.Period(period.replace(":", "-"), freq="Q").end_time.normalize()


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    revisions = load_expansion_table("philadelphia_fed_rtdsm")
    cohort = revisions.loc[
        revisions.series.eq(GDP_SERIES)
        & revisions.period.between(START_PERIOD, END_PERIOD)
    ].dropna(subset=["first_release", "second_release", "third_release", "most_recent"]).copy()
    cohort["signed_revision"] = cohort.most_recent - cohort.first_release
    cohort["absolute_revision"] = cohort.signed_revision.abs()
    cohort = cohort.sort_values(["absolute_revision", "period"], ascending=[False, True]).reset_index(drop=True)
    leader = cohort.iloc[0]
    transitions = {
        "first-to-second": float(leader.second_release - leader.first_release),
        "second-to-third": float(leader.third_release - leader.second_release),
        "third-to-current": float(leader.most_recent - leader.third_release),
    }
    largest_stage = sorted(transitions, key=lambda key: (-abs(transitions[key]), list(transitions).index(key)))[0]

    end_boundary = _quarter_end(str(leader.period))
    start_boundary = end_boundary - pd.offsets.QuarterEnd()
    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["Date"] = pd.to_datetime(curve.Date)
    curve_start = curve.loc[curve.Date.le(start_boundary)].sort_values("Date").iloc[-1]
    curve_end = curve.loc[curve.Date.le(end_boundary)].sort_values("Date").iloc[-1]
    start_spread = (float(curve_start["10 Yr"]) - float(curve_start["2 Yr"])) * 100
    end_spread = (float(curve_end["10 Yr"]) - float(curve_end["2 Yr"])) * 100

    cot = load_expansion_table("cftc_cot")
    cot = cot.loc[cot.report_scope.eq("futures_only")].copy()
    cot["report_date"] = pd.to_datetime(cot["Report_Date_as_YYYY-MM-DD"])
    contract_results = {}
    report_dates = []
    for label, code in CONTRACTS.items():
        rows = cot.loc[cot.CFTC_Contract_Market_Code.eq(code)]
        start = rows.loc[rows.report_date.le(start_boundary)].sort_values("report_date").iloc[-1]
        end = rows.loc[rows.report_date.le(end_boundary)].sort_values("report_date").iloc[-1]
        start_net = int(start.Lev_Money_Positions_Long_All - start.Lev_Money_Positions_Short_All)
        end_net = int(end.Lev_Money_Positions_Long_All - end.Lev_Money_Positions_Short_All)
        start_pct = start_net / int(start.Open_Interest_All) * 100
        end_pct = end_net / int(end.Open_Interest_All) * 100
        contract_results[label] = (start_net, end_net, end_net - start_net, start_pct, end_pct)
        report_dates.append((start.report_date, end.report_date))
    if len(set(report_dates)) != 1:
        raise ValueError("2-year and 10-year report boundaries do not align")
    cftc_start, cftc_end = report_dates[0]

    return (
        len(cohort), str(leader.period).replace(":", "-"), float(leader.first_release),
        float(leader.most_recent),
        float(leader.absolute_revision - cohort.iloc[1].absolute_revision),
        float(leader.second_release), float(leader.third_release), largest_stage,
        curve_start.Date.strftime("%Y-%m-%d"), curve_end.Date.strftime("%Y-%m-%d"),
        float(curve_start["2 Yr"]), float(curve_start["10 Yr"]),
        float(curve_end["2 Yr"]), float(curve_end["10 Yr"]),
        start_spread, end_spread,
        cftc_start.strftime("%Y-%m-%d"), cftc_end.strftime("%Y-%m-%d"),
        *contract_results["2Y"], *contract_results["10Y"],
    )


DECIMALS = [
    0, None, 4, 4, 4, 4, 4, None,
    None, None, 2, 2, 2, 2, 1, 1,
    None, None, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4,
]


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip((variable.name for variable in variables), ground_truth()))
    places = dict(zip((variable.name for variable in variables), DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [places[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_revision_screen": turn_validator(validate_turn_1),
    "validate_release_path": turn_validator(validate_turn_2),
    "validate_curve_path": turn_validator(validate_turn_3),
    "validate_positioning_path": turn_validator(validate_turn_4),
}
