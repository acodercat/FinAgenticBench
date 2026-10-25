"""Trace a disaster-screened state delinquency leader into institution concentration.

This hard baseline carries a state selected jointly from NCUA loan performance,
FEMA county declarations and Census county denominators into a credit-union
contribution analysis, a QCEW macro overlay and an exclusion sensitivity.  The
query pins the declaration window, valid-county population, state thresholds,
weighted delinquency denominator and all rank directions; alternatives remain
regression probes rather than live traps.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, qcew_state_totals
from core.validation import turn_validator, validate_ordered_outputs


STATE_USPS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI",
    "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
    "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
    "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}

TURN_1_NAMES = [
    "eligible_disaster_screened_state_count", "delinquency_leader_state",
    "leader_state_credit_union_count", "leader_state_total_loans_usd_billions",
    "leader_state_delinquent_loans_usd_billions", "leader_state_delinquency_rate_pct",
    "leader_state_census_county_count", "leader_state_disaster_county_count",
    "leader_state_disaster_county_coverage_pct", "state_delinquency_leader_margin_pp",
]
TURN_2_NAMES = [
    "dominant_credit_union_number", "dominant_credit_union_name",
    "dominant_credit_union_loans_usd_billions",
    "dominant_credit_union_delinquent_usd_billions",
    "dominant_credit_union_delinquency_rate_pct",
    "dominant_credit_union_state_delinquent_share_pct",
]
TURN_3_NAMES = [
    "leader_state_private_employment_2024", "leader_state_private_employment_2025",
    "leader_state_private_employment_growth_pct",
    "leader_state_employment_growth_rank_lowest_first",
    "leader_state_average_annual_pay_2024_usd",
    "leader_state_average_annual_pay_2025_usd",
    "leader_state_average_annual_pay_growth_pct",
    "leader_state_pay_growth_rank_lowest_first",
]
TURN_4_NAMES = [
    "adjusted_leader_state_total_loans_usd_billions",
    "adjusted_leader_state_delinquent_loans_usd_billions",
    "adjusted_leader_state_delinquency_rate_pct",
    "adjusted_leader_state_rank", "adjusted_screen_leader_state",
    "adjusted_screen_leader_delinquency_rate_pct",
]


def _v(name: str, description: str) -> Variable:
    return Variable(name, None, description)


variables = [
    _v("eligible_disaster_screened_state_count", "Store the eligible state-level jurisdiction count as an integer."),
    _v("delinquency_leader_state", "Store the selected state as its two-letter USPS abbreviation."),
    _v("leader_state_credit_union_count", "Store the selected state's credit-union count as an integer."),
    _v("leader_state_total_loans_usd_billions", "Store selected-state total loans and leases in USD billions, rounded to 6 decimals."),
    _v("leader_state_delinquent_loans_usd_billions", "Store selected-state loans delinquent two or more months in USD billions, rounded to 6 decimals. Compute from the unrounded source values, not from the other rounded figures reported here."),
    _v("leader_state_delinquency_rate_pct", "Store selected-state delinquent loans divided by total loans in percent, rounded to 4 decimals. Compute from the unrounded source values, not from the other rounded figures reported here."),
    _v("leader_state_census_county_count", "Store the selected state's Census county-equivalent count as an integer."),
    _v("leader_state_disaster_county_count", "Store the selected state's distinct qualifying disaster-county count as an integer."),
    _v("leader_state_disaster_county_coverage_pct", "Store qualifying disaster counties divided by Census county equivalents in percent, rounded to 4 decimals."),
    _v("state_delinquency_leader_margin_pp",
       "Store the selected state's delinquency-rate margin over the runner-up in percentage points, "
       "rounded to 4 decimals. Compute from the unrounded values, not from the rounded component "
       "figures."),
    _v("dominant_credit_union_number", "Store the selected credit union's NCUA number as a string."),
    _v("dominant_credit_union_name", "Store the selected credit union's name exactly as reported."),
    _v("dominant_credit_union_loans_usd_billions", "Store selected-credit-union total loans and leases in USD billions, rounded to 6 decimals."),
    _v("dominant_credit_union_delinquent_usd_billions", "Store selected-credit-union delinquent loans in USD billions, rounded to 6 decimals. Compute from the unrounded source values, not from the other rounded figures reported here."),
    _v("dominant_credit_union_delinquency_rate_pct", "Store selected-credit-union delinquent loans divided by its total loans in percent, rounded to 4 decimals."),
    _v("dominant_credit_union_state_delinquent_share_pct", "Store the selected credit union's share of selected-state delinquent dollars in percent, rounded to 4 decimals."),
    _v("leader_state_private_employment_2024", "Store selected-state 2024 annual-average private employment as an integer."),
    _v("leader_state_private_employment_2025", "Store selected-state 2025 annual-average private employment as an integer."),
    _v("leader_state_private_employment_growth_pct", "Store calculated 2024-to-2025 private-employment growth in percent, rounded to 4 decimals."),
    _v("leader_state_employment_growth_rank_lowest_first", "Store the selected state's employment-growth rank within the eligible cohort, with 1 the lowest, as an integer."),
    _v("leader_state_average_annual_pay_2024_usd", "Store selected-state 2024 private average annual pay in USD as an integer."),
    _v("leader_state_average_annual_pay_2025_usd", "Store selected-state 2025 private average annual pay in USD as an integer."),
    _v("leader_state_average_annual_pay_growth_pct", "Store calculated 2024-to-2025 average-annual-pay growth in percent, rounded to 4 decimals."),
    _v("leader_state_pay_growth_rank_lowest_first", "Store the selected state's pay-growth rank within the eligible cohort, with 1 the lowest, as an integer."),
    _v("adjusted_leader_state_total_loans_usd_billions", "Store selected-state loans after the stated exclusion in USD billions, rounded to 6 decimals."),
    _v("adjusted_leader_state_delinquent_loans_usd_billions", "Store selected-state delinquent loans after the stated exclusion in USD billions, rounded to 6 decimals. Compute from the unrounded source values, not from the other rounded figures reported here."),
    _v("adjusted_leader_state_delinquency_rate_pct", "Store the selected state's adjusted delinquency rate in percent, rounded to 4 decimals."),
    _v("adjusted_leader_state_rank", "Store the selected state's adjusted rank in the fixed eligible cohort as an integer."),
    _v("adjusted_screen_leader_state", "Store the adjusted screen leader as a two-letter USPS abbreviation."),
    _v("adjusted_screen_leader_delinquency_rate_pct", "Store the adjusted screen leader's delinquency rate in percent, rounded to 4 decimals. Compute from the unrounded source values, not from the other rounded figures reported here."),
]


def _state_screen() -> tuple[pd.DataFrame, pd.DataFrame]:
    counties = load_expansion_table("census_counties")
    county_population = counties.loc[counties.USPS.isin(STATE_USPS)].copy()
    county_counts = county_population.groupby("USPS").GEOID.nunique()
    valid_counties = set(county_population.GEOID)

    disasters = load_expansion_table("fema_disasters")
    disasters = disasters.loc[
        disasters.declaration_type.eq("DR")
        & disasters.incident_begin_date.between("2024-10-01", "2025-09-30")
        & disasters.declaration_date.le("2025-09-30")
        & disasters.fips_county_code.ne("000")
        & disasters.county_fips.isin(valid_counties)
    ]
    disaster_counts = disasters.groupby("state_usps").county_fips.nunique()

    ncua = load_expansion_table("ncua_call_reports")
    ncua = ncua.loc[
        ncua.report_date.eq("2025-09-30")
        & ncua.state.isin(STATE_USPS)
        & ncua.total_loans_and_leases_usd.gt(0)
    ].copy()
    states = ncua.groupby("state").agg(
        credit_union_count=("credit_union_number", "nunique"),
        total_loans=("total_loans_and_leases_usd", "sum"),
        delinquent=("delinquent_loans_two_plus_months_usd", "sum"),
    )
    states = states.join(county_counts.rename("county_count")).join(
        disaster_counts.rename("disaster_count")
    ).fillna({"disaster_count": 0})
    states["coverage"] = states.disaster_count / states.county_count * 100
    states["delinquency_rate"] = states.delinquent / states.total_loans * 100
    eligible = states.loc[
        states.credit_union_count.ge(25) & states.coverage.ge(25)
    ].sort_values(["delinquency_rate", "state"], ascending=[False, True])
    return eligible, ncua


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    eligible, ncua = _state_screen()
    leader = eligible.iloc[0]
    state = str(eligible.index[0])

    state_cus = ncua.loc[ncua.state.eq(state)].copy()
    dominant = state_cus.sort_values(
        ["delinquent_loans_two_plus_months_usd", "credit_union_number"],
        ascending=[False, True],
    ).iloc[0]
    dominant_rate = (
        float(dominant.delinquent_loans_two_plus_months_usd)
        / float(dominant.total_loans_and_leases_usd) * 100
    )
    dominant_share = (
        float(dominant.delinquent_loans_two_plus_months_usd)
        / float(leader.delinquent) * 100
    )

    counties = load_expansion_table("census_counties")
    bridge = counties.assign(state_fips=counties.GEOID.str[:2])[
        ["USPS", "state_fips"]
    ].drop_duplicates()
    qcew = qcew_state_totals(load_expansion_table("bls_qcew"))
    qcew = qcew.loc[qcew.ownership_scope.eq("private")].pivot(
        index="state_fips", columns="year",
        values=["annual_avg_employment", "average_annual_pay_usd"],
    ).dropna()
    macro_rows = []
    for candidate in eligible.index:
        fips = bridge.loc[bridge.USPS.eq(candidate), "state_fips"].iloc[0]
        row = qcew.loc[fips]
        employment_growth = (
            float(row[("annual_avg_employment", 2025)])
            / float(row[("annual_avg_employment", 2024)]) - 1
        ) * 100
        pay_growth = (
            float(row[("average_annual_pay_usd", 2025)])
            / float(row[("average_annual_pay_usd", 2024)]) - 1
        ) * 100
        macro_rows.append((candidate, fips, employment_growth, pay_growth))
    macro = pd.DataFrame(
        macro_rows, columns=["state", "state_fips", "employment_growth", "pay_growth"]
    )
    employment_rank = macro.sort_values(
        ["employment_growth", "state"]
    ).reset_index(drop=True)
    pay_rank = macro.sort_values(["pay_growth", "state"]).reset_index(drop=True)
    selected_macro = macro.loc[macro.state.eq(state)].iloc[0]
    selected_fips = selected_macro.state_fips
    selected_levels = qcew.loc[selected_fips]

    adjusted = eligible.copy()
    adjusted.loc[state, "total_loans"] -= int(dominant.total_loans_and_leases_usd)
    adjusted.loc[state, "delinquent"] -= int(
        dominant.delinquent_loans_two_plus_months_usd
    )
    adjusted["delinquency_rate"] = adjusted.delinquent / adjusted.total_loans * 100
    adjusted = adjusted.sort_values(
        ["delinquency_rate", "state"], ascending=[False, True]
    )
    adjusted_leader_state = str(adjusted.index[0])

    return (
        len(eligible),
        state,
        int(leader.credit_union_count),
        float(leader.total_loans) / 1e9,
        float(leader.delinquent) / 1e9,
        float(leader.delinquency_rate),
        int(leader.county_count),
        int(leader.disaster_count),
        float(leader.coverage),
        float(leader.delinquency_rate - eligible.iloc[1].delinquency_rate),
        str(dominant.credit_union_number),
        str(dominant.credit_union_name),
        float(dominant.total_loans_and_leases_usd) / 1e9,
        float(dominant.delinquent_loans_two_plus_months_usd) / 1e9,
        dominant_rate,
        dominant_share,
        int(selected_levels[("annual_avg_employment", 2024)]),
        int(selected_levels[("annual_avg_employment", 2025)]),
        float(selected_macro.employment_growth),
        int(employment_rank.index[employment_rank.state.eq(state)][0] + 1),
        int(selected_levels[("average_annual_pay_usd", 2024)]),
        int(selected_levels[("average_annual_pay_usd", 2025)]),
        float(selected_macro.pay_growth),
        int(pay_rank.index[pay_rank.state.eq(state)][0] + 1),
        float(adjusted.loc[state, "total_loans"]) / 1e9,
        float(adjusted.loc[state, "delinquent"]) / 1e9,
        float(adjusted.loc[state, "delinquency_rate"]),
        int(adjusted.index.get_loc(state) + 1),
        adjusted_leader_state,
        float(adjusted.loc[adjusted_leader_state, "delinquency_rate"]),
    )


DECIMALS = [
    0, None, 0, 6, 6, 4, 0, 0, 4, 4,
    None, None, 6, 6, 4, 4,
    0, 0, 4, 0, 0, 0, 4, 0,
    6, 6, 4, 0, None, 4,
]


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[name] for name in names],
        [truth[name] for name in names], [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_state_screen": turn_validator(validate_turn_1),
    "validate_credit_union_concentration": turn_validator(validate_turn_2),
    "validate_state_macro_overlay": turn_validator(validate_turn_3),
    "validate_exclusion_sensitivity": turn_validator(validate_turn_4),
}
