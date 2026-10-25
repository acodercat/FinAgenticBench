"""Compare SBA 504 financing weights with state establishment-normalized activity.

The frozen privacy-reduced SBA extract has no guaranteed unique loan identifier,
so the population is explicitly reported source rows rather than deduplicated
borrowers or loans. Across the 50 states and District of Columbia, 2,135 rows
meet the calendar-2025 approval, disbursement and positive-financing rules. Their
2.013060 billion USD of SBA gross approvals and 2.657696711 billion USD of
third-party financing produce a 43.0992% ratio of sums, versus a 44.4259%
unweighted mean row share and a -1.3267-point gap.

South Dakota has 30 eligible rows and 38,496 preliminary 2025 private annual-
average establishments, or 7.7930 rows per 10,000 establishments. Its 47.484228
million USD combined financing has a 43.5008% aggregate SBA share versus a
43.3444% mean row share, a +0.1563-point gap.

Fiscal-year rather than calendar-year approval, approval-only rather than
disbursed rows, total-covered rather than private establishments, and including
Puerto Rico are measured regression alternatives. The query pins all four, so
this is a hard baseline. Numeric validation uses 0.6 x 10^-N rounding-boundary
tolerance for N requested decimals; counts are exact and USPS labels normalize
casing and repeated whitespace.

Fragility: the leading state (SD, 30 loan rows) is 0.36 loans per 10,000
establishments ahead of UT, so a handful of reclassified rows would flip the
leader, and the 2025 QCEW establishment counts are a preliminary vintage.
"""

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, qcew_state_totals
from core.validation import validate_ordered_outputs, turn_validator


STATE_OUTPUTS = {"highest_intensity_state_usps"}

variables = [
    Variable("matched_state_entity_count", None, "Store the matched state-level entity count as an integer."),
    Variable("eligible_reported_row_count", None, "Store the eligible reported SBA row count as an integer."),
    Variable("aggregate_sba_gross_approval_usd_billions", None, "Store aggregate SBA gross approvals in USD billions, rounded to 6 decimals."),
    Variable("aggregate_third_party_dollars_usd_billions", None, "Store aggregate third-party dollars in USD billions, rounded to 6 decimals."),
    Variable("aggregate_total_financing_usd_billions", None, "Store aggregate SBA-plus-third-party financing in USD billions, rounded to 6 decimals."),
    Variable("aggregate_sba_financing_share_pct", None, "Store aggregate SBA gross approvals divided by aggregate financing in percent, rounded to 4 decimals."),
    Variable("unweighted_mean_row_sba_share_pct", None, "Store the unweighted mean of reported-row SBA financing shares in percent, rounded to 4 decimals."),
    Variable("highest_intensity_state_usps", None, "Store the USPS abbreviation of the highest-intensity state-level entity; casing and repeated whitespace are not significant."),
    Variable("highest_intensity_reported_row_count", None, "Store that entity's eligible reported SBA row count as an integer."),
    Variable("highest_intensity_private_establishment_count", None, "Store that entity's private annual-average establishment count as an integer."),
    Variable("highest_rows_per_10000_establishments", None, "Store that entity's reported rows per 10,000 private establishments, rounded to 4 decimals."),
    Variable("highest_intensity_total_financing_usd_millions", None, "Store that entity's aggregate SBA-plus-third-party financing in USD millions, rounded to 6 decimals."),
    Variable("highest_intensity_aggregate_sba_share_pct", None, "Store that entity's aggregate SBA financing share in percent, rounded to 4 decimals."),
    Variable("highest_intensity_unweighted_mean_row_share_pct", None, "Store that entity's unweighted mean reported-row SBA share in percent, rounded to 4 decimals."),
]


def _normalized_label(value):
    if not isinstance(value, str):
        return value
    return " ".join(value.split()).casefold()


def _state_population():
    counties = load_expansion_table("census_counties").copy()
    counties["state_fips"] = counties.GEOID.str[:2]
    population = counties[["USPS", "state_fips"]].drop_duplicates()
    if population.USPS.duplicated().any():
        raise ValueError("Census USPS abbreviation maps to multiple state FIPS codes")
    population = population.loc[pd.to_numeric(population.state_fips).le(56)]
    if len(population) != 51:
        raise ValueError(f"expected 51 states or districts, found {len(population)}")
    return population


def _analysis_population():
    population = _state_population()
    loans = load_expansion_table("sba_504").copy()
    loans["total_financing"] = loans.GrossApproval + loans.ThirdPartyDollars
    loans = loans.loc[
        loans.ApprovalDate.between("2025-01-01", "2025-12-31")
        & loans.FirstDisbursementDate.notna()
        & loans.FirstDisbursementDate.le("2025-12-31")
        & loans.GrossApproval.notna()
        & loans.ThirdPartyDollars.notna()
        & loans.total_financing.gt(0)
    ].merge(
        population, left_on="ProjectState", right_on="USPS", validate="many_to_one"
    )
    loans["row_sba_share"] = loans.GrossApproval / loans.total_financing * 100

    qcew = qcew_state_totals(load_expansion_table("bls_qcew")).loc[
        lambda frame: frame.year.eq(2025)
        & frame.aggregation_level.eq("state_by_ownership")
        & frame.ownership_scope.eq("private")
        & frame.industry_code.eq("10"),
        ["state_fips", "annual_avg_establishments"],
    ]
    state_base = population.merge(qcew, on="state_fips", validate="one_to_one")
    if len(state_base) != 51 or state_base.annual_avg_establishments.le(0).any():
        raise ValueError("invalid 50-state-plus-DC QCEW establishment population")

    grouped = loans.groupby("state_fips", as_index=False).agg(
        reported_rows=("ProjectState", "size"),
        gross_approval=("GrossApproval", "sum"),
        third_party=("ThirdPartyDollars", "sum"),
        total_financing=("total_financing", "sum"),
        mean_row_share=("row_sba_share", "mean"),
    )
    states = state_base.merge(grouped, on="state_fips", how="left", validate="one_to_one")
    for column in ("reported_rows", "gross_approval", "third_party", "total_financing"):
        states[column] = states[column].fillna(0)
    states["rows_per_10000"] = (
        states.reported_rows / states.annual_avg_establishments * 10000
    )
    states["aggregate_share"] = states.gross_approval / states.total_financing * 100
    states["share_gap"] = states.aggregate_share - states.mean_row_share
    return loans, states


def ground_truth():
    loans, states = _analysis_population()
    aggregate_share = loans.GrossApproval.sum() / loans.total_financing.sum() * 100
    mean_share = loans.row_sba_share.mean()
    leader = states.sort_values(
        ["rows_per_10000", "USPS"], ascending=[False, True]
    ).iloc[0]
    return (
        len(states),
        len(loans),
        loans.GrossApproval.sum() / 1e9,
        loans.ThirdPartyDollars.sum() / 1e9,
        loans.total_financing.sum() / 1e9,
        aggregate_share,
        mean_share,
        leader.USPS,
        int(leader.reported_rows),
        int(leader.annual_avg_establishments),
        leader.rows_per_10000,
        leader.total_financing / 1e6,
        leader.aggregate_share,
        leader.mean_row_share,
    )


TURN_1_NAMES = [
    "matched_state_entity_count", "eligible_reported_row_count",
    "aggregate_sba_gross_approval_usd_billions", "aggregate_third_party_dollars_usd_billions",
    "aggregate_total_financing_usd_billions", "aggregate_sba_financing_share_pct",
    "unweighted_mean_row_sba_share_pct",
]
TURN_2_NAMES = [
    "highest_intensity_state_usps", "highest_intensity_reported_row_count",
    "highest_intensity_private_establishment_count", "highest_rows_per_10000_establishments",
    "highest_intensity_total_financing_usd_millions", "highest_intensity_aggregate_sba_share_pct",
    "highest_intensity_unweighted_mean_row_share_pct",
]

DECIMALS = [0, 0, 6, 6, 6, 4, 4, None, 0, 0, 4, 6, 4, 4]


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in STATE_OUTPUTS:
        normalized[name] = _normalized_label(outputs.get(name))
        truth[name] = _normalized_label(truth[name])
    return validate_ordered_outputs(
        normalized,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


# The turn contract: `validators` maps the name a turn's `validator`
# field cites to the callable that judges it.
validators = {
    "validate_national_aggregate": turn_validator(validate_turn_1),
    "validate_intensity_leader": turn_validator(validate_turn_2),
}
