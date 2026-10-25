"""Screen equal-horizon SBA charge-offs with a later state labor measure.

Across the 50 states and DC, 24 states have at least 500 reported FY2022 7(a)
loan rows after the disbursement and cutoff rules.  New York has the highest
36-month charge-off incidence overall at 3.4247%, but Maryland leads the six
states with negative 2024-to-2025 private annual-average employment growth at
2.9008%.  Maryland's charge-off amount is 0.4545% of cohort gross approvals.

Dropping exact-looking public rows changes New York's count and incidence; using
a 250-row floor instead of 500 selects Louisiana overall; using total-covered
rather than private employment, the source's rounded change field rather than
the two employment levels, or the later snapshot status rather than dated
36-month events changes the population or outputs.  The query pins these
conventions, so the alternatives are regression probes for a hard baseline.
Numeric validation uses 0.6 x 10^-N rounding-boundary tolerance; counts and state
labels are exact.
"""

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, qcew_state_totals
from core.validation import validate_ordered_outputs, turn_validator


CUTOFF = pd.Timestamp("2025-12-31")

variables = [
    Variable("matched_state_and_dc_count", None, "Store the matched 50-state-plus-DC count as an integer."),
    Variable("minimum_500_loan_state_count", None, "Store the count meeting the minimum reported-loan-row threshold as an integer."),
    Variable("negative_private_employment_growth_state_count", None, "Store the count within that population having negative private annual-average employment growth as an integer."),
    Variable("overall_highest_chargeoff_incidence_state", None, "Store the overall leading state's two-letter abbreviation."),
    Variable("overall_highest_chargeoff_incidence_pct", None, "Store the overall leading 36-month charge-off incidence in percent, rounded to 4 decimals."),
    Variable("overall_leader_private_employment_growth_pct", None, "Store the overall leader's private annual-average employment growth in percent, rounded to 4 decimals."),
    Variable("negative_growth_highest_chargeoff_state", None, "Store the negative-employment-growth subset's leading state abbreviation."),
    Variable("screen_leader_reported_loan_count", None, "Store the screened leader's reported loan-row count as an integer."),
    Variable("screen_leader_36_month_chargeoff_count", None, "Store the screened leader's qualifying 36-month charge-off count as an integer."),
    Variable("screen_leader_chargeoff_incidence_pct", None, "Store the screened leader's 36-month charge-off incidence in percent, rounded to 4 decimals."),
    Variable("screen_leader_chargeoff_amount_to_approval_pct", None, "Store its qualifying gross charge-off amount divided by cohort gross approvals in percent, rounded to 4 decimals."),
    Variable("screen_leader_2024_private_annual_avg_employment", None, "Store its 2024 private annual-average employment as an integer."),
    Variable("screen_leader_2025_private_annual_avg_employment", None, "Store its 2025 private annual-average employment as an integer."),
    Variable("screen_leader_private_employment_growth_pct", None, "Store its private annual-average employment growth in percent, rounded to 4 decimals."),
]


def _state_metrics():
    loans = load_expansion_table("sba_7a").copy()
    for column in ["ApprovalDate", "FirstDisbursementDate", "ChargeOffDate"]:
        loans[column] = pd.to_datetime(loans[column], errors="coerce")
    cohort = loans.loc[
        loans.ApprovalFY.eq(2022)
        & loans.FirstDisbursementDate.notna()
        & loans.FirstDisbursementDate.le(CUTOFF)
    ].copy()
    cohort["qualifying_chargeoff"] = (
        cohort.ChargeOffDate.notna()
        & cohort.ChargeOffDate.le(cohort.ApprovalDate + pd.DateOffset(months=36))
        & cohort.ChargeOffDate.le(CUTOFF)
    )
    cohort["qualifying_chargeoff_amount"] = cohort.GrossChargeOffAmount.where(
        cohort.qualifying_chargeoff, 0
    )
    states = cohort.groupby("ProjectState", as_index=False).agg(
        reported_loan_count=("LocationID", "size"),
        chargeoff_count=("qualifying_chargeoff", "sum"),
        gross_approval=("GrossApproval", "sum"),
        chargeoff_amount=("qualifying_chargeoff_amount", "sum"),
    )
    states["chargeoff_incidence_pct"] = (
        states.chargeoff_count / states.reported_loan_count * 100
    )
    states["chargeoff_amount_to_approval_pct"] = (
        states.chargeoff_amount / states.gross_approval * 100
    )

    counties = load_expansion_table("census_counties").copy()
    counties["state_fips"] = counties.GEOID.str[:2]
    state_map = counties[["USPS", "state_fips"]].drop_duplicates()
    state_map = state_map.loc[~state_map.state_fips.eq("72")]
    if len(state_map) != 51 or state_map.USPS.duplicated().any():
        raise ValueError("unexpected Census 50-state-plus-DC bridge")

    qcew = qcew_state_totals(load_expansion_table("bls_qcew"))
    private = qcew.loc[qcew.ownership_scope.eq("private")].pivot(
        index="state_fips", columns="year", values="annual_avg_employment"
    ).dropna(subset=[2024, 2025])
    private = private.rename(columns={2024: "employment_2024", 2025: "employment_2025"})
    private["employment_growth_pct"] = (
        private.employment_2025 / private.employment_2024 - 1
    ) * 100
    matched = states.merge(
        state_map, left_on="ProjectState", right_on="USPS", validate="one_to_one"
    ).merge(private, left_on="state_fips", right_index=True, validate="one_to_one")
    if len(matched) != 51:
        raise ValueError("unexpected SBA-Census-QCEW state population")
    return matched


def ground_truth():
    matched = _state_metrics()
    eligible = matched.loc[matched.reported_loan_count.ge(500)].copy()
    negative = eligible.loc[eligible.employment_growth_pct.lt(0)].copy()
    overall = eligible.sort_values(
        ["chargeoff_incidence_pct", "ProjectState"], ascending=[False, True]
    ).iloc[0]
    screened = negative.sort_values(
        ["chargeoff_incidence_pct", "ProjectState"], ascending=[False, True]
    ).iloc[0]
    if len(eligible) != 24 or len(negative) != 6:
        raise ValueError("unexpected SBA labor-screen populations")
    return (
        len(matched),
        len(eligible),
        len(negative),
        overall.ProjectState,
        overall.chargeoff_incidence_pct,
        overall.employment_growth_pct,
        screened.ProjectState,
        int(screened.reported_loan_count),
        int(screened.chargeoff_count),
        screened.chargeoff_incidence_pct,
        screened.chargeoff_amount_to_approval_pct,
        int(screened.employment_2024),
        int(screened.employment_2025),
        screened.employment_growth_pct,
    )


def validate(outputs):
    return validate_ordered_outputs(
        outputs,
        variables,
        ground_truth(),
        [0, 0, 0, None, 4, 4, None, 0, 0, 4, 4, 0, 0, 4],
    )



TURN_1_NAMES = [
    "matched_state_and_dc_count",
    "minimum_500_loan_state_count",
    "negative_private_employment_growth_state_count",
]
TURN_2_NAMES = [
    "overall_highest_chargeoff_incidence_state",
    "overall_highest_chargeoff_incidence_pct",
    "overall_leader_private_employment_growth_pct",
]
TURN_3_NAMES = [
    "negative_growth_highest_chargeoff_state",
    "screen_leader_reported_loan_count",
    "screen_leader_36_month_chargeoff_count",
    "screen_leader_chargeoff_incidence_pct",
    "screen_leader_chargeoff_amount_to_approval_pct",
    "screen_leader_2024_private_annual_avg_employment",
    "screen_leader_2025_private_annual_avg_employment",
    "screen_leader_private_employment_growth_pct",
]


def _turn_subset(names):
    """Score only `names`, filling the other outputs with their own answers."""
    def run(outputs):
        merged = dict(zip((variable.name for variable in variables), ground_truth()))
        merged.update({name: outputs.get(name) for name in names})
        return validate(merged)

    return run


# The turn contract: `validators` maps the name a turn's `validator`
# field cites to the callable that judges it.
validators = {
    "validate_state_population": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_overall_leader": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_screened_leader": turn_validator(_turn_subset(TURN_3_NAMES)),
}
