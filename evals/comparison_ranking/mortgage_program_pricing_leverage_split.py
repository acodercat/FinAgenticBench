"""Compare reported HMDA pricing and leverage by loan-type code.

The 1.5-percentage-point spread screen is an analyst-defined common cutoff for
this first-lien population, not a legal HPML classification.  Regulation Z uses
1.5 points for non-jumbo first liens and 2.5 points for jumbo first liens, while
subordinate liens use 3.5 points.  The runtime lacks rate-lock dates and the
applicable conforming-limit test needed to apply those legal thresholds row by
row.

Loan type 1 is conventional, 2 FHA-insured, 3 VA-guaranteed, and 4 RHS- or
FSA-guaranteed.  Cross-sectional differences in reported APR-minus-APOR spread
and loan-to-value are descriptive.  They do not isolate the effect of insurance
or a guarantee, establish borrower quality, or prove a program eligibility or
down-payment rule.  LTV medians use nonmissing reported LTV and are accompanied
by their valid-row counts because missingness differs materially by type.

The rate-spread screen separately reports missing and outside-band rows.  A
first lien is selected for priority comparability; a junior lien has a different
claim priority, not different physical collateral by definition.

The FHFA traditional purchase-only HPI is a weighted repeat-sales index based on
Enterprise mortgage data.  It estimates same-property price change and is not a
median of HMDA loan amounts.  The source populations, financing coverage and
measures differ, so state HPI growth cannot be interpreted as growth in the
typical originated loan.

The pinned-reading comparison measures the lien scope, the spread band, the higher-priced
threshold, the index series and the index window.  Admitting subordinate liens
raises the origination count from 4,766,398 to 6,191,825; keeping any reported
spread instead of the plausibility band moves the banded count from 4,140,735 to
4,146,528; applying the 3.5-point subordinate threshold to this first-lien
population drops the conventional above-cutoff share from 7.2303% to 1.7744% and
the FHA share from 6.4608% to 0.0189%, which is why the legal thresholds are not
interchangeable across lien positions; reading the seasonally adjusted HPI reports
2.3888% growth rather than 2.4265%; and measuring within the activity year instead
of fourth quarter over fourth quarter reports 1.0810%.  The query pins all five
choices.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and state codes match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


ACTIVITY_YEAR = "2024"
ORIGINATED_ACTION = "1"
FIRST_LIEN = "1"
SPREAD_BAND = (-5.0, 20.0)
HIGHER_PRICED_THRESHOLD = 1.5
LOAN_TYPES = ("1", "2", "3", "4")
INDEX_FLAVOR = "purchase-only"
INDEX_TYPE = "traditional"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "record_count", "first_lien_origination_count", "banded_record_count",
    "excluded_record_count", "missing_rate_spread_count", "outside_band_rate_spread_count",
    "conventional_count", "fha_count", "va_count", "rhs_fsa_count",
]
TURN_2_NAMES = [
    "conventional_above_cutoff_share_pct", "fha_above_cutoff_share_pct",
    "va_above_cutoff_share_pct", "rhs_fsa_above_cutoff_share_pct",
    "conventional_median_spread_pp", "va_median_spread_pp", "negative_median_spread_loan_type_code",
]
TURN_3_NAMES = [
    "conventional_valid_ltv_count", "fha_valid_ltv_count", "va_valid_ltv_count", "rhs_fsa_valid_ltv_count",
    "conventional_median_ltv_pct", "fha_median_ltv_pct", "va_median_ltv_pct",
    "rhs_fsa_median_ltv_pct", "highest_median_ltv_loan_type_code", "conventional_median_loan_amount",
]
TURN_4_NAMES = [
    "leading_state_code", "leading_state_origination_count", "leading_state_median_loan_amount",
    "leading_state_index_prior_year", "leading_state_index_current_year", "leading_state_index_growth_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the number of loan records in the table as an integer."),
    _v(TURN_1_NAMES[1], "Store the number of first-lien originations as an integer."),
    _v(TURN_1_NAMES[2], "Store how many of those carry a rate spread inside the stated band as an integer."),
    _v(TURN_1_NAMES[3], "Store how many are missing a rate spread or fall outside the band as an integer."),
    _v(TURN_1_NAMES[4], "Store how many first-lien originations have missing rate spread as an integer."),
    _v(TURN_1_NAMES[5], "Store how many nonmissing spreads fall outside the inclusive band as an integer."),
    _v(TURN_1_NAMES[6], "Store the banded count for conventional loan type 1 as an integer."),
    _v(TURN_1_NAMES[7], "Store the banded count for FHA loan type 2 as an integer."),
    _v(TURN_1_NAMES[8], "Store the banded count for VA loan type 3 as an integer."),
    _v(TURN_1_NAMES[9], "Store the banded count for RHS/FSA loan type 4 as an integer."),
    _v(TURN_2_NAMES[0], "Store the share at or above the analytic spread cutoff for loan type 1 in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store the share at or above the analytic spread cutoff for loan type 2 in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the share at or above the analytic spread cutoff for loan type 3 in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store the share at or above the analytic spread cutoff for loan type 4 in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store the median rate spread for loan type 1 in percentage points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[5], "Store the median rate spread for loan type 3 in percentage points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[6], "Store the loan-type code whose median rate spread is below zero as text."),
    _v(TURN_3_NAMES[0], "Store the nonmissing LTV row count for conventional loan type 1 as an integer."),
    _v(TURN_3_NAMES[1], "Store the nonmissing LTV row count for FHA loan type 2 as an integer."),
    _v(TURN_3_NAMES[2], "Store the nonmissing LTV row count for VA loan type 3 as an integer."),
    _v(TURN_3_NAMES[3], "Store the nonmissing LTV row count for RHS/FSA loan type 4 as an integer."),
    _v(TURN_3_NAMES[4], "Store the median nonmissing LTV for conventional loan type 1 in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[5], "Store the median nonmissing LTV for FHA loan type 2 in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[6], "Store the median nonmissing LTV for VA loan type 3 in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[7], "Store the median nonmissing LTV for RHS/FSA loan type 4 in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[8], "Store the lowest loan-type code holding the highest median LTV as text."),
    _v(TURN_3_NAMES[9], "Store the median loan amount for conventional loan type 1 as an integer."),
    _v(TURN_4_NAMES[0], "Store the two-letter code of the state with the most banded first-lien originations as text."),
    _v(TURN_4_NAMES[1], "Store that state's banded first-lien origination count as an integer."),
    _v(TURN_4_NAMES[2], "Store that state's median loan amount as an integer."),
    _v(TURN_4_NAMES[3], "Store that state's unadjusted index for the fourth quarter of the previous year, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store its unadjusted index for the fourth quarter of the activity year, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store the growth between those two readings in percent, rounded to 4 decimals."),
]

DECIMALS = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 4, 4, 4, 4, 4, None,
    0, 0, 0, 0, 4, 4, 4, 4, None, 0,
    None, 0, 0, 4, 4, 4,
]


@lru_cache(maxsize=1)
def _banded():
    records = load_expansion_table("hmda")
    first_lien = records.loc[
        records.action_taken.astype(str).eq(ORIGINATED_ACTION)
        & records.lien_status.astype(str).eq(FIRST_LIEN)
    ].copy()
    for column in ("rate_spread", "loan_to_value_ratio", "loan_amount"):
        first_lien[column] = pd.to_numeric(first_lien[column], errors="coerce")
    inside = first_lien.rate_spread.between(*SPREAD_BAND)
    missing = int(first_lien.rate_spread.isna().sum())
    outside = int((first_lien.rate_spread.notna() & ~inside).sum())
    return len(records), len(first_lien), missing, outside, first_lien.loc[inside].copy()


def _by_type(banded, code):
    return banded.loc[banded.loan_type.astype(str).eq(code)]


@lru_cache(maxsize=1)
def ground_truth():
    record_count, first_lien_count, missing_spread, outside_spread, banded = _banded()
    groups = {code: _by_type(banded, code) for code in LOAN_TYPES}
    shares = {
        code: float(group.rate_spread.ge(HIGHER_PRICED_THRESHOLD).mean()) * 100
        for code, group in groups.items()
    }
    medians = {code: float(group.rate_spread.median()) for code, group in groups.items()}
    below_prime = sorted(code for code, value in medians.items() if value < 0)
    if len(below_prime) != 1:
        raise ValueError(f"expected exactly one programme priced below prime, found {below_prime}")
    ltv = {code: float(group.loan_to_value_ratio.median()) for code, group in groups.items()}
    ltv_counts = {code: int(group.loan_to_value_ratio.notna().sum()) for code, group in groups.items()}
    highest_ltv = min(code for code in LOAN_TYPES if ltv[code] == max(ltv.values()))

    by_state = banded.groupby(banded.state_code.astype(str))
    counts = by_state.size().sort_values(ascending=False)
    leading_state = min(code for code in counts.index if counts[code] == counts.iloc[0])
    state_rows = banded.loc[banded.state_code.astype(str).eq(leading_state)]

    index = load_expansion_table("fhfa_hpi").copy()
    index = index.loc[
        index.level.astype(str).eq("State")
        & index.hpi_type.astype(str).eq(INDEX_TYPE)
        & index.hpi_flavor.astype(str).eq(INDEX_FLAVOR)
        & index.frequency.astype(str).eq("quarterly")
        & index.place_id.astype(str).eq(leading_state)
    ]

    def reading(year):
        cell = index.loc[index.yr.astype(str).eq(year) & index.period.astype(str).eq("4"), "index_nsa"]
        if cell.empty:
            raise ValueError(f"no fourth-quarter index for {leading_state} in {year}")
        return float(pd.to_numeric(cell, errors="coerce").iloc[0])

    prior = reading(str(int(ACTIVITY_YEAR) - 1))
    current = reading(ACTIVITY_YEAR)

    return (
        int(record_count), int(first_lien_count), int(len(banded)),
        int(first_lien_count - len(banded)), missing_spread, outside_spread,
        int(len(groups["1"])), int(len(groups["2"])), int(len(groups["3"])), int(len(groups["4"])),
        shares["1"], shares["2"], shares["3"], shares["4"],
        medians["1"], medians["3"], below_prime[0],
        ltv_counts["1"], ltv_counts["2"], ltv_counts["3"], ltv_counts["4"],
        ltv["1"], ltv["2"], ltv["3"], ltv["4"], highest_ltv,
        int(groups["1"].loan_amount.median()),
        leading_state, int(len(state_rows)), int(state_rows.loan_amount.median()),
        prior, current, (current / prior - 1) * 100,
    )


NAME_OUTPUTS = (
    "negative_median_spread_loan_type_code", "highest_median_ltv_loan_type_code", "leading_state_code",
)


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in NAME_OUTPUTS:
        if name in normalized:
            normalized[name] = str(normalized[name]).strip().upper()
            truth[name] = str(truth[name]).strip().upper()
    return validate_ordered_outputs(
        normalized, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_population_and_band": turn_validator(validate_turn_1),
    "validate_programme_pricing": turn_validator(validate_turn_2),
    "validate_programme_leverage": turn_validator(validate_turn_3),
    "validate_state_price_context": turn_validator(validate_turn_4),
}
