"""Test two balance-sheet identities, one that holds and one that cannot.

A balance sheet foots: assets equal the total of liabilities and equity. Across
the 4,274 annual statement observations with a 2024-12-31 balance-sheet date, 4,165 of the
4,168 that report both sides agree to the dollar and the largest disagreement
anywhere is 4,000 USD. The identity that looks like the same statement does not
hold: subtracting liabilities and stockholders' equity from assets leaves a
non-zero remainder on 1,145 of 3,516 observations, with a median of 26.1 million USD
and a maximum of 38,333,124,000 USD at KKR. The reason is a reporting
convention, not an error -- the equity line is the parent's share -- and the
remainder can be closed exactly: KKR reports 36,747,947,000 USD of
noncontrolling interests and 1,585,177,000 USD of redeemable noncontrolling
interests carried outside permanent equity, and those two sum to the remainder
with nothing left over.

The last turn measures a selection effect that any ratio screen on this table
inherits. Of the 4,274 observations, 3,096, or 72.4380 percent, present both
current subtotals; 3,095 of those also have positive current liabilities. The 1,178 that do not present both are led
by real estate investment trusts and commercial banks, whose balance sheets are
ordered by liquidity rather than split into current and non-current. A current
ratio computed on the positive-denominator observations is therefore a statistic about
non-financial issuers, and its median of 1.6604 should not be quoted as a
market-wide figure.

Rejected alternatives (each measured in the `balance_sheet_identity` comparison): including quarterly reports and amendments in the annual
population; counting a remainder as zero when it is merely small rather than
exactly zero; closing the remainder with noncontrolling interests alone and
omitting the redeemable portion; and computing the current ratio over all
observations by treating an absent subtotal as zero. The query pins all five, so this
is a hard baseline.

Ranking the remainders by absolute size rather than by signed size reproduces
the same leader and is registered as an equivalent: the largest positive
remainder is 38,333,124,000 USD while the most negative is -744,442,000, so no
negative case comes close.

The population carries one repeated issuer: 4,274 statement observations map to
4,273 central index keys, so an observation count and an issuer count are not
interchangeable here.
Ratio screens are also scale-sensitive, which is why the ratio leader is drawn
from observations with at least one billion USD of current assets; without that
floor the ranking is dominated by shell and pre-revenue registrants.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts, identifiers and names match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_facts, load_financial_statements
from core.validation import turn_validator, validate_ordered_outputs


BALANCE_SHEET_DATE = "2024-12-31"
ANNUAL_FORM = "10-K"
CURRENT_ASSET_FLOOR = 1_000_000_000.0
MINORITY_CONCEPT = "MinorityInterest"
REDEEMABLE_CONCEPT = "RedeemableNoncontrollingInterestEquityCarryingAmount"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "annual_statement_observation_count", "distinct_issuer_count", "footing_identity_observation_count",
    "footing_identity_exact_count", "footing_identity_max_absolute_usd",
]
TURN_2_NAMES = [
    "parent_identity_observation_count", "parent_identity_exact_count", "parent_identity_residual_count",
    "median_nonzero_residual_usd", "largest_residual_company", "largest_residual_usd",
    "largest_residual_runner_up_company", "largest_residual_runner_up_usd",
]
TURN_3_NAMES = [
    "largest_residual_minority_interest_usd", "largest_residual_redeemable_interest_usd",
    "largest_residual_components_total_usd", "largest_residual_closure_difference_usd",
    "largest_residual_minority_share_pct",
]
TURN_4_NAMES = [
    "current_subtotals_present_observation_count", "classified_balance_sheet_share_pct",
    "current_subtotals_missing_observation_count", "unclassified_leading_sic_code", "unclassified_leading_sic_count",
    "median_current_ratio", "large_current_asset_observation_count", "large_current_asset_ratio_leader",
    "large_current_asset_leader_ratio", "large_current_asset_leader_margin",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the number of qualifying annual statement observations as an integer."),
    _v(TURN_1_NAMES[1], "Store how many distinct central index keys they cover as an integer."),
    _v(TURN_1_NAMES[2], "Store how many observations report both sides of the footing identity as an integer."),
    _v(TURN_1_NAMES[3], "Store how many of those agree exactly as an integer."),
    _v(TURN_1_NAMES[4], "Store the largest absolute difference on the footing identity in USD, rounded to 2 decimals."),
    _v(TURN_2_NAMES[0], "Store how many observations report assets, liabilities and the parent equity line as an integer."),
    _v(TURN_2_NAMES[1], "Store how many of them leave exactly zero as an integer."),
    _v(TURN_2_NAMES[2], "Store how many leave a non-zero remainder as an integer."),
    _v(TURN_2_NAMES[3], "Store the median non-zero remainder in USD, rounded to 2 decimals."),
    _v(TURN_2_NAMES[4], "Store the company with the largest positive remainder as the table names it, as text."),
    _v(TURN_2_NAMES[5], "Store that remainder in USD, rounded to 2 decimals."),
    _v(TURN_2_NAMES[6], "Store the company with the second-largest positive remainder as text."),
    _v(TURN_2_NAMES[7], "Store that remainder in USD, rounded to 2 decimals."),
    _v(TURN_3_NAMES[0], "Store the leading company's reported noncontrolling interests in USD, rounded to 2 decimals."),
    _v(TURN_3_NAMES[1], "Store its reported redeemable noncontrolling interests in USD, rounded to 2 decimals."),
    _v(TURN_3_NAMES[2], "Store the sum of those two components in USD, rounded to 2 decimals."),
    _v(TURN_3_NAMES[3], "Store the remainder minus that sum in USD, rounded to 2 decimals."),
    _v(TURN_3_NAMES[4], "Store the noncontrolling interests as a percent of the remainder, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store how many observations present both current subtotals as an integer."),
    _v(TURN_4_NAMES[1], "Store that count as a percent of all qualifying observations, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store how many observations lack at least one current subtotal as an integer."),
    _v(TURN_4_NAMES[3], "Store the most common industry classification code among those observations as its "
       "four-digit SIC code in text form, without a decimal point."),
    _v(TURN_4_NAMES[4], "Store how many of them carry that code as an integer."),
    _v(TURN_4_NAMES[5], "Store the median current ratio among observations with both subtotals and positive current liabilities, rounded to 4 decimals."),
    _v(TURN_4_NAMES[6], "Store how many of those observations report at least one billion USD of current assets as an integer."),
    _v(TURN_4_NAMES[7], "Store the highest current ratio among them by company name, as text."),
    _v(TURN_4_NAMES[8], "Store that ratio, rounded to 4 decimals."),
    _v(TURN_4_NAMES[9], "Store that ratio minus the runner-up's, rounded to 6 decimals."),
]

DECIMALS = [
    0, 0, 0, 0, 2,
    0, 0, 0, 2, None, 2, None, 2,
    2, 2, 2, 2, 4,
    0, 4, 0, None, 0, 4, 0, None, 4, 6,
]


def _normalized_name(value):
    return " ".join(str(value).split()).casefold() if isinstance(value, str) else value


NUMERIC_COLUMNS = (
    "Assets", "Liabilities", "StockholdersEquity", "LiabilitiesAndStockholdersEquity",
    "AssetsCurrent", "LiabilitiesCurrent",
)


@lru_cache(maxsize=1)
def _annual():
    statements = load_financial_statements().copy()
    annual = statements.loc[
        statements.form.astype(str).eq(ANNUAL_FORM)
        & statements.report_date.astype(str).eq(BALANCE_SHEET_DATE)
    ].copy()
    for column in NUMERIC_COLUMNS:
        annual[column] = pd.to_numeric(annual[column], errors="coerce")
    return annual


def _concept_value(facts, cik, concept):
    rows = facts.loc[
        facts.cik.astype(str).str.lstrip("0").eq(str(cik).lstrip("0"))
        & facts.concept.astype(str).eq(concept)
        & facts.end_date.astype(str).eq(BALANCE_SHEET_DATE),
        "value",
    ].dropna().unique()
    if len(rows) != 1:
        raise ValueError(f"{concept} for {cik}: expected one value, found {len(rows)}")
    return float(rows[0])


@lru_cache(maxsize=1)
def ground_truth():
    annual = _annual()
    footing = annual.dropna(subset=["Assets", "LiabilitiesAndStockholdersEquity"]).copy()
    footing["difference"] = footing.Assets - footing.LiabilitiesAndStockholdersEquity

    parent = annual.dropna(subset=["Assets", "Liabilities", "StockholdersEquity"]).copy()
    parent["residual"] = parent.Assets - parent.Liabilities - parent.StockholdersEquity
    non_zero = parent.loc[parent.residual.ne(0)]
    ranked = parent.sort_values(["residual", "cik"], ascending=[False, True]).reset_index(drop=True)
    leader, runner_up = ranked.iloc[0], ranked.iloc[1]

    facts = load_facts()
    minority = _concept_value(facts, leader.cik, MINORITY_CONCEPT)
    redeemable = _concept_value(facts, leader.cik, REDEEMABLE_CONCEPT)

    current_subtotals_present = annual.dropna(subset=["AssetsCurrent", "LiabilitiesCurrent"])
    ratio_population = current_subtotals_present.loc[
        current_subtotals_present.LiabilitiesCurrent.gt(0)
    ].copy()
    ratio_population["current_ratio"] = (
        ratio_population.AssetsCurrent / ratio_population.LiabilitiesCurrent
    )
    unclassified = annual.loc[annual.AssetsCurrent.isna() | annual.LiabilitiesCurrent.isna()]
    # The sic column arrives as float64, so a plain astype(str) reports '6798.0'.
    # SIC codes are four-digit integers and that is how the filer's own record
    # states them, so the code is normalised before it is counted or reported.
    sic_counts = (
        unclassified.sic.dropna().astype(float).astype(int).astype(str).value_counts()
    )

    large = ratio_population.loc[ratio_population.AssetsCurrent.ge(CURRENT_ASSET_FLOOR)].sort_values(
        ["current_ratio", "cik"], ascending=[False, True]
    ).reset_index(drop=True)

    return (
        int(len(annual)),
        int(annual.cik.nunique()),
        int(len(footing)),
        int(footing.difference.eq(0).sum()),
        float(footing.difference.abs().max()),
        int(len(parent)),
        int(parent.residual.eq(0).sum()),
        int(len(non_zero)),
        float(non_zero.residual.median()),
        str(leader.company_name),
        float(leader.residual),
        str(runner_up.company_name),
        float(runner_up.residual),
        minority,
        redeemable,
        minority + redeemable,
        float(leader.residual) - minority - redeemable,
        minority / float(leader.residual) * 100,
        int(len(current_subtotals_present)),
        float(len(current_subtotals_present) / len(annual) * 100),
        int(len(unclassified)),
        str(sic_counts.index[0]),
        int(sic_counts.iloc[0]),
        float(ratio_population.current_ratio.median()),
        int(len(large)),
        str(large.iloc[0].company_name),
        float(large.iloc[0].current_ratio),
        float(large.iloc[0].current_ratio - large.iloc[1].current_ratio),
    )


NAME_OUTPUTS = (
    "largest_residual_company", "largest_residual_runner_up_company",
    "large_current_asset_ratio_leader",
)


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in NAME_OUTPUTS:
        if name in normalized:
            normalized[name] = _normalized_name(normalized[name])
            truth[name] = _normalized_name(truth[name])
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
    "validate_footing_identity": turn_validator(validate_turn_1),
    "validate_parent_equity_residual": turn_validator(validate_turn_2),
    "validate_residual_closure": turn_validator(validate_turn_3),
    "validate_classified_selection": turn_validator(validate_turn_4),
}
