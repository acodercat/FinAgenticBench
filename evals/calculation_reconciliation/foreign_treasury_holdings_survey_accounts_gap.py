"""Reconcile a TIC monthly publication with a vintage Financial Accounts line.

The first source is Treasury's Major Foreign Holders of Treasury Securities
table, not a standalone annual survey.  It combines country rows with "All
Other" and "Grand Total" aggregate rows.  Its displayed components are rounded,
so country rows plus the residual need not reproduce the displayed total
exactly.  Country labels are based on legal residence as reported through the
TIC custody chain and remain subject to custodial bias; they are not a clean
beneficial-owner classification.

The second source is the Financial Accounts rest-of-world Treasury-securities
asset line at market value.  It is an aggregate sector line with no country
allocation in the runtime table.  Federal Reserve documentation identifies TIC
data as a basis for many rest-of-world Financial Accounts items, so the two
series are not independent measurements that can be contrasted as residence
versus custody.  Their numerical difference is a release-specific reconciliation
between published series with distinct compilation, valuation, timing and
revision conventions; it cannot identify the cause of the gap or designate one
series as correct.

The four turns first reconcile TIC's displayed rows, then select the newest Z.1
vintage, compare quarter-end levels, and finally quantify revision sensitivity
for the December observation.  The signed revision relative to the first gap is
descriptive; movement toward the TIC level does not make TIC a revision target.

The pinned-reading comparison measures the survey population, the Financial Accounts
release, the survey total basis, the holder scope and the comparison window.
Excluding the "All Other" residual from the reconciliation leaves a $669.700
billion rounding residual instead of $0.200 billion, while counting that same row
as a holder instead reports 38 rows, a $8,619.100 billion country sum and a
-$669.300 billion residual; summing the displayed rows instead
of reading the published total moves every quarterly gap by about $0.100 to
$0.400 billion; taking the earliest rather than the newest Z.1 vintage selects
2024-03-07 and leaves the quarter levels unreported in that release; and
comparing only the final quarter selects 2024-12-31 rather than 2024-03-31 as the
closest gap.  The query pins all five choices.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts, dates and country labels match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


SURVEY_MONTH = "2024-12"
QUARTER_ENDS = ("2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31")
ACCOUNTS_SERIES = "LM263061105.Q"
ACCOUNTS_TABLE = "l133"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "tic_country_row_count", "tic_country_sum_usd_billions", "tic_residual_row_usd_billions",
    "tic_grand_total_usd_billions", "tic_rounding_residual_usd_billions",
    "tic_leading_holder", "tic_leading_holder_usd_billions", "tic_leading_holder_share_pct",
    "tic_leading_holder_margin_usd_billions", "tic_top_five_share_pct",
]
TURN_2_NAMES = [
    "accounts_newest_vintage_date", "accounts_first_quarter_usd_billions",
    "accounts_final_quarter_usd_billions",
]
TURN_3_NAMES = [
    "gap_first_quarter_usd_billions", "gap_second_quarter_usd_billions",
    "gap_third_quarter_usd_billions", "gap_final_quarter_usd_billions",
    "widest_gap_quarter", "widest_gap_pct_of_survey", "closest_gap_quarter",
]
TURN_4_NAMES = [
    "vintages_covering_final_quarter", "first_covering_vintage_date",
    "first_published_accounts_usd_billions", "gap_at_first_release_usd_billions",
    "accounts_revision_as_pct_of_first_gap",
]

variables = [
    _v(TURN_1_NAMES[0], "Store how many country rows the TIC table reports for the month as an integer."),
    _v(TURN_1_NAMES[1], "Store the sum of those country rows in USD billions, rounded to 4 decimals."),
    _v(TURN_1_NAMES[2], "Store the residual aggregate row in USD billions, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3], "Store the published grand total in USD billions, rounded to 4 decimals."),
    _v(TURN_1_NAMES[4], "Store the grand total minus the country sum and the residual row, in USD billions rounded to 4 decimals."),
    _v(TURN_1_NAMES[5], "Store the largest reporting country as the survey names it, as text."),
    _v(TURN_1_NAMES[6], "Store its holding in USD billions, rounded to 4 decimals."),
    _v(TURN_1_NAMES[7], "Store its holding as a percent of the grand total, rounded to 4 decimals."),
    _v(TURN_1_NAMES[8], "Store its holding minus the next country's, in USD billions rounded to 4 decimals."),
    _v(TURN_1_NAMES[9], "Store the five largest countries' combined holding as a percent of the grand total, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the release date of the newest Financial Accounts vintage as an ISO YYYY-MM-DD string."),
    _v(TURN_2_NAMES[1], "Store that release's value for the first quarter end in USD billions, rounded to 6 decimals."),
    _v(TURN_2_NAMES[2], "Store its value for the final quarter end in USD billions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[0], "Store the survey minus accounts gap at the first quarter end in USD billions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[1], "Store the gap at the second quarter end in USD billions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[2], "Store the gap at the third quarter end in USD billions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[3], "Store the gap at the final quarter end in USD billions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[4], "Store the quarter end with the largest absolute gap as an ISO YYYY-MM-DD string."),
    _v(TURN_3_NAMES[5], "Store that gap as a percent of the TIC figure for the same quarter, rounded to 4 decimals."),
    _v(TURN_3_NAMES[6], "Store the quarter end with the smallest absolute gap as an ISO YYYY-MM-DD string."),
    _v(TURN_4_NAMES[0], "Store how many vintages report the series for the final quarter end as an integer."),
    _v(TURN_4_NAMES[1], "Store the release date of the earliest of them as an ISO YYYY-MM-DD string."),
    _v(TURN_4_NAMES[2], "Store that release's value for the final quarter end in USD billions, rounded to 6 decimals."),
    _v(TURN_4_NAMES[3], "Store the gap against the TIC total using that release, in USD billions rounded to 6 decimals."),
    _v(TURN_4_NAMES[4], "Store the revision as a percent of the gap at first release, rounded to 4 decimals."),
]

DECIMALS = [
    0, 4, 4, 4, 4, None, 4, 4, 4, 4,
    None, 6, 6,
    6, 6, 6, 6, None, 4, None,
    0, None, 6, 6, 4,
]


def _normalized_name(value):
    return " ".join(str(value).split()).casefold() if isinstance(value, str) else value


def _survey():
    survey = load_expansion_table("treasury_tic").copy()
    survey["month"] = survey.observation_month.astype(str)
    survey["holdings"] = pd.to_numeric(survey.treasury_holdings_usd_billions, errors="coerce")
    return survey


def _accounts():
    accounts = load_expansion_table("fed_z1_vintages").copy()
    accounts["value"] = pd.to_numeric(accounts.value, errors="coerce")
    return accounts.loc[
        accounts.series.astype(str).eq(ACCOUNTS_SERIES) & accounts.table.astype(str).eq(ACCOUNTS_TABLE)
    ]


def _survey_total(survey, quarter_end):
    row = survey.loc[
        survey.row_type.astype(str).eq("aggregate")
        & survey.month.eq(quarter_end[:7])
        & survey.country_name.astype(str).eq("Grand Total"),
        "holdings",
    ]
    if row.empty:
        raise ValueError(f"the survey has no grand total for {quarter_end}")
    return float(row.iloc[0])


@lru_cache(maxsize=1)
def ground_truth():
    survey = _survey()
    countries = survey.loc[
        survey.row_type.astype(str).eq("country") & survey.month.eq(SURVEY_MONTH)
    ].sort_values(["holdings", "country_name"], ascending=[False, True]).reset_index(drop=True)
    aggregates = survey.loc[survey.row_type.astype(str).eq("aggregate") & survey.month.eq(SURVEY_MONTH)]
    residual_row = aggregates.loc[aggregates.country_name.astype(str).eq("All Other"), "holdings"]
    grand_total = _survey_total(survey, f"{SURVEY_MONTH}-01")
    if residual_row.empty:
        raise ValueError("the survey has no residual aggregate row for the month")
    residual = float(residual_row.iloc[0])
    country_sum = float(countries.holdings.sum())
    leader, runner_up = countries.iloc[0], countries.iloc[1]

    accounts = _accounts()
    vintages = sorted(accounts.vintage.astype(str).unique())
    newest = vintages[-1]
    current = accounts.loc[accounts.vintage.astype(str).eq(newest)]
    values = {}
    for quarter_end in QUARTER_ENDS:
        cell = current.loc[current.date.astype(str).eq(quarter_end), "value"]
        if cell.empty:
            raise ValueError(f"the newest release does not cover {quarter_end}")
        values[quarter_end] = float(cell.iloc[0]) / 1000

    gaps = {
        quarter_end: _survey_total(survey, quarter_end) - values[quarter_end]
        for quarter_end in QUARTER_ENDS
    }
    widest = max(gaps, key=lambda key: abs(gaps[key]))
    closest = min(gaps, key=lambda key: abs(gaps[key]))

    final = QUARTER_ENDS[-1]
    covering = sorted(
        accounts.loc[accounts.date.astype(str).eq(final)].vintage.astype(str).unique()
    )
    first_release = covering[0]
    first_value = float(
        accounts.loc[
            accounts.vintage.astype(str).eq(first_release) & accounts.date.astype(str).eq(final), "value"
        ].iloc[0]
    ) / 1000
    gap_at_first = _survey_total(survey, final) - first_value
    revision = values[final] - first_value

    return (
        int(len(countries)),
        country_sum,
        residual,
        grand_total,
        grand_total - country_sum - residual,
        str(leader.country_name),
        float(leader.holdings),
        float(leader.holdings) / grand_total * 100,
        float(leader.holdings - runner_up.holdings),
        float(countries.holdings.head(5).sum()) / grand_total * 100,
        newest,
        values[QUARTER_ENDS[0]],
        values[final],
        gaps[QUARTER_ENDS[0]],
        gaps[QUARTER_ENDS[1]],
        gaps[QUARTER_ENDS[2]],
        gaps[final],
        widest,
        gaps[widest] / _survey_total(survey, widest) * 100,
        closest,
        int(len(covering)),
        first_release,
        first_value,
        gap_at_first,
        revision / gap_at_first * 100,
    )


NAME_OUTPUTS = ("tic_leading_holder",)


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
    "validate_survey_composition": turn_validator(validate_turn_1),
    "validate_accounts_series": turn_validator(validate_turn_2),
    "validate_source_gap_path": turn_validator(validate_turn_3),
    "validate_release_dependence": turn_validator(validate_turn_4),
}
