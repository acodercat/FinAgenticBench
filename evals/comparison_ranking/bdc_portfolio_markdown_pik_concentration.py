"""Compare BDC portfolio marks, positive PIK-rate exposure, and rate structure.

This is a four-turn baseline.  It fixes a 2024-12-31 cross-section by keeping
the latest filed accession for each filer, restricts portfolio aggregation to
rows with both cost and fair value, and compares dollar cost-minus-fair-value
with fair-value-to-cost severity.  A positive signed gap is an unrealized net
markdown in the reported schedule; a negative gap is net appreciation.  It is
not a realized loss and the table does not establish the filer's eventual
recovery or exit proceeds.

Turn 2 defines a PIK-tagged position as one with a strictly positive reported
PIK rate.  Missing and zero rates are both outside that tagged set.  The result
is the fair value of tagged positions, not PIK cash income, accrued income, or
principal capitalization actually recognized during a period.

The two BDC rate columns use mixed representations in the frozen table.  Turn
3 therefore states a reproducible normalization: values above 1 are percentage
points and are divided by 100; values at or below 1 are decimal fractions.
Rows with a negative normalized component or an all-in rate below the spread
are reported and excluded before taking medians.  The implied base is an
arithmetic residual, not proof that an individual contract references SOFR;
the comparison with the published 2024-12-31 SOFR is descriptive.

Turn 4 lowers the minimum number of priced positions from fifty to twenty-five.
It measures population-cutoff sensitivity and the one-sided rows excluded by
the paired-value rule.  One-sided means exactly one of cost and fair value is
reported: counting every row missing either side instead reports 6,539 rather
than 3,853, because 2,686 rows carry neither.  A narrow ranking margin does
not by itself establish that a restatement occurred or predict that the
ordering will reverse.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and names match exactly.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


REPORT_DATE = "2024-12-31"
INVESTMENT_FLOOR = 50
SENSITIVITY_FLOOR = 25
BASE_RATE_TYPE = "SOFR"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "cohort_filer_count", "cohort_investment_count", "cohort_cost_usd_billions",
    "cohort_fair_value_usd_billions", "cohort_marked_pct",
]
TURN_2_NAMES = [
    "largest_depreciation_filer", "largest_depreciation_usd_millions", "largest_depreciation_marked_pct",
    "largest_depreciation_margin_usd_millions",
    "lowest_marked_filer", "lowest_marked_pct", "lowest_marked_depreciation_usd_millions",
    "lowest_marked_margin_pct", "lowest_marked_depreciation_rank",
]
TURN_3_NAMES = [
    "pik_reporting_filer_count", "cohort_pik_fair_value_share_pct",
    "highest_pik_share_filer", "highest_pik_share_pct", "highest_pik_share_margin_pct",
    "lowest_marked_filer_pik_share_pct",
]
TURN_4_NAMES = [
    "rate_pair_row_count", "rate_unit_normalized_row_count", "rate_invalid_row_count",
    "floating_rate_investment_count", "median_all_in_rate_pct", "median_spread_pct",
    "median_implied_base_rate_pct", "reference_rate_pct", "implied_base_minus_reference_bp",
]
TURN_5_NAMES = [
    "sensitivity_filer_count", "sensitivity_lowest_marked_filer", "sensitivity_lowest_marked_pct",
    "sensitivity_margin_pct", "sensitivity_rank_of_pinned_leader", "excluded_single_sided_row_count",
]

variables = [
    _v("cohort_filer_count", "Store the number of filers in the cohort as an integer."),
    _v("cohort_investment_count", "Store the number of cohort investments as an integer."),
    _v("cohort_cost_usd_billions", "Store the cohort's total cost in USD billions, rounded to 6 decimals."),
    _v("cohort_fair_value_usd_billions", "Store the cohort's total fair value in USD billions, rounded to 6 decimals."),
    _v("cohort_marked_pct", "Store cohort fair value as a percent of cohort cost, rounded to 4 decimals."),
    _v("largest_depreciation_filer", "Store the name of the filer with the largest unrealized depreciation as text."),
    _v("largest_depreciation_usd_millions", "Store that filer's unrealized depreciation in USD millions, rounded to 3 decimals."),
    _v("largest_depreciation_marked_pct", "Store that filer's fair value as a percent of its cost, rounded to 4 decimals."),
    _v("largest_depreciation_margin_usd_millions", "Store that filer's unrealized depreciation minus the runner-up's, in USD millions rounded to 3 decimals."),
    _v("lowest_marked_filer", "Store the name of the filer with the lowest marked percentage as text."),
    _v("lowest_marked_pct", "Store that filer's fair value as a percent of its cost, rounded to 4 decimals."),
    _v("lowest_marked_depreciation_usd_millions", "Store that filer's unrealized depreciation in USD millions, rounded to 3 decimals."),
    _v("lowest_marked_margin_pct", "Store the runner-up's marked percentage minus that filer's, in percentage points rounded to 4 decimals."),
    _v("lowest_marked_depreciation_rank", "Store that filer's rank by unrealized depreciation, 1 being largest, as an integer."),
    _v("pik_reporting_filer_count", "Store how many cohort filers report at least one investment carrying a strictly positive paid-in-kind rate, as an integer."),
    _v("cohort_pik_fair_value_share_pct", "Store the cohort fair value in investments with a strictly positive paid-in-kind rate as a percent of cohort fair value, rounded to 4 decimals."),
    _v("highest_pik_share_filer", "Store the name of the filer with the highest paid-in-kind fair-value share as text."),
    _v("highest_pik_share_pct", "Store that share in percent, rounded to 4 decimals."),
    _v("highest_pik_share_margin_pct", "Store that share minus the runner-up filer's share, in percentage points rounded to 4 decimals."),
    _v("lowest_marked_filer_pik_share_pct", "Store the paid-in-kind fair-value share of the lowest-marked filer from turn 1, in percent rounded to 4 decimals."),
    _v("rate_pair_row_count", "Store the number of cohort rows reporting both an all-in rate and a spread as an integer."),
    _v("rate_unit_normalized_row_count", "Store how many paired rows have either raw rate above 1 and therefore require percentage-point normalization, as an integer."),
    _v("rate_invalid_row_count", "Store how many paired rows are excluded after normalization for a negative component or an all-in rate below the spread, as an integer."),
    _v("floating_rate_investment_count", "Store the number of valid normalized rate pairs entering the median calculations as an integer."),
    _v("median_all_in_rate_pct", "Store the median normalized all-in rate in percent, rounded to 4 decimals."),
    _v("median_spread_pct", "Store the median normalized spread in percent, rounded to 4 decimals."),
    _v("median_implied_base_rate_pct", "Store the median implied base rate in percent, rounded to 4 decimals."),
    _v("reference_rate_pct", "Store the reference rate published for the report date in percent, rounded to 4 decimals."),
    _v("implied_base_minus_reference_bp", "Store the median implied base rate minus the reference rate in basis points, rounded to 2 decimals."),
    _v("sensitivity_filer_count", "Store the number of filers in the lower-floor cohort as an integer."),
    _v("sensitivity_lowest_marked_filer", "Store the name of the lowest-marked filer under the lower floor as text."),
    _v("sensitivity_lowest_marked_pct", "Store that filer's marked percentage, rounded to 4 decimals."),
    _v("sensitivity_margin_pct", "Store the runner-up's marked percentage minus that filer's, in percentage points rounded to 4 decimals."),
    _v("sensitivity_rank_of_pinned_leader", "Store the rank the pinned-floor leader holds under the lower floor as an integer."),
    _v("excluded_single_sided_row_count", "Store how many rows at the report date report only one of cost and fair value, as an integer."),
]

DECIMALS = [
    0, 0, 6, 6, 4, None, 3, 4, 3, None, 4, 3, 4, 0,
    0, 4, None, 4, 4, 4,
    0, 0, 0, 0, 4, 4, 4, 4, 2,
    0, None, 4, 4, 0, 0,
]


def _normalized_name(value):
    return " ".join(str(value).split()).casefold() if isinstance(value, str) else value


def _latest_filings_at_report_date():
    schedule = load_expansion_table("sec_bdc").copy()
    schedule["report_date"] = schedule.report_date.astype(str)
    at_date = schedule.loc[schedule.report_date.eq(REPORT_DATE)].copy()
    latest = at_date.groupby("cik").filed_date.max().rename("latest_filed_date")
    at_date = at_date.merge(latest, on="cik")
    return at_date.loc[at_date.filed_date.eq(at_date.latest_filed_date)].copy()


def _cohort(rows, floor):
    priced = rows.dropna(subset=["cost_usd", "fair_value_usd"])
    counts = priced.groupby("cik").investment_id.size()
    return priced.loc[priced.cik.isin(set(counts[counts.ge(floor)].index))].copy()


def _by_filer(cohort):
    frame = cohort.groupby(["cik", "filer_name"], as_index=False).agg(
        cost=("cost_usd", "sum"), fair_value=("fair_value_usd", "sum"), investments=("investment_id", "size"),
    )
    frame = frame.loc[frame.cost.gt(0)].copy()
    frame["depreciation"] = frame.cost - frame.fair_value
    frame["marked"] = frame.fair_value / frame.cost * 100
    return frame


@lru_cache(maxsize=1)
def ground_truth():
    at_date = _latest_filings_at_report_date()
    cohort = _cohort(at_date, INVESTMENT_FLOOR)
    filers = _by_filer(cohort)

    by_dollars = filers.sort_values(["depreciation", "cik"], ascending=[False, True]).reset_index(drop=True)
    by_marked = filers.sort_values(["marked", "cik"]).reset_index(drop=True)
    dollar_leader, marked_leader = by_dollars.iloc[0], by_marked.iloc[0]
    marked_leader_dollar_rank = int(by_dollars.index[by_dollars.cik.eq(marked_leader.cik)][0]) + 1

    pik_value = cohort.loc[cohort.pik_interest_rate.fillna(0).gt(0)].groupby("cik").fair_value_usd.sum()
    filers["pik_fair_value"] = filers.cik.map(pik_value).fillna(0.0)
    filers["pik_share"] = filers.pik_fair_value / filers.fair_value * 100
    by_pik = filers.sort_values(["pik_share", "cik"], ascending=[False, True]).reset_index(drop=True)
    marked_leader_pik = float(filers.loc[filers.cik.eq(marked_leader.cik), "pik_share"].iloc[0])

    rate_pairs = cohort.dropna(subset=["total_interest_rate", "basis_spread"]).copy()
    requires_normalization = rate_pairs.total_interest_rate.gt(1) | rate_pairs.basis_spread.gt(1)
    rate_pairs["normalized_all_in"] = rate_pairs.total_interest_rate.where(
        rate_pairs.total_interest_rate.le(1), rate_pairs.total_interest_rate / 100
    )
    rate_pairs["normalized_spread"] = rate_pairs.basis_spread.where(
        rate_pairs.basis_spread.le(1), rate_pairs.basis_spread / 100
    )
    invalid_rate = (
        rate_pairs.normalized_all_in.lt(0)
        | rate_pairs.normalized_spread.lt(0)
        | rate_pairs.normalized_all_in.lt(rate_pairs.normalized_spread)
    )
    floating = rate_pairs.loc[~invalid_rate].copy()
    floating["implied_base"] = floating.normalized_all_in - floating.normalized_spread
    rates = load_expansion_table("nyfed_reference_rates").copy()
    reference = rates.loc[
        rates["Effective Date"].astype(str).eq(REPORT_DATE) & rates["Rate Type"].astype(str).eq(BASE_RATE_TYPE),
        "Rate (%)",
    ]
    if reference.empty:
        raise ValueError("the reference rate is missing for the report date")
    reference_rate = float(reference.iloc[0])
    median_base = float(floating.implied_base.median() * 100)

    lower = _by_filer(_cohort(at_date, SENSITIVITY_FLOOR)).sort_values(["marked", "cik"]).reset_index(drop=True)
    pinned_rank = int(lower.index[lower.cik.eq(marked_leader.cik)][0]) + 1
    # "exactly one" is one side missing, not either side missing: 2,686 rows carry
    # neither cost nor fair value, and counting those as single-sided would report
    # 6,539 where the question asks for 3,853.
    single_sided = int((at_date.cost_usd.isna() ^ at_date.fair_value_usd.isna()).sum())

    return (
        int(len(filers)), int(filers.investments.sum()), float(filers.cost.sum()) / 1e9,
        float(filers.fair_value.sum()) / 1e9, float(filers.fair_value.sum() / filers.cost.sum() * 100),
        str(dollar_leader.filer_name), float(dollar_leader.depreciation) / 1e6, float(dollar_leader.marked),
        float(dollar_leader.depreciation - by_dollars.iloc[1].depreciation) / 1e6,
        str(marked_leader.filer_name), float(marked_leader.marked), float(marked_leader.depreciation) / 1e6,
        float(by_marked.iloc[1].marked - marked_leader.marked), marked_leader_dollar_rank,
        int((filers.pik_fair_value > 0).sum()),
        float(filers.pik_fair_value.sum() / filers.fair_value.sum() * 100),
        str(by_pik.iloc[0].filer_name), float(by_pik.iloc[0].pik_share),
        float(by_pik.iloc[0].pik_share - by_pik.iloc[1].pik_share), marked_leader_pik,
        int(len(rate_pairs)), int(requires_normalization.sum()), int(invalid_rate.sum()), int(len(floating)),
        float(floating.normalized_all_in.median() * 100),
        float(floating.normalized_spread.median() * 100), median_base, reference_rate,
        (median_base - reference_rate) * 100,
        int(len(lower)), str(lower.iloc[0].filer_name), float(lower.iloc[0].marked),
        float(lower.iloc[1].marked - lower.iloc[0].marked), pinned_rank, single_sided,
    )


NAME_OUTPUTS = (
    "largest_depreciation_filer", "lowest_marked_filer", "highest_pik_share_filer",
    "sensitivity_lowest_marked_filer",
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
def validate_turn_5(outputs): return _validate_subset(outputs, TURN_5_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_cohort_construction": turn_validator(validate_turn_1),
    "validate_markdown_leadership": turn_validator(validate_turn_2),
    "validate_pik_concentration": turn_validator(validate_turn_3),
    "validate_rate_structure": turn_validator(validate_turn_4),
    "validate_floor_sensitivity": turn_validator(validate_turn_5),
}
