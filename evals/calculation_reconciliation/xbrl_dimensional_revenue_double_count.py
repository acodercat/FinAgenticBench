"""Diagnose overlap in SEC Notes dimensioned revenue facts.

The case pairs full-year dimensioned revenue facts with zero-dimension primary
presentation facts on accession and period. Summing every dimensioned row is a
diagnostic, not a valid revenue aggregate: presentations can repeat facts, axes
can offer alternative views of the same amount, an axis can include a subtotal
beside its components, and some axes disclose amounts outside consolidation.

Turn two explicitly returns to all zero-dimension presentation rows to measure
presentation repetition, then restricts dimensioned facts to presentation index
zero for the axis analysis. The observed association between axis count and the
sum-to-consolidated ratio is summarized by bands; it does not establish that
every axis is a complete, nonoverlapping partition.

The single-axis test is also diagnostic. An axis close to consolidated revenue
may be a useful reconciliation candidate, but proximity within a tolerance does
not prove completeness, while an axis above the total may contain internal
overlap or a different consolidation scope. Because the queries pin each scope
and ask for these diagnostics directly, this is a hard baseline rather than a
wrong-method trap with competing PV answers.

Numeric validation uses the shared rounding tolerance; counts and filer names
match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


TAG = "RevenueFromContractWithCustomerExcludingAssessedTax"
FORM = "10-K"
ANNUAL_QUARTERS = "4"
CURRENCY = "USD"
PRIMARY_PRESENTATION = "0"
OVERLAP_THRESHOLD = 1.05
STRONG_OVERLAP_THRESHOLD = 1.5
RECONCILE_TOLERANCE = 0.01
TIGHT_TOLERANCE = 0.001


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["dimensioned_rows", "dimensioned_filings", "consolidated_observations",
                "paired_observations", "median_ratio", "seventy_fifth_ratio", "maximum_ratio",
                "share_above_double_pct"]
TURN_2_NAMES = ["multirow_consolidated_observations", "multivalue_consolidated_observations", "primary_median_ratio",
                "primary_maximum_ratio", "median_axes", "one_axis_observations", "one_axis_median_ratio",
                "two_axis_observations", "two_axis_median_ratio", "three_axis_observations",
                "three_axis_median_ratio"]
TURN_3_NAMES = ["anchor_name", "anchor_consolidated_usd_billions", "anchor_rows",
                "anchor_sum_usd_billions", "anchor_ratio", "anchor_segment_axis_usd_billions",
                "anchor_investee_axis_usd_billions", "anchor_investee_axis_ratio"]
TURN_4_NAMES = ["axis_group_count", "axis_groups_above_1_05_pct", "axis_groups_above_1_5_pct",
                "reconciling_observation_share_pct", "tightly_reconciling_share_pct"]

variables = [
    _v(TURN_1_NAMES[0], "Store how many dimensioned rows the population holds as an integer."),
    _v(TURN_1_NAMES[1], "Store how many filings they come from as an integer."),
    _v(TURN_1_NAMES[2], "Store how many filing-and-period observations carry a consolidated figure as an integer."),
    _v(TURN_1_NAMES[3], "Store how many observations carry both as an integer."),
    _v(TURN_1_NAMES[4], "Store the median ratio of the added rows to the consolidated figure, rounded to 4 decimals."),
    _v(TURN_1_NAMES[5], "Store its seventy-fifth percentile, rounded to 4 decimals."),
    _v(TURN_1_NAMES[6], "Store its maximum, rounded to 4 decimals."),
    _v(TURN_1_NAMES[7], "Store the percent of observations above 2, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store how many all-presentation consolidated observations carry more than one row as an integer."),
    _v(TURN_2_NAMES[1], "Store how many all-presentation consolidated observations carry more than one distinct value as an integer."),
    _v(TURN_2_NAMES[2], "Store the median ratio using the primary presentation only, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store its maximum, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store the median number of distinct axes per observation, rounded to 4 decimals."),
    _v(TURN_2_NAMES[5], "Store how many observations carry exactly one axis as an integer."),
    _v(TURN_2_NAMES[6], "Store their median ratio, rounded to 4 decimals."),
    _v(TURN_2_NAMES[7], "Store how many carry exactly two axes as an integer."),
    _v(TURN_2_NAMES[8], "Store their median ratio, rounded to 4 decimals."),
    _v(TURN_2_NAMES[9], "Store how many carry exactly three axes as an integer."),
    _v(TURN_2_NAMES[10], "Store their median ratio, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store the selected filer name verbatim from SEC notes as a string, preserving capitalization, punctuation and spacing."),
    _v(TURN_3_NAMES[1], "Store its consolidated figure in USD billions, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store how many dimensioned rows that observation carries as an integer."),
    _v(TURN_3_NAMES[3], "Store their sum in USD billions, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4], "Store the ratio of the sum to the consolidated figure, rounded to 4 decimals."),
    _v(TURN_3_NAMES[5], "Store the business segment axis total in USD billions, rounded to 4 decimals."),
    _v(TURN_3_NAMES[6], "Store the equity method investee axis total in USD billions, rounded to 4 decimals."),
    _v(TURN_3_NAMES[7], "Store that axis as a ratio of the consolidated figure, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store how many filing-period-axis groups the primary presentation holds as an integer."),
    _v(TURN_4_NAMES[1], "Store the percent summing above 1.05 times consolidated, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the percent above 1.5 times, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3], "Store the percent of observations with an axis within 1 percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the percent with an axis within a tenth of a percent, rounded to 4 decimals."),
]

DECIMALS = [0, 0, 0, 0, 4, 4, 4, 4,
            0, 0, 4, 4, 4, 0, 4, 0, 4, 0, 4,
            None, 4, 0, 4, 4, 4, 4, 4,
            0, 4, 4, 4, 4]

SEGMENT_AXIS = "BusinessSegments"
INVESTEE_AXIS = "ScheduleOfEquityMethodInvestmentEquityMethodInvesteeName"


def _common_filter(frame):
    return (
        frame.tag.astype(str).eq(TAG)
        & frame.form.astype(str).eq(FORM)
        & frame.qtrs.astype(str).eq(ANNUAL_QUARTERS)
        & frame.uom.astype(str).eq(CURRENCY)
        & frame.coreg.isna()
    )


@lru_cache(maxsize=1)
def _dimensioned():
    frame = load_expansion_table("sec_notes_dimensional")
    kept = frame.loc[_common_filter(frame), ["adsh", "name", "ddate", "value", "axis", "iprx"]].copy()
    kept["amount"] = pd.to_numeric(kept.value, errors="coerce")
    kept["period"] = kept.ddate.astype(str)
    return kept


@lru_cache(maxsize=1)
def _consolidated():
    frame = load_expansion_table("sec_notes")
    kept = frame.loc[_common_filter(frame) & frame.dimn.astype(str).eq("0"),
                     ["adsh", "name", "ddate", "value", "iprx"]].copy()
    kept["amount"] = pd.to_numeric(kept.value, errors="coerce")
    kept["period"] = kept.ddate.astype(str)
    return kept


def _reference():
    rows = _consolidated()
    primary = rows.loc[rows.iprx.astype(str).eq(PRIMARY_PRESENTATION)]
    return primary.groupby(["adsh", "period"], as_index=False).agg(
        consolidated=("amount", "max"), filer=("name", "first"),
    )


def _ratios(rows, reference):
    totals = rows.groupby(["adsh", "period"], as_index=False).amount.sum()
    paired = reference.merge(totals, on=["adsh", "period"])
    paired = paired.loc[paired.consolidated.gt(0)].copy()
    paired["ratio"] = paired.amount / paired.consolidated
    return paired


@lru_cache(maxsize=1)
def ground_truth():
    dimensioned = _dimensioned()
    consolidated = _consolidated()
    reference = _reference()

    every = _ratios(dimensioned, reference)
    primary_rows = dimensioned.loc[dimensioned.iprx.astype(str).eq(PRIMARY_PRESENTATION)]
    primary = _ratios(primary_rows, reference)

    grouped = consolidated.groupby(["adsh", "period"])
    repeated = int(grouped.size().gt(1).sum())
    disagreeing = int(grouped.amount.nunique().gt(1).sum())

    axes_per = primary_rows.groupby(["adsh", "period"]).axis.nunique()
    primary_with_axes = primary.merge(
        axes_per.rename("axis_count").reset_index(), on=["adsh", "period"], how="left",
    )

    def axis_band(count):
        band = primary_with_axes.loc[primary_with_axes["axis_count"].eq(count)]
        return int(len(band)), float(band.ratio.median())

    leader = every.sort_values(["ratio", "adsh", "period"], ascending=[False, True, True]).iloc[0]
    anchor = dimensioned.loc[dimensioned.adsh.eq(leader.adsh) & dimensioned.period.eq(leader.period)]
    by_axis = anchor.groupby(anchor.axis.astype(str)).amount.sum()

    axis_groups = primary_rows.groupby(["adsh", "period", "axis"], as_index=False).amount.sum()
    axis_groups = axis_groups.merge(reference, on=["adsh", "period"])
    axis_groups = axis_groups.loc[axis_groups.consolidated.gt(0)].copy()
    axis_groups["share"] = axis_groups.amount / axis_groups.consolidated
    axis_groups["gap"] = (axis_groups.amount - axis_groups.consolidated).abs() / axis_groups.consolidated
    best = axis_groups.groupby(["adsh", "period"]).gap.min()

    return (
        int(len(dimensioned)), int(dimensioned.adsh.nunique()), int(len(reference)), int(len(every)),
        float(every.ratio.median()), float(every.ratio.quantile(0.75)), float(every.ratio.max()),
        float(every.ratio.gt(2).mean()) * 100,
        repeated, disagreeing,
        float(primary.ratio.median()), float(primary.ratio.max()), float(axes_per.median()),
        *axis_band(1), *axis_band(2), *axis_band(3),
        str(leader.filer), float(leader.consolidated) / 1e9, int(len(anchor)),
        float(leader.amount) / 1e9, float(leader.ratio),
        float(by_axis.get(SEGMENT_AXIS, 0.0)) / 1e9,
        float(by_axis.get(INVESTEE_AXIS, 0.0)) / 1e9,
        float(by_axis.get(INVESTEE_AXIS, 0.0)) / float(leader.consolidated),
        int(len(axis_groups)),
        float(axis_groups.share.gt(OVERLAP_THRESHOLD).mean()) * 100,
        float(axis_groups.share.gt(STRONG_OVERLAP_THRESHOLD).mean()) * 100,
        float(best.lt(RECONCILE_TOLERANCE).mean()) * 100,
        float(best.lt(TIGHT_TOLERANCE).mean()) * 100,
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_naive_aggregation": turn_validator(validate_turn_1),
    "validate_presentation_repeats": turn_validator(validate_turn_2),
    "validate_anchor_decomposition": turn_validator(validate_turn_3),
    "validate_axis_reconciliation": turn_validator(validate_turn_4),
}
