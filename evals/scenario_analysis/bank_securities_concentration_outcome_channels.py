"""Compare later bank outcomes by starting securities concentration.

The case forms a survivor panel at two year-end dates, records the coincident
ten-year par-yield move, and assigns equal-frequency fifths by each bank's
starting securities-to-assets share with certificate tie-breaking. Later turns
compare margin, deposit, book-equity and cumulative-income outcomes across those
fixed groups and within a large-bank subset.

The design is descriptive. Total securities do not identify accounting
classification, duration, purchase date or unrealized gains and losses. Book
equity changes combine earnings with distributions, capital actions, other
comprehensive income and other balance-sheet changes, none of which these fields
decompose. Positive cumulative earnings alongside lower ending equity establishes
only a negative net contribution from all non-earnings changes.

The survivor requirement omits banks that failed or merged, group comparisons do
not control for business model or other confounders, and a large-bank restriction
does not turn the result into a causal estimate. Correlations and tail-group
medians are different descriptive summaries and need not convey the same pattern.

This is a hard baseline case. Numeric validation uses the shared rounding
tolerance; counts match exactly.

The `bank_securities_channels` pinned-reading comparison records sensitivity to sorting on the
later date, to sorting on the securities amount rather than its share of assets, to
using group means rather than medians, to differencing the two group medians rather
than taking each bank's own change and then the median, to substituting domestic for
total deposits, and to dropping the asset floor. The aggregation order is the one worth
naming: the median is not linear, so the lowest group's margin change is 0.1901
percentage points as the median of per-bank changes and 0.134534 as the difference of
the two medians, and the highest group's is 0.0551 against 0.097894. Reporting both
group medians and then subtracting them is the natural reading of a loose phrasing, so
the query now states that the change is taken per bank first. Every one of these
choices is pinned by the query, making them robustness comparisons rather than hidden
answer paths.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


START_DATE = "2021-12-31"
END_DATE = "2024-12-31"
MINIMUM_ASSETS_THOUSAND = 1e5
LARGE_BANK_THRESHOLD_THOUSAND = 1e7
QUINTILE_COUNT = 5
INCOME_YEARS = ("2022-12-31", "2023-12-31", "2024-12-31")
CURVE_TENOR = "10 Yr"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "panel_bank_count", "start_yield_pct", "end_yield_pct", "yield_change_bp",
    "lowest_fifth_median_securities_share_pct", "highest_fifth_median_securities_share_pct",
    "top_fifth_lower_bound_pct",
]
TURN_2_NAMES = [
    "lowest_fifth_start_margin_pct", "lowest_fifth_end_margin_pct", "lowest_fifth_margin_change_pp",
    "highest_fifth_start_margin_pct", "highest_fifth_end_margin_pct", "highest_fifth_margin_change_pp",
    "margin_change_correlation",
]
TURN_3_NAMES = [
    "lowest_fifth_deposit_growth_pct", "middle_fifth_deposit_growth_pct",
    "highest_fifth_deposit_growth_pct", "lowest_fifth_equity_growth_pct",
    "highest_fifth_equity_growth_pct", "negative_equity_growth_fifth_count",
    "lowest_fifth_earnings_over_equity_pct", "highest_fifth_earnings_over_equity_pct",
]
TURN_4_NAMES = [
    "large_bank_count", "large_lowest_fifth_deposit_growth_pct", "large_highest_fifth_deposit_growth_pct",
    "large_lowest_fifth_equity_growth_pct", "large_highest_fifth_equity_growth_pct",
    "deposit_growth_correlation", "equity_growth_correlation",
]

variables = [
    _v(TURN_1_NAMES[0], "Store how many banks the matched panel holds as an integer."),
    _v(TURN_1_NAMES[1], "Store the ten-year par yield at the start date in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[2], "Store it at the end date in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3], "Store the change between them in basis points, rounded to 4 decimals."),
    _v(TURN_1_NAMES[4], "Store the median starting securities share of the lowest fifth in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[5], "Store the median starting securities share of the highest fifth in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[6], "Store the securities share at which the highest fifth begins, in percent rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the lowest fifth's median margin at the start date in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store its median margin at the end date, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the change between them in percentage points, rounded to 4 decimals. Take each bank's own change first, then the median across banks."),
    _v(TURN_2_NAMES[3], "Store the highest fifth's median margin at the start date, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store its median margin at the end date, rounded to 4 decimals."),
    _v(TURN_2_NAMES[5], "Store the change between them in percentage points, rounded to 4 decimals. Take each bank's own change first, then the median across banks."),
    _v(TURN_2_NAMES[6], "Store the correlation across the panel between starting securities share and margin change, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store the lowest fifth's median deposit growth in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the middle fifth's median deposit growth in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the highest fifth's median deposit growth in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store the lowest fifth's median equity growth in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4], "Store the highest fifth's median equity growth in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[5], "Store how many of the five fifths show negative median equity growth as an integer."),
    _v(TURN_3_NAMES[6], "Store the lowest fifth's median cumulative net income as a percent of starting equity, rounded to 4 decimals."),
    _v(TURN_3_NAMES[7], "Store the highest fifth's median of the same quantity, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store how many panel banks exceed the large-bank threshold at the start date as an integer."),
    _v(TURN_4_NAMES[1], "Store the lowest fifth's median deposit growth among them in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the highest fifth's median deposit growth among them, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3], "Store the lowest fifth's median equity growth among them, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the highest fifth's median equity growth among them, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store the panel correlation between starting securities share and deposit growth, rounded to 4 decimals."),
    _v(TURN_4_NAMES[6], "Store the panel correlation between starting securities share and equity growth, rounded to 4 decimals."),
]

DECIMALS = [
    0, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 0, 4, 4,
    0, 4, 4, 4, 4, 4, 4,
]


@lru_cache(maxsize=1)
def _panel():
    frame = load_expansion_table("fdic_bankfind").copy()
    frame["day"] = frame.REPDTE.astype(str)
    for column in ("ASSET", "DEP", "SC", "NIMY", "EQ", "NETINC"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    income = frame.loc[frame.day.isin(INCOME_YEARS)].groupby("CERT").NETINC.sum().rename("cumulative_income")
    start = frame.loc[frame.day.eq(START_DATE)].copy()
    end = frame.loc[frame.day.eq(END_DATE)].copy()
    start["securities_share"] = start.SC / start.ASSET * 100
    merged = start[["CERT", "securities_share", "ASSET", "NIMY", "DEP", "EQ"]].rename(
        columns={"ASSET": "start_assets", "NIMY": "start_margin", "DEP": "start_deposits", "EQ": "start_equity"}
    ).merge(
        end[["CERT", "NIMY", "DEP", "EQ"]].rename(
            columns={"NIMY": "end_margin", "DEP": "end_deposits", "EQ": "end_equity"}
        ), on="CERT",
    ).merge(income, on="CERT", how="left")
    merged = merged.loc[
        merged.start_assets.ge(MINIMUM_ASSETS_THOUSAND)
        & merged.start_deposits.gt(0) & merged.start_equity.gt(0)
    ].copy()
    merged["deposit_growth"] = (merged.end_deposits / merged.start_deposits - 1) * 100
    merged["equity_growth"] = (merged.end_equity / merged.start_equity - 1) * 100
    merged["margin_change"] = merged.end_margin - merged.start_margin
    merged["earnings_over_equity"] = merged.cumulative_income / merged.start_equity * 100
    return _assign_fifths(merged)


def _assign_fifths(frame):
    ordered = frame.sort_values(["securities_share", "CERT"]).copy()
    ordered["fifth"] = pd.qcut(
        ordered.securities_share.rank(method="first"),
        QUINTILE_COUNT,
        labels=range(1, QUINTILE_COUNT + 1),
    )
    return ordered


def _fifth(panel, index, column):
    rows = panel.loc[panel.fifth.astype(int).eq(index), column].dropna()
    if rows.empty:
        raise ValueError(f"fifth {index} has no values for {column}")
    return float(rows.median())


def _yield(report_date):
    curve = load_expansion_table("treasury_yield_curve").copy()
    cell = curve.loc[curve.Date.astype(str).eq(report_date), CURVE_TENOR]
    if cell.empty:
        raise ValueError(f"no curve reading for {report_date}")
    return float(pd.to_numeric(cell, errors="coerce").iloc[0])


@lru_cache(maxsize=1)
def ground_truth():
    panel = _panel()
    start_yield, end_yield = _yield(START_DATE), _yield(END_DATE)
    top_bound = float(panel.loc[panel.fifth.astype(int).eq(QUINTILE_COUNT), "securities_share"].min())

    large = _assign_fifths(panel.loc[panel.start_assets.gt(LARGE_BANK_THRESHOLD_THOUSAND)].copy())

    equity_by_fifth = [_fifth(panel, index, "equity_growth") for index in range(1, QUINTILE_COUNT + 1)]

    return (
        int(len(panel)),
        start_yield,
        end_yield,
        (end_yield - start_yield) * 100,
        _fifth(panel, 1, "securities_share"),
        _fifth(panel, QUINTILE_COUNT, "securities_share"),
        top_bound,
        _fifth(panel, 1, "start_margin"),
        _fifth(panel, 1, "end_margin"),
        _fifth(panel, 1, "margin_change"),
        _fifth(panel, QUINTILE_COUNT, "start_margin"),
        _fifth(panel, QUINTILE_COUNT, "end_margin"),
        _fifth(panel, QUINTILE_COUNT, "margin_change"),
        float(panel.securities_share.corr(panel.margin_change)),
        _fifth(panel, 1, "deposit_growth"),
        _fifth(panel, 3, "deposit_growth"),
        _fifth(panel, QUINTILE_COUNT, "deposit_growth"),
        equity_by_fifth[0],
        equity_by_fifth[-1],
        int(sum(1 for value in equity_by_fifth if value < 0)),
        _fifth(panel, 1, "earnings_over_equity"),
        _fifth(panel, QUINTILE_COUNT, "earnings_over_equity"),
        int(len(large)),
        _fifth(large, 1, "deposit_growth"),
        _fifth(large, QUINTILE_COUNT, "deposit_growth"),
        _fifth(large, 1, "equity_growth"),
        _fifth(large, QUINTILE_COUNT, "equity_growth"),
        float(panel.securities_share.corr(panel.deposit_growth)),
        float(panel.securities_share.corr(panel.equity_growth)),
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
    "validate_panel_construction": turn_validator(validate_turn_1),
    "validate_margin_channel": turn_validator(validate_turn_2),
    "validate_funding_and_capital_channels": turn_validator(validate_turn_3),
    "validate_size_control": turn_validator(validate_turn_4),
}
