"""A quarter whose published direction reversed, and the regulatory reading beside it.

The Financial Accounts are republished as complete vintages, so the same quarter
carries a different value in every release that covers it. Households' checkable
deposits and currency for 2024-Q3 were first published on 2024-12-12 as a rise of
114,760 million over 2024-Q2; the 2025-09-11 vintage reports the same quarter as
a fall of 68,796 million. Both readings are internally consistent, and neither is
an error: the level for 2024-Q3 was revised down by 241,573 million while the
level for 2024-Q2 was revised down by only 58,017 million, and a quarter that
loses more than the quarter before it changes sign.

FDIC BankFind sits beside this as a reading that is not revised into vintages:
total deposits at insured institutions rose 262,470.957 million over the same two
quarter-ends. It agrees in direction with the first Z.1 release and disagrees
with the current one, which is a scope result rather than an arbitration. Z.1
measures what one holder sector owns; BankFind measures what every depositor has
placed at insured institutions, so business, government, foreign and trust
balances sit inside the second number and outside the first.

Rejected alternative conventions, each measured against the pinned reading:

- Differencing across vintages -- taking 2024-Q3 from the 2024-12-12 release that
  introduced it against 2024-Q2 from the 2024-09-12 release that introduced that,
  which is what a reader of successive press releases accumulates -- gives
  +134,577. It is not the first-release reading (+114,760) and not the current
  reading (-68,796); a quarterly change is defined inside one vintage, and this
  is the wrong path this case measures.
- Reading the series from a single statement table rather than deduplicating it
  across the four that carry it changes no value, because the four agree, but a
  naive row count then reports 819 rows where 273 distinct observations exist.
- Taking BankFind domestic deposits (DEPDOM) instead of total deposits (DEP)
  gives a smaller change of 198,363.293 million; the direction survives but the
  level does not, and the query pins the total.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


SERIES = "FL153020005.Q"
MINIMUM_COVERING_VINTAGES = 4

TURN_1_NAMES = [
    "z1_vintage_count", "series_statement_table_count", "target_quarter_end",
    "first_publishing_vintage_date", "newest_vintage_date",
    "first_published_level_usd_millions", "current_level_usd_millions",
]
TURN_2_NAMES = [
    "first_published_quarterly_change_usd_millions",
    "current_quarterly_change_usd_millions",
    "published_direction_reversed",
]
TURN_3_NAMES = [
    "bankfind_prior_quarter_institution_count",
    "bankfind_target_quarter_institution_count",
    "bankfind_target_quarter_total_deposits_usd_millions",
    "bankfind_deposit_change_usd_millions",
]
TURN_4_NAMES = [
    "reading_matching_bankfind_direction",
    "target_quarter_revision_usd_millions",
    "prior_quarter_revision_usd_millions",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v(TURN_1_NAMES[0], "Store the number of distinct release vintages the Financial Accounts table holds as an integer."),
    _v(TURN_1_NAMES[1], "Store the number of distinct statement tables that carry the requested series as an integer."),
    _v(TURN_1_NAMES[2], "Store the selected quarter end as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[3], "Store the release date of the earliest vintage that covers the selected quarter as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[4], "Store the release date of the newest vintage in the table as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[5], "Store the selected quarter's level in that earliest covering vintage, in USD millions as an integer."),
    _v(TURN_1_NAMES[6], "Store the selected quarter's level in the newest vintage, in USD millions as an integer."),
    _v(TURN_2_NAMES[0], "Store the target-minus-prior quarterly change read inside the earliest covering vintage, in USD millions as an integer."),
    _v(TURN_2_NAMES[1], "Store the target-minus-prior quarterly change read inside the newest vintage, in USD millions as an integer."),
    _v(TURN_2_NAMES[2], "Store whether the two quarterly changes carry opposite signs as a boolean."),
    _v(TURN_3_NAMES[0], "Store the reporting institution count at the prior quarter end as an integer."),
    _v(TURN_3_NAMES[1], "Store the reporting institution count at the target quarter end as an integer."),
    _v(TURN_3_NAMES[2], "Store total deposits at the target quarter end in USD millions, rounded to 3 decimals."),
    _v(TURN_3_NAMES[3], "Store the target-minus-prior change in total deposits in USD millions, rounded to 3 decimals. Compute it from the unrounded quarter-end totals."),
    _v(TURN_4_NAMES[0], "Store exactly one token from: first-published | current | neither."),
    _v(TURN_4_NAMES[1], "Store the newest-minus-earliest revision to the target quarter's level, in USD millions as an integer."),
    _v(TURN_4_NAMES[2], "Store the newest-minus-earliest revision to the prior quarter's level, in USD millions as an integer."),
]

DECIMALS = [0, 0, None, None, None, 0, 0, 0, 0, None, 0, 0, 3, 3, None, 0, 0]


@lru_cache(maxsize=1)
def _vintage_panel() -> pd.DataFrame:
    """One row per vintage and quarter for the requested series.

    The series is printed in several statement tables with the same value, so the
    statement column is dropped and the result asserted unique; a duplicate that
    disagreed would mean the tables had drifted and is not silently averaged.
    """
    rows = load_expansion_table("fed_z1_vintages")
    rows = rows.loc[rows.frequency.eq("Q") & rows.series.eq(SERIES)]
    panel = rows[["vintage", "date", "value"]].drop_duplicates()
    if panel.duplicated(["vintage", "date"]).any():
        raise ValueError("the same vintage and quarter carry more than one value")
    return panel


@lru_cache(maxsize=1)
def _quarters() -> tuple:
    """The selected quarter and the one before it, chosen from vintage coverage.

    The target is the most recent quarter that at least `MINIMUM_COVERING_VINTAGES`
    releases have published, which is the newest quarter whose restatement history
    is long enough to read; the prior quarter is the one immediately before it in
    the same series.
    """
    panel = _vintage_panel()
    covering = panel.groupby(panel.date.astype(str)).vintage.nunique()
    eligible = sorted(covering.loc[covering.ge(MINIMUM_COVERING_VINTAGES)].index)
    if not eligible:
        raise ValueError("no quarter carries enough covering vintages")
    target = eligible[-1]
    quarters = sorted(panel.date.astype(str).unique())
    position = quarters.index(target)
    if position == 0:
        raise ValueError("the selected quarter has no prior quarter in the series")
    return target, quarters[position - 1]


@lru_cache(maxsize=1)
def _deposit_totals() -> dict:
    """Insured-institution deposit totals at the two quarter ends, in USD millions."""
    banks = load_expansion_table("fdic_bankfind")
    out = {}
    for quarter in _quarters():
        rows = banks.loc[banks.REPDTE.astype(str).eq(quarter)]
        if rows.empty:
            raise ValueError(f"no insured-institution rows at {quarter}")
        out[quarter] = (len(rows), float(rows.DEP.sum()) / 1000.0)
    return out


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    panel = _vintage_panel()
    tables = load_expansion_table("fed_z1_vintages")
    tables = tables.loc[tables.frequency.eq("Q") & tables.series.eq(SERIES)]
    target_quarter, prior_quarter = _quarters()
    covering = panel.loc[panel.date.astype(str).eq(target_quarter)]
    earliest = str(covering.vintage.min())
    newest = str(panel.vintage.max())

    def level(vintage: str, quarter: str) -> int:
        cell = panel.loc[panel.vintage.astype(str).eq(vintage) & panel.date.astype(str).eq(quarter)]
        if len(cell) != 1:
            raise ValueError(f"expected one value for {vintage} {quarter}, found {len(cell)}")
        return int(cell.value.iloc[0])

    first_target, first_prior = level(earliest, target_quarter), level(earliest, prior_quarter)
    current_target, current_prior = level(newest, target_quarter), level(newest, prior_quarter)
    first_change = first_target - first_prior
    current_change = current_target - current_prior

    deposits = _deposit_totals()
    prior_count, prior_total = deposits[prior_quarter]
    target_count, target_total = deposits[target_quarter]
    deposit_change = target_total - prior_total

    matches = [
        name for name, change in (("first-published", first_change), ("current", current_change))
        if (change > 0) == (deposit_change > 0)
    ]
    matching = matches[0] if len(matches) == 1 else "neither"

    return (
        int(panel.vintage.nunique()), int(tables.table.nunique()), target_quarter, earliest, newest,
        first_target, current_target,
        first_change, current_change, bool(first_change * current_change < 0),
        prior_count, target_count, target_total, deposit_change,
        matching, current_target - first_target, current_prior - first_prior,
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    places = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[n] for n in names], [truth[n] for n in names], [places[n] for n in names]
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_vintage_population": turn_validator(validate_turn_1),
    "validate_within_vintage_direction": turn_validator(validate_turn_2),
    "validate_deposit_reading": turn_validator(validate_turn_3),
    "validate_reconciliation": turn_validator(validate_turn_4),
}
