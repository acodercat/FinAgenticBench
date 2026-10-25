"""Securities that failed to deliver every day of a quarter, and what short volume shows.

A fail to deliver is a settlement outcome, and the reflex reading is that a
security failing day after day is being shorted hard. The 2025-Q1 record does not
support that reading. Forty-four symbols failed on every one of the 61 settlement
dates in the quarter. Of those, only sixteen appear in the FINRA daily
short-volume file at all, and their median short share of reported volume is
54.7488 per cent against a market-wide median of 46.4495 -- above it, but by
eight points rather than by the margin the reflex predicts. Their median reported
volume is 23,297,682.5 shares against 2,862,359.5 for the file as a whole, so the
persistent failures sit in liquid names, not obscure ones.

The twenty-eight unmatched members carry the case's second point. The largest
peak among them, CYDY at 6,783,255 shares, has no short-volume record at all: the
FINRA file reports what its member firms print on their own tapes, so a security
can fail every day of a quarter and leave no trace in it. An analyst who screens
for shorting pressure by joining these two files sees neither the whole
population nor a position: short volume is executed daily flow flagged short,
which is not short interest, and a fail is a settlement failure, which is not a
position either.

Rejected alternative conventions, each measured against the pinned reading:

- Keying the fail population by CUSIP rather than by trading symbol also returns
  44 persistent securities here, so the two agree on the count; the symbol is
  pinned because it is the key the short-volume file uses.
- Requiring a fail on at least half the settlement dates rather than on all of
  them returns 3,883 securities, of which 3,372 are reported, and a cohort median
  short share of 51.0812 per cent, which moves the comparison most of the way
  back to the market median.
- Taking the mean rather than the median gives 57.3545 per cent for the cohort
  against 46.5410 for the file, preserving the direction while changing both
  levels.
- Ranking the cohort by fails summed across dates instead of by the largest
  single-date level selects CYDY, whose 198,947,427 summed shares lead the
  cohort, rather than PACB. That is the wrong path this case measures: a summed
  level double counts a fail that stays open across settlement dates, and CYDY is
  precisely the security the short-volume file does not report.
"""

from __future__ import annotations

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


QUARTER_START = "2025-01-01"
QUARTER_END = "2025-03-31"

TURN_1_NAMES = [
    "settlement_date_count", "failing_symbol_count", "persistent_fail_symbol_count",
    "persistent_peak_leader_symbol", "persistent_peak_leader_quantity_shares",
]
TURN_2_NAMES = [
    "short_volume_symbol_count", "cohort_matched_symbol_count",
    "cohort_unmatched_symbol_count", "peak_leader_short_volume_share_pct",
]
TURN_3_NAMES = [
    "cohort_median_short_volume_share_pct", "market_median_short_volume_share_pct",
    "cohort_median_reported_volume_shares", "market_median_reported_volume_shares",
]
TURN_4_NAMES = [
    "largest_unmatched_symbol", "largest_unmatched_peak_quantity_shares",
    "cohort_short_share_exceeds_market",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v(TURN_1_NAMES[0], "Store the number of distinct settlement dates in the quarter as an integer."),
    _v(TURN_1_NAMES[1], "Store the number of distinct trading symbols with at least one fail in the quarter as an integer."),
    _v(TURN_1_NAMES[2], "Store the number of symbols that fail on every settlement date of the quarter as an integer."),
    _v(TURN_1_NAMES[3], "Store the selected symbol exactly as the fails file writes it."),
    _v(TURN_1_NAMES[4], "Store that symbol's largest single-date fail quantity in shares as an integer."),
    _v(TURN_2_NAMES[0], "Store the number of distinct symbols the short-volume file reports in the quarter as an integer."),
    _v(TURN_2_NAMES[1], "Store how many of the persistent-fail symbols the short-volume file reports as an integer."),
    _v(TURN_2_NAMES[2], "Store how many of them it does not report as an integer."),
    _v(TURN_2_NAMES[3], "Store the selected symbol's quarter short volume divided by its quarter reported volume in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store the median short share across the matched persistent-fail symbols in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the median short share across every symbol the file reports in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the median quarter reported volume across the matched persistent-fail symbols in shares, rounded to 1 decimal; an even-sized population puts the median between two share counts."),
    _v(TURN_3_NAMES[3], "Store the median quarter reported volume across every symbol the file reports in shares, rounded to 1 decimal; an even-sized population puts the median between two share counts."),
    _v(TURN_4_NAMES[0], "Store the selected unmatched symbol exactly as the fails file writes it."),
    _v(TURN_4_NAMES[1], "Store that symbol's largest single-date fail quantity in shares as an integer."),
    _v(TURN_4_NAMES[2], "Store whether the matched cohort's median short share is above the market median as a boolean."),
]

DECIMALS = [0, 0, 0, None, 0, 0, 0, 0, 4, 4, 4, 1, 1, None, 0, None]


@lru_cache(maxsize=1)
def _fails():
    """Quarter fails keyed by trading symbol, with the settlement date as the day."""
    rows = load_expansion_table("sec_ftd")
    return rows.loc[rows.settlement_date.astype(str).between(QUARTER_START, QUARTER_END)].assign(
        symbol=lambda frame: frame.symbol.astype(str)
    )


@lru_cache(maxsize=1)
def _short_volume():
    """Quarter short volume per symbol, dropping symbols with no reported volume.

    A zero denominator is not a zero share, so those symbols leave the ratio
    population rather than entering it as 0 per cent.
    """
    rows = load_expansion_table("finra_short_volume")
    rows = rows.loc[rows.date.astype(str).between(QUARTER_START, QUARTER_END)]
    agg = rows.groupby(rows.symbol.astype(str)).agg(
        short=("short_volume_shares", "sum"), total=("total_volume_shares", "sum")
    )
    agg = agg.loc[agg.total > 0]
    return agg.assign(share=agg.short / agg.total * 100.0)


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    fails = _fails()
    grouped = fails.groupby("symbol")
    date_count = int(fails.settlement_date.nunique())
    fail_days = grouped.settlement_date.nunique()
    peak = grouped.quantity_fails.max()
    cohort = sorted(fail_days.loc[fail_days.eq(date_count)].index)

    def top(symbols):
        subset = peak.reindex(symbols)
        best = subset.max()
        return sorted(subset.loc[subset.eq(best)].index)[0], int(best)

    leader, leader_peak = top(cohort)

    volume = _short_volume()
    matched = [s for s in cohort if s in volume.index]
    unmatched = [s for s in cohort if s not in volume.index]
    if not matched or not unmatched:
        raise ValueError("the cohort must split into reported and unreported symbols")
    unmatched_leader, unmatched_peak = top(unmatched)

    cohort_share = float(volume.loc[matched, "share"].median())
    market_share = float(volume.share.median())
    return (
        date_count, int(fail_days.size), len(cohort), leader, leader_peak,
        int(len(volume)), len(matched), len(unmatched), float(volume.loc[leader, "share"]),
        cohort_share, market_share,
        float(volume.loc[matched, "total"].median()), float(volume.total.median()),
        unmatched_leader, unmatched_peak, bool(cohort_share > market_share),
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
    "validate_fail_population": turn_validator(validate_turn_1),
    "validate_short_volume_match": turn_validator(validate_turn_2),
    "validate_cohort_comparison": turn_validator(validate_turn_3),
    "validate_unreported_scope": turn_validator(validate_turn_4),
}
