"""Three measures of the same settlement failures, and three different answers.

The trap is the denominator of an average, and it fires in the first turn.
A security's undelivered balance is a level that stands on every settlement date,
including the dates the file leaves it out, because leaving it out means the
balance was nil. The average balance per settlement date therefore divides the
summed balance by the 126 dates in the window. Grouping by symbol and calling the
frame's own mean instead divides by the number of dates the security happens to
appear on, which is the pandas default and reads as the same sentence in code.

The window's settlement-date count is reported in the last turn rather than the
first, so the first turn never hands the denominator over as a prior result. The
reader still has to decide that a date the file omits is a date with no
undelivered balance, which is the judgement the query leaves open.

Measured. The correct leader averages 3,226,336.2778 shares per settlement date
and stands 17.4323 percent above the runner-up. The default mean returns
3,946,780.3010 shares for that same security, 22.3301 percent high, and collapses
the margin to 1.4707 percent because the runner-up gains more from the shorter
denominator than the leader does. The wrong path therefore changes two of the
first turn's outputs, the average and the margin, and leaves the leading symbol
correct -- the model reaches the right security and reports the wrong size of its
problem. At four requested decimals the rounding-boundary tolerance is 0.00006
shares against a separation of 720,444.0232, so neither wrong figure can enter
the accepted band. Turns two through four rest on summed balances and summed
valued balances, which the denominator never touches, so the path stays confined
to the first turn.

The second measure is the case's financial point. Valuing each balance at its own
date's reported price moves the lead from a low-priced satellite-radio issuer to
an index fund: 10,295.9579 million USD against a runner-up 16.8980 percent
behind, while the share-count leader falls to fifteenth. Counting shares ranks
securities by how many are cheap; counting dollars ranks them by how much capital
is undelivered, and a failure in a fund share is generally an artifact of
creation and redemption rather than a short position. The third measure shows
that the obvious way to normalise either of them is unavailable where it matters:
511 of the 3,694 cohort securities carry a reported share count and only 4 of the
twenty largest by valued balance do, so the carrying securities hold 35.6545
percent of the cohort's balance but only 18.5381 percent of its valued balance.
Coverage falls as exposure rises, because an exchange-traded fund files no
periodic report carrying a share count.

Two conventions the query leaves out because they change nothing. Whether the
share-count search honours amendments as well as original periodic filings moves
the qualifying ticker universe from 3,824 to 3,828 and leaves all four of the
third turn's outputs identical. Choosing a single filing per ticker is likewise
idle: the case asks whether a count exists, never what it is. Only the filing
cutoff is load bearing, and it excludes 199 tickers whose first reported count
arrives after the window.

Measured alternative readings, all pinned by the query. Dropping the dates that
report no price before summing balances leaves the leader and its average
unchanged but moves the margin to 74.8327 percent, because a fifth of the
window's failed shares -- 8,101 rows, 1.2574 percent of the window, carrying
20.2522 percent of its shares -- sit on rows with no price; the query says a
balance stands whether or not a price is reported, and the module values a
balance only where a price exists. Moving the recurrence floor to a quarter or
three quarters of the dates changes the cohort to 8,580 or 944 securities and
leaves both leaders where they are, so the floor is pinned for the count it
reports rather than for the ranking.

No rank the case uses is tied: the average-balance leader stands 17.4323 percent
above the runner-up, the valued leader 16.8980 percent, and the twentieth valued
security 6.6970 percent above the twenty-first.

The final turn spans a shortened settlement cycle. Average balances rose from
78,554,279.1452 to 93,658,451.4444 shares per date across the two halves of 2024,
which hold 124 and 126 settlement dates, a ratio of 1.1923, which runs opposite to the expectation that a shorter cycle
leaves less room for delivery to fail. Nothing in these two sources separates
that move from issuance, volume or composition, and the required claim asks the
response to say so.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, load_financial_statements
from core.validation import turn_validator, validate_ordered_outputs


WINDOW_START = "2024-07-01"
WINDOW_END = "2024-12-31"
FIRST_HALF_START = "2024-01-01"
FIRST_HALF_END = "2024-06-30"
PRESENCE_FLOOR = 0.5
LEADING_DOLLAR_COUNT = 20


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "recurring_symbol_count",
    "largest_average_balance_symbol",
    "largest_average_balance_shares",
    "largest_average_balance_margin_pct",
]
TURN_2_NAMES = [
    "dollar_exposure_leader_symbol",
    "dollar_exposure_leader_usd_millions",
    "dollar_exposure_runner_up_symbol",
    "dollar_exposure_leader_margin_pct",
    "share_leader_dollar_exposure_rank",
]
TURN_3_NAMES = [
    "cohort_symbols_with_share_count",
    "leading_dollar_symbols_with_share_count",
    "share_weighted_share_count_coverage_pct",
    "dollar_weighted_share_count_coverage_pct",
]
TURN_4_NAMES = [
    "first_half_settlement_date_count",
    "later_half_settlement_date_count",
    "first_half_mean_daily_balance_shares",
    "second_half_mean_daily_balance_shares",
    "second_half_to_first_half_ratio",
]

variables = [
    _v(TURN_1_NAMES[0],
       "Store the number of securities meeting the recurrence floor as an integer."),
    _v(TURN_1_NAMES[1],
       "Store the leading security's trading symbol as its exact source identifier string."),
    _v(TURN_1_NAMES[2],
       "Store its average balance in shares, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3],
       "Store how far that average stands above the runner-up's in percent, computed from the "
       "unrounded averages, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0],
       "Store that security's trading symbol as its exact source identifier string."),
    _v(TURN_2_NAMES[1],
       "Store its summed valued balance in USD millions, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2],
       "Store the runner-up's trading symbol as its exact source identifier string."),
    _v(TURN_2_NAMES[3],
       "Store how far the leader stands above the runner-up in percent, computed from the "
       "unrounded sums, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4],
       "Store the share-count leader's position in the valued ranking as an integer, counting "
       "from one."),
    _v(TURN_3_NAMES[0],
       "Store how many cohort securities carry a reported share count as an integer."),
    _v(TURN_3_NAMES[1],
       "Store how many of the leading valued securities carry one as an integer."),
    _v(TURN_3_NAMES[2],
       "Store the share-count-carrying securities' portion of the cohort's summed balance in "
       "percent, computed from the unrounded sums, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3],
       "Store their portion of the cohort's summed valued balance in percent, computed from the "
       "unrounded sums, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0],
       "Store the number of settlement dates the earlier half holds as an integer."),
    _v(TURN_4_NAMES[1],
       "Store the number the later half holds as an integer."),
    _v(TURN_4_NAMES[2],
       "Store the cohort's average balance per settlement date in the earlier half, in shares, "
       "rounded to 4 decimals."),
    _v(TURN_4_NAMES[3],
       "Store the same figure for the later half, in shares, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4],
       "Store the later figure divided by the earlier one, computed from the unrounded averages, "
       "rounded to 4 decimals."),
]

DECIMALS = [
    0, None, 4, 4,
    None, 4, None, 4, 0,
    0, 0, 4, 4,
    0, 0, 4, 4, 4,
]


@lru_cache(maxsize=1)
def _fails():
    """Fails-to-deliver rows for the whole of 2024, with the valued balance attached."""
    fails = load_expansion_table("sec_ftd")
    frame = fails.loc[:, ["settlement_date", "symbol", "quantity_fails", "price"]].copy()
    del fails
    frame["settled_on"] = pd.to_datetime(frame.settlement_date)
    frame = frame.loc[frame.settled_on.between(FIRST_HALF_START, WINDOW_END)].copy()
    frame["symbol"] = frame.symbol.astype(str)
    frame["balance"] = pd.to_numeric(frame.quantity_fails, errors="coerce")
    frame["reported_price"] = pd.to_numeric(frame.price, errors="coerce")
    if frame.balance.isna().any():
        raise ValueError("a fails row reports an unparseable balance")
    # A fifth of the window's failed shares sit on rows with no reported price, so
    # the valued balance exists only where a price does and the query says so.
    frame["valued"] = frame.balance * frame.reported_price
    return frame


def _window(frame, start, end):
    return frame.loc[frame.settled_on.between(start, end)]


@lru_cache(maxsize=1)
def _cohort():
    """Securities recorded on at least half the later half's settlement dates."""
    later = _window(_fails(), WINDOW_START, WINDOW_END)
    dates = int(later.settled_on.nunique())
    appearances = later.groupby("symbol").settled_on.nunique()
    return dates, later, appearances.loc[appearances.ge(dates * PRESENCE_FLOOR)].index


def _ranked(series):
    """Descending by value, breaking an exact tie by the lower symbol."""
    return series.sort_index().sort_values(ascending=False, kind="stable")


@lru_cache(maxsize=1)
def _share_count_universe():
    """Tickers a periodic filing made by the window's end reports a share count for.

    The case asks whether a share count exists, never what it is, so this is a
    membership test and no filing has to be chosen among a ticker's several.

    The count is the balance-sheet shares outstanding above zero, as the query
    states. Filings also carry the cover-page count, and the two readings part:
    16 cohort securities report only the cover-page count and one (RENT) only a
    balance-sheet zero, so counting either figure, reported at all, gives 528
    securities holding 36.0394 and 18.9858 percent where the stated count gives
    511, 35.6545 and 18.5381.
    """
    statements = load_financial_statements()
    frame = statements.loc[:, ["ticker", "filed_date", "CommonStockSharesOutstanding"]].copy()
    del statements
    frame["ticker"] = frame.ticker.astype(str).str.upper()
    frame["filed_on"] = pd.to_datetime(frame.filed_date, errors="coerce")
    frame["shares"] = pd.to_numeric(frame.CommonStockSharesOutstanding, errors="coerce")
    return set(frame.loc[
        frame.shares.gt(0)
        & frame.filed_on.notna()
        & frame.filed_on.le(WINDOW_END)
        & ~frame.ticker.isin(("", "NAN", "NONE"))
    ].ticker)


@lru_cache(maxsize=1)
def ground_truth():
    dates, later, cohort = _cohort()
    held = later.loc[later.symbol.isin(cohort)]

    balance_total = held.groupby("symbol").balance.sum()
    average_balance = _ranked(balance_total / dates)
    share_leader = str(average_balance.index[0])

    valued_total = _ranked(held.groupby("symbol").valued.sum())
    dollar_leader = str(valued_total.index[0])

    carried = _share_count_universe()
    # Membership is a property of the cohort, not of the valued ranking: a security
    # whose dates all lack a price still belongs to the cohort, and counting over
    # the valued index would make this answer depend on how absent prices are held.
    with_count = sorted(symbol for symbol in cohort if symbol.upper() in carried)
    leading = valued_total.head(LEADING_DOLLAR_COUNT).index

    earlier = _window(_fails(), FIRST_HALF_START, FIRST_HALF_END)
    earlier_dates = int(earlier.settled_on.nunique())
    earlier_mean = float(earlier.loc[earlier.symbol.isin(cohort)].balance.sum()) / earlier_dates
    later_mean = float(held.balance.sum()) / dates

    return (
        int(len(cohort)),
        share_leader,
        float(average_balance.iloc[0]),
        100.0 * (float(average_balance.iloc[0]) / float(average_balance.iloc[1]) - 1.0),
        dollar_leader,
        float(valued_total.iloc[0]) / 1e6,
        str(valued_total.index[1]),
        100.0 * (float(valued_total.iloc[0]) / float(valued_total.iloc[1]) - 1.0),
        int(list(valued_total.index).index(share_leader) + 1),
        int(len(with_count)),
        int(sum(1 for symbol in leading if symbol.upper() in carried)),
        100.0 * float(balance_total.reindex(with_count).sum()) / float(balance_total.sum()),
        100.0 * float(valued_total.reindex(with_count).fillna(0.0).sum())
        / float(valued_total.sum()),
        earlier_dates,
        dates,
        earlier_mean,
        later_mean,
        later_mean / earlier_mean,
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_recurring_balance(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_valued_exposure(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_share_count_coverage(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_half_year_comparison(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_recurring_balance": turn_validator(validate_recurring_balance),
    "validate_valued_exposure": turn_validator(validate_valued_exposure),
    "validate_share_count_coverage": turn_validator(validate_share_count_coverage),
    "validate_half_year_comparison": turn_validator(validate_half_year_comparison),
}
