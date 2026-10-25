"""Lining up fails-to-deliver with the trades that produced them.

The SEC fails-to-deliver files are dated by settlement date, while FINRA's
daily short-volume file is dated by trading day. A trade settles a fixed number
of settlement days later: two before 2024-05-28 and one from that trading day
on. Settlement days are the days the depository settles: weekdays other than
Federal Reserve holidays and Good Friday. A bank holiday on which the stock
market trades (Columbus Day, Veterans Day) is therefore not a settlement day,
while a day the market closed and the depository did not (2025-01-09, the
national day of mourning) still is.

Trap. Reading the trade-to-settlement step with the wrong cycle is the
plausible wrong path, and it first becomes live in turn 1. The 2025 trading day
with the largest reported volume is 2025-04-09; its trades settled on
2025-04-10, when 515,030,616 shares stood failed across all securities. A
two-day cycle reads 2025-04-11 and 398,091,073; matching the trading day
itself reads 268,107,061. In turn 2 the largest rise into the settlement date
is DMN's 98,318,799 shares; on the two-day reading DMN still leads but with
97,607,770. In turn 3 the wrong path runs the other way: the busiest day in
January through April 2024 is 2024-01-16, whose trades settled on 2024-01-18
under the two-day cycle then in force, with 118,542,824 shares failed, while a
one-day reading gives 2024-01-17 and 121,872,221. In turn 4 a one-day reading
puts the 2024-05-24 trades on 2024-05-28 rather than 2024-05-29, and a
trading-day calendar puts the 2025-01-08 trades on 2025-01-10 and the
2025-10-10 trades on 2025-10-13, a day with no settlement record. Each wrong
date fails exact matching, and each wrong share count differs by far more than
the rounding-boundary tolerance, which is zero for integer counts.

The cycle is visible in the data. Around symbol-days whose reported volume
exceeds ten times the median of the previous twenty trading days and one
million shares, the average fails balance rises most two trading days later
for spikes from 2024-01-03 through 2024-05-20 (by 196,968 shares, over 3,474
events) and one trading day later for spikes from 2024-06-03 on (by 271,282
shares, over 17,269 events).

Readings measured so they are not mistaken for oversights:

1. Volume is the short-volume file's total volume for the day, summed over
   symbols. The busiest 2025 day leads the next, 2025-10-22, by 458,653,085
   shares; the busiest early-2024 day leads 2024-03-04 by 174,121,428. Ranking
   by short volume instead picks 2025-06-12 for 2025 (and the same 2024-01-16
   for early 2024), so the query names total volume.
2. Balances are summed over every row on the date. Four symbol-dates carry two
   rows (different CUSIPs under one symbol); the rise in turn 2 sums them per
   symbol and reads a symbol absent on a date as zero. The runner-up, SUNE,
   rose 96,204,262.
3. The settlement calendar is taken as weekdays less Federal Reserve holidays
   and Good Friday, and reproduces the file's 499 settlement dates exactly.
   Federal Reserve holidays alone would add 2024-03-29 and 2025-04-18, two Good
   Fridays the file does not settle on; no output falls near either.
"""

from datetime import date, timedelta
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


ONE_DAY_CYCLE_FROM = "2024-05-28"
SETTLEMENT_HOLIDAYS = frozenset({
    # Federal Reserve holidays, plus Good Friday (2024-03-29, 2025-04-18).
    "2024-03-29", "2025-04-18",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19",
    "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-11", "2024-11-28",
    "2024-12-25", "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11",
    "2025-11-27", "2025-12-25",
})
RECENT_WINDOW = ("2025-01-01", "2025-12-31")
EARLY_WINDOW = ("2024-01-01", "2024-04-30")
CALENDAR_TRADES = ("2024-05-24", "2025-01-08", "2025-10-10")

OUTPUTS = (
    ("recent_peak_trading_day", None, "the trading day as an ISO YYYY-MM-DD string"),
    ("recent_peak_settlement_date", None, "its trades' settlement date as an ISO YYYY-MM-DD string"),
    ("recent_peak_failed_shares", 0, "the shares failed across all securities on that settlement date as an integer"),
    ("largest_rise_symbol", None, "the symbol as it appears in the fails file"),
    ("largest_rise_shares", 0, "its rise in shares failed as an integer"),
    ("early_peak_trading_day", None, "the trading day as an ISO YYYY-MM-DD string"),
    ("early_peak_settlement_date", None, "its trades' settlement date as an ISO YYYY-MM-DD string"),
    ("early_peak_failed_shares", 0, "the shares failed across all securities on that settlement date as an integer"),
    ("settlement_for_2024_05_24", None, "the settlement date of the first listed day's trades as an ISO YYYY-MM-DD string"),
    ("settlement_for_2025_01_08", None, "the settlement date of the second listed day's trades as an ISO YYYY-MM-DD string"),
    ("settlement_for_2025_10_10", None, "the settlement date of the third listed day's trades as an ISO YYYY-MM-DD string"),
)


variables = [Variable(name, None, f"Store {text}.") for name, _, text in OUTPUTS]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:3]),
    tuple(n for n, _, _ in OUTPUTS[3:5]),
    tuple(n for n, _, _ in OUTPUTS[5:8]),
    tuple(n for n, _, _ in OUTPUTS[8:]),
)


def is_settlement_day(day):
    weekday = date.fromisoformat(day).weekday()
    return weekday < 5 and day not in SETTLEMENT_HOLIDAYS


def settlement_date(trading_day):
    cycle = 1 if trading_day >= ONE_DAY_CYCLE_FROM else 2
    current = date.fromisoformat(trading_day)
    while cycle:
        current += timedelta(days=1)
        if is_settlement_day(current.isoformat()):
            cycle -= 1
    return current.isoformat()


@lru_cache(maxsize=1)
def _fails_by_symbol():
    fails = load_expansion_table("sec_ftd")
    frame = fails.assign(day=fails.settlement_date.astype(str), symbol=fails.symbol.astype(str))
    return frame.groupby(["day", "symbol"]).quantity_fails.sum().astype("int64")


@lru_cache(maxsize=1)
def _daily_volume():
    volume = load_expansion_table("finra_short_volume")
    return volume.assign(day=volume.date.astype(str)).groupby("day").total_volume_shares.sum().astype("int64")


def _peak(window):
    volume = _daily_volume()
    span = volume.loc[(volume.index >= window[0]) & (volume.index <= window[1])]
    return span.sort_index().idxmax()


def _failed_on(day):
    return int(_fails_by_symbol().xs(day, level="day").sum())


@lru_cache(maxsize=1)
def analysis():
    recent_day = _peak(RECENT_WINDOW)
    recent_settlement = settlement_date(recent_day)
    early_day = _peak(EARLY_WINDOW)
    early_settlement = settlement_date(early_day)

    fails = _fails_by_symbol()
    settlement_days = sorted(fails.index.get_level_values("day").unique())
    previous = settlement_days[settlement_days.index(recent_settlement) - 1]
    rise = fails.xs(recent_settlement, level="day").sub(fails.xs(previous, level="day"), fill_value=0)
    rise = rise.sort_index().sort_values(ascending=False, kind="stable")

    return {
        "recent_peak_trading_day": recent_day,
        "recent_peak_settlement_date": recent_settlement,
        "recent_peak_failed_shares": _failed_on(recent_settlement),
        "largest_rise_symbol": rise.index[0],
        "largest_rise_shares": int(rise.iloc[0]),
        "early_peak_trading_day": early_day,
        "early_peak_settlement_date": early_settlement,
        "early_peak_failed_shares": _failed_on(early_settlement),
        **{f"settlement_for_{day.replace('-', '_')}": settlement_date(day) for day in CALENDAR_TRADES},
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[variable.name] for variable in variables)


def _check(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(
        outputs,
        [by_name[n] for n in names],
        [truth[n] for n in names],
        [PRECISION[n] for n in names],
    )


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


def validate_turn_1(outputs):
    return _check(outputs, TURNS[0])


def validate_turn_2(outputs):
    return _check(outputs, TURNS[1])


def validate_turn_3(outputs):
    return _check(outputs, TURNS[2])


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
