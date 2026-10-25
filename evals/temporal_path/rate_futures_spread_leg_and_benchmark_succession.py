"""A third position leg, a contract that stops, and a rate that compounds.

The Commitments of Traders report gives each trader category three numbers, not
two. Long and short are directional. Spreading is an equal long and short held at
once, counted in neither, and on the successor contract's final report date it is
the largest of the three: 4,094,368 contracts against 1,683,620 long and
2,160,187 short.

The trap is the net position, and it fires in the first turn. Net is the two
directional legs, -476,567 contracts. Adding the third returns +3,617,801, which
is the wrong sign as well as the wrong size. The count is stored at zero
decimals, where the rounding-boundary tolerance is exactly 0 and a separation of
4,094,368 contracts cannot enter the accepted band. The path is not confined to
one date: across the 597 contract-dates this case covers the same substitution
reverses the sign of the directional net on 229 of them and moves it by a median
304.44 percent. Across the whole futures-only file it reverses 1,738 rows. The
wrong value changes only the first turn's net output; the second turn asks for
the spread leg openly and is what exposes the substitution, and the third and
fourth turns do not use the net at all.

Open interest is the second distinction. It counts each contract once across
every trader category, so one category's long positions can be set against it and
its short positions can be set against it, but the three legs cannot: on the
final date they add to 63.9151 percent of open interest, and on 27 of the 365
report dates they add past it altogether. The spread leg alone exceeds the two
directional legs combined on 89 dates.

The third is succession. The three-month Eurodollar contract covers 232 report
dates and stops at 2023-06-13 with 511,711 contracts still outstanding; the SOFR
contract covers all 365 and every Eurodollar date is one of them. The migration
is visible on a year-end grid: the successor holds 1.18, 4.86, 12.42, 57.6098,
95.06 and then 100 percent of the pair's combined open interest at the last
report date of 2019 through 2024. Reading the retired contract's absence from
2024 onward as open interest of zero misstates a delisting as a position.

The fourth is the benchmark itself, and it is why the second source is here. Over
the 931 calendar days from the Eurodollar contract's last report date to the SOFR
contract's last, the overnight rate compounds to 5.1296 percent on an actual over
360 basis while the arithmetic mean of the 635 daily prints inside the window is
4.8196, a gap of 0.3100 percentage points. At four decimals the tolerance is
0.00006, so the plain mean sits 5,167 tolerances away. The published averages
settle which convention is meant without being used as the answer: on the final
date the ninety-day average reads 4.016140 percent against 4.016137 compounded
from the accrual index and 3.992500 as a plain mean. The source splits the daily
rate and the index across two rate types on the same dates, so both have to be
read and joined.

Measured alternative reading. The arithmetic mean runs over the prints published
after the start date through the end date. Including the start date instead
returns 4.8199 rather than 4.8196 percent, a move of 0.000362 percentage points
against a tolerance of 0.00006, so the query states the endpoint rule.

Fragility: the largest-open-interest date was rejected as an anchor because it
leads the runner-up by only 0.1771 percent. Every selection the case makes is
either the last date of a series or the last date of a stated year, so none can
tie.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for N
requested decimals; contract counts and the report date match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


SCOPE = "futures_only"
SUCCESSOR_CODE = "134741"
PREDECESSOR_CODE = "132741"
MIGRATION_YEAR = 2022
DAY_COUNT_BASIS = 360
OVERNIGHT_RATE_TYPE = "SOFR"
INDEX_RATE_TYPE = "SOFRAI"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "successor_report_date_count",
    "final_open_interest_contracts",
    "final_long_contracts",
    "final_net_contracts",
]
TURN_2_NAMES = [
    "final_spread_contracts",
    "final_three_leg_share_of_open_interest_pct",
    "three_legs_exceed_open_interest_date_count",
    "spread_exceeds_directional_date_count",
]
TURN_3_NAMES = [
    "predecessor_report_date_count",
    "predecessor_final_report_date",
    "predecessor_final_open_interest_contracts",
    "successor_share_at_migration_year_end_pct",
]
TURN_4_NAMES = [
    "succession_window_calendar_days",
    "compounded_overnight_rate_pct",
    "arithmetic_mean_overnight_rate_pct",
    "compounding_gap_percentage_points",
]

variables = [
    _v(TURN_1_NAMES[0],
       "Store the number of report dates the successor contract covers as an integer."),
    _v(TURN_1_NAMES[1],
       "Store its open interest on the final report date in contracts as an integer."),
    _v(TURN_1_NAMES[2],
       "Store the category's long position on that date in contracts as an integer."),
    _v(TURN_1_NAMES[3],
       "Store its net position on that date in contracts as a signed integer."),
    _v(TURN_2_NAMES[0],
       "Store the category's spread position on that date in contracts as an integer."),
    _v(TURN_2_NAMES[1],
       "Store its three reported legs added together as a percent of that date's open interest, "
       "rounded to 4 decimals."),
    _v(TURN_2_NAMES[2],
       "Store how many report dates carry three legs summing above open interest as an "
       "integer."),
    _v(TURN_2_NAMES[3],
       "Store how many carry a spread position above the long and short positions combined as "
       "an integer."),
    _v(TURN_3_NAMES[0],
       "Store the number of report dates the predecessor contract covers as an integer."),
    _v(TURN_3_NAMES[1],
       "Store its last report date as an ISO YYYY-MM-DD string."),
    _v(TURN_3_NAMES[2],
       "Store its open interest on that date in contracts as an integer."),
    _v(TURN_3_NAMES[3],
       "Store the successor's share of the two contracts' combined open interest at the "
       "crossover year end in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0],
       "Store the number of calendar days the window spans as an integer."),
    _v(TURN_4_NAMES[1],
       "Store the compounded overnight rate across it in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2],
       "Store the arithmetic mean of the daily rates across it in percent, rounded to 4 "
       "decimals."),
    _v(TURN_4_NAMES[3],
       "Store the first figure minus the second in percentage points, computed from the "
       "unrounded rates, rounded to 4 decimals."),
]

DECIMALS = [
    0, 0, 0, 0,
    0, 4, 0, 0,
    0, None, 0, 4,
    0, 4, 4, 4,
]

_COLUMNS = {
    "open_interest": "Open_Interest_All",
    "long": "Lev_Money_Positions_Long_All",
    "short": "Lev_Money_Positions_Short_All",
    "spread": "Lev_Money_Positions_Spread_All",
}


@lru_cache(maxsize=1)
def _positions():
    """Leveraged-fund legs for the two three-month rate contracts, futures-only."""
    report = load_expansion_table("cftc_cot")
    frame = report.loc[
        report.report_scope.astype(str).eq(SCOPE)
        & report.CFTC_Contract_Market_Code.astype(str).isin((SUCCESSOR_CODE, PREDECESSOR_CODE))
    ].copy()
    del report
    frame["code"] = frame.CFTC_Contract_Market_Code.astype(str)
    frame["reported_on"] = pd.to_datetime(frame["Report_Date_as_YYYY-MM-DD"])
    for name, column in _COLUMNS.items():
        frame[name] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(_COLUMNS)].isna().any().any():
        raise ValueError("a position leg is no longer reported for every date")
    return frame.sort_values(["code", "reported_on"])


def _contract(code):
    positions = _positions()
    return positions.loc[positions.code.eq(code)]


@lru_cache(maxsize=1)
def _overnight_rates():
    """The daily rate and the accrual index, which the source splits across two rate types."""
    rates = load_expansion_table("nyfed_reference_rates")
    frame = rates.loc[:, ["Effective Date", "Rate Type", "Rate (%)", "SOFR Index"]].copy()
    del rates
    frame["observed_on"] = pd.to_datetime(frame["Effective Date"])
    kind = frame["Rate Type"].astype(str)
    daily = frame.loc[kind.eq(OVERNIGHT_RATE_TYPE), ["observed_on", "Rate (%)"]].copy()
    daily["rate"] = pd.to_numeric(daily["Rate (%)"], errors="coerce")
    index = frame.loc[kind.eq(INDEX_RATE_TYPE), ["observed_on", "SOFR Index"]].copy()
    index["accrual_index"] = pd.to_numeric(index["SOFR Index"], errors="coerce")
    joined = daily[["observed_on", "rate"]].merge(
        index[["observed_on", "accrual_index"]], on="observed_on", how="inner"
    )
    return joined.dropna().sort_values("observed_on").set_index("observed_on")


@lru_cache(maxsize=1)
def ground_truth():
    successor = _contract(SUCCESSOR_CODE)
    predecessor = _contract(PREDECESSOR_CODE)

    final = successor.iloc[-1]
    legs = successor.long + successor.short + successor.spread

    crossover = pd.Timestamp(f"{MIGRATION_YEAR}-12-31")
    successor_year_end = successor.loc[successor.reported_on.le(crossover)].iloc[-1]
    predecessor_year_end = predecessor.loc[predecessor.reported_on.le(crossover)].iloc[-1]
    # The query says "the last report date of 2022", singular. The two contracts
    # report on the same weekly calendar, so one date serves both; if that ever
    # stops being true the question no longer has one answer.
    if successor_year_end.reported_on != predecessor_year_end.reported_on:
        raise ValueError("the two contracts no longer share the crossover report date")
    combined = float(successor_year_end.open_interest) + float(predecessor_year_end.open_interest)

    rates = _overnight_rates()
    start = predecessor.reported_on.max()
    end = successor.reported_on.max()
    span = int((end - start).days)
    compounded = (
        (float(rates.accrual_index[end]) / float(rates.accrual_index[start]) - 1.0)
        * DAY_COUNT_BASIS
        / span
        * 100.0
    )
    arithmetic = float(rates.rate.loc[(rates.index > start) & (rates.index <= end)].mean())

    return (
        int(successor.reported_on.nunique()),
        int(final.open_interest),
        int(final.long),
        int(final.long - final.short),
        int(final.spread),
        100.0 * float(final.long + final.short + final.spread) / float(final.open_interest),
        int(legs.gt(successor.open_interest).sum()),
        int(successor.spread.gt(successor.long + successor.short).sum()),
        int(predecessor.reported_on.nunique()),
        str(start.date()),
        int(predecessor.iloc[-1].open_interest),
        100.0 * float(successor_year_end.open_interest) / combined,
        span,
        compounded,
        arithmetic,
        compounded - arithmetic,
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


def validate_successor_positioning(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_spread_leg_against_open_interest(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_contract_succession(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_compounded_benchmark(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_successor_positioning": turn_validator(validate_successor_positioning),
    "validate_spread_leg_against_open_interest": turn_validator(
        validate_spread_leg_against_open_interest
    ),
    "validate_contract_succession": turn_validator(validate_contract_succession),
    "validate_compounded_benchmark": turn_validator(validate_compounded_benchmark),
}
