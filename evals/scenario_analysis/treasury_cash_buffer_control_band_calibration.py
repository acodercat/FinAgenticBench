"""A control band calibrated on the Treasury's cash, and what the calibration refuses.

Miller and Orr (1966) solve the cash manager's impulse-control problem: when the
balance wanders outside a band the manager transfers it back to a return point,
and the optimal band trades a fixed transfer cost against forgone interest. With
the lower limit at zero, divide three times the transfer cost times the variance
of the daily net flow by four times the daily holding rate and take the cube root:
that is the return point, and the upper limit is three times it. The model is
the reason this case exists: it is a closed-form intertemporal policy, and the
data decide every input to it.

The reservoir is the Treasury General Account. Across the 501 business days of
calendar 2024 and 2025 the daily net flow averages 208.1218 million dollars
against a sample standard deviation of 37,597.6367, so the drift is a
one-hundred-eightieth of a standard deviation and the zero-drift premise looks
unobjectionable at the unconditional level. At an effective federal funds rate of
3.6400 percent, a daily holding rate on a three-hundred-sixty-day basis and a
hypothetical transfer cost of one million dollars, the return point is 21,887.416
million dollars and the upper limit 65,662.248.

The account it is calibrated on never came near that band. Its mean closing
balance over the same window is 705,267.2974 million dollars, more than ten times
the upper limit. Inverting the model rather than discarding it gives the case its
point: the fixed cost per transfer that would make three times the return point
equal to the observed mean balance is 1,239.12 million dollars. A cash balance
sized by a transaction-cost tradeoff would require every transfer to cost about
1.24 billion dollars, which quantifies the distance between that rationale and
the prudential-liquidity rationale the balance is actually set by. The case
reports the inversion, not a recommendation.

The band is narrow against the flow that runs through it, and the fourth turn
measures that directly: on 38 of the 501 days a single day's net movement is
larger than the entire 65,662.248 million dollar distance between the two control
limits, and on 8 of them larger than twice it. A policy with this band would be
driven from one edge to the other inside a single day.

The maintained assumption fails in a second, measurable way. Overlapping rolling
sums give a five-day variance ratio of 0.6753 and a twenty-day ratio of 0.7230
against the 1.0000 that serial independence requires, so the flows mean-revert
rather than accumulate. Grouping by calendar day of the month, the first of the
month carries a mean net flow of -66,627.8571 million dollars across its 14
observations -- 1.77 daily standard deviations of predictable outflow, and 3.93
times the next most negative day, the third at -16,943.4444. The largest single
day inflow in the window is 184,784 million dollars on 2025-04-15. Flows this
seasonal are not the serially independent increments the policy is derived from.

The model turn is formula-free, as the other two cases in this capability are. It
names Miller-Orr, states the objective and the band's structure, and pins every
input the data decides -- the lower limit, the transfer cost, which dispersion,
which overnight rate, which day-count basis -- but withholds the closed form, so
the agent supplies the method rather than evaluating a supplied expression.

Difficulty is medium, not hard. The general rubric would read five pinned
conventions and a named model as multi-convention synthesis, but the
quantitative overlay is the more specific rule and requires a quantitative-hard
case to carry a substantive estimation, recursion, decomposition, root solve,
optimization or simulation. This case carries none: a mean, a sample standard
deviation, a cube root and one inversion. Promoting it on convention count alone
is what the guide warns against.

Measured alternative readings, all pinned in the query because all of them clear
the tolerance. Using the rate as an annual figure rather than a daily one returns
a return point of 3,076.7603 rather than 21,887.416, a factor of 0.1406. A
three-hundred-sixty-five-day basis returns 21,988.2811, 100.8651 away. Taking the
standard deviation of the balance level instead of the daily flow returns
61,205.6266, a factor of 2.7964. A population standard deviation instead of a
sample one returns 21,872.8438, 14.5722 away. Substituting the overnight financing
rate of 3.8700 percent for the federal funds rate returns 21,444.9299.

Window choice. Calendar 2024 and 2025 sits entirely inside the four-line
publication era and entirely outside the 2023 debt-limit drawdown, whose 22,892
million dollar trough no control band could produce; the in-window minimum is
260,807. The withdrawals line is labelled with a minus sign but stores positive
magnitudes, so the net flow subtracts it. The query takes the days carrying all
four lines and the module drops any that do not; on the current release that rule
is vacuous, since every one of the 501 days publishes all four, and a test asserts
the vacuity rather than leaving the clause unexercised.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for N
requested decimals; counts, the two balance levels taken from the file and the
date match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


WINDOW_START = "2024-01-01"
WINDOW_END = "2025-12-31"
CLOSING_LINE = "Treasury General Account (TGA) Closing Balance"
OPENING_LINE = "Treasury General Account (TGA) Opening Balance"
DEPOSITS_LINE = "Total TGA Deposits (Table II)"
WITHDRAWALS_LINE = "Total TGA Withdrawals (Table II) (-)"
HOLDING_RATE_TYPE = "EFFR"
DAY_COUNT_BASIS = 360
TRANSFER_COST = 1.0
LOWER_LIMIT = 0.0
SHORT_HORIZON = 5
LONG_HORIZON = 20


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "business_day_count",
    "mean_daily_net_flow_usd_millions",
    "net_flow_standard_deviation_usd_millions",
    "final_closing_balance_usd_millions",
]
TURN_2_NAMES = [
    "holding_cost_rate_pct",
    "control_band_return_point_usd_millions",
    "mean_closing_balance_usd_millions",
    "implied_transfer_cost_usd_millions",
]
TURN_3_NAMES = [
    "short_horizon_variance_ratio",
    "long_horizon_variance_ratio",
    "largest_outflow_calendar_day",
    "largest_outflow_calendar_day_mean_usd_millions",
]
TURN_4_NAMES = [
    "minimum_closing_balance_usd_millions",
    "minimum_closing_balance_date",
    "maximum_closing_balance_usd_millions",
    "days_single_flow_exceeds_band_count",
]

variables = [
    _v(TURN_1_NAMES[0],
       "Store the number of business days the window carries all four account lines as an "
       "integer."),
    _v(TURN_1_NAMES[1],
       "Store the mean daily net cash flow across them in millions of dollars, rounded to 4 "
       "decimals."),
    _v(TURN_1_NAMES[2],
       "Store the sample standard deviation of that daily net flow in millions of dollars, "
       "rounded to 4 decimals."),
    _v(TURN_1_NAMES[3],
       "Store the closing balance on the last of those days in millions of dollars as an "
       "integer."),
    _v(TURN_2_NAMES[0],
       "Store the holding cost rate the band uses, before the daily conversion, in percent, "
       "rounded to 4 decimals."),
    _v(TURN_2_NAMES[1],
       "Store the return point above the lower control limit in millions of dollars, rounded "
       "to 3 decimals."),
    _v(TURN_2_NAMES[2],
       "Store the mean closing balance across the window in millions of dollars, rounded to 4 "
       "decimals."),
    _v(TURN_2_NAMES[3],
       "Store the fixed transfer cost that places the optimal upper control limit at that mean "
       "closing balance, in millions of dollars, computed from the unrounded inputs, rounded "
       "to 2 decimals."),
    _v(TURN_3_NAMES[0],
       "Store the five-business-day variance ratio, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1],
       "Store the twenty-business-day variance ratio, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2],
       "Store the calendar day of the month carrying the most negative mean net flow as an "
       "integer."),
    _v(TURN_3_NAMES[3],
       "Store that day's mean net flow in millions of dollars, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0],
       "Store the smallest closing balance in the window in millions of dollars as an "
       "integer."),
    _v(TURN_4_NAMES[1],
       "Store the date it occurred as an ISO YYYY-MM-DD string."),
    _v(TURN_4_NAMES[2],
       "Store the largest closing balance in the window in millions of dollars as an "
       "integer."),
    _v(TURN_4_NAMES[3],
       "Store how many business days carry an absolute net cash flow larger than the whole "
       "distance from the lower control limit to the upper one as an integer."),
]

DECIMALS = [
    0, 4, 4, 0,
    4, 3, 4, 2,
    4, 4, 0, 4,
    0, None, 0, 0,
]


@lru_cache(maxsize=1)
def _account():
    """The four modern TGA lines, one row per business day of the window."""
    cash = load_expansion_table("treasury_dts_operating_cash")
    lines = (CLOSING_LINE, OPENING_LINE, DEPOSITS_LINE, WITHDRAWALS_LINE)
    frame = cash.loc[cash.account_type.astype(str).isin(lines)].copy()
    del cash
    frame["observed_on"] = pd.to_datetime(frame.record_date)
    frame["amount"] = pd.to_numeric(frame.today_amount_million_usd, errors="coerce")
    # One statement publishes each line once a day. If that ever stops holding, a
    # pivot would silently keep whichever row came first and the window would be
    # built from an arbitrary half of the duplicates.
    if frame.groupby(["observed_on", frame.account_type.astype(str)]).size().max() > 1:
        raise ValueError("a statement line is published more than once for one date")
    wide = frame.pivot_table(
        index="observed_on", columns=frame.account_type.astype(str), values="amount",
        aggfunc="first",
    ).sort_index()
    # The query takes the days on which all four lines are published, so a day
    # missing one is dropped rather than raised on.
    window = wide.loc[WINDOW_START:WINDOW_END, list(lines)].dropna()
    if window.empty:
        raise ValueError("no business day in the window carries all four lines")
    # The withdrawals line is labelled as a negative but stores positive magnitudes.
    if window[WITHDRAWALS_LINE].le(0).any():
        raise ValueError("the withdrawals line no longer stores positive magnitudes")
    window = window.assign(net_flow=window[DEPOSITS_LINE] - window[WITHDRAWALS_LINE])
    return window


def _holding_rate_pct():
    """The federal funds rate published for the window's last business day."""
    rates = load_expansion_table("nyfed_reference_rates")
    frame = rates.loc[rates["Rate Type"].astype(str).eq(HOLDING_RATE_TYPE),
                      ["Effective Date", "Rate (%)"]].copy()
    del rates
    frame["observed_on"] = pd.to_datetime(frame["Effective Date"])
    frame["rate"] = pd.to_numeric(frame["Rate (%)"], errors="coerce")
    final = _account().index.max()
    quoted = frame.loc[frame.observed_on.eq(final), "rate"].dropna()
    if len(quoted) != 1:
        raise ValueError("the holding cost rate is not published exactly once for that date")
    return float(quoted.iloc[0])


def _return_point(variance, daily_rate, transfer_cost=TRANSFER_COST):
    return LOWER_LIMIT + (3.0 * transfer_cost * variance / (4.0 * daily_rate)) ** (1.0 / 3.0)


def _variance_ratio(net_flow, horizon):
    daily = float(net_flow.var(ddof=1))
    rolled = float(net_flow.rolling(horizon).sum().dropna().var(ddof=1))
    return rolled / (horizon * daily)


@lru_cache(maxsize=1)
def ground_truth():
    account = _account()
    net_flow = account.net_flow
    closing = account[CLOSING_LINE]

    deviation = float(net_flow.std(ddof=1))
    rate_pct = _holding_rate_pct()
    daily_rate = rate_pct / 100.0 / DAY_COUNT_BASIS
    return_point = _return_point(deviation**2, daily_rate)

    mean_balance = float(closing.mean())
    # The same band, read backwards: the transfer cost that lifts the upper limit
    # to the balance the account actually held.
    implied_cost = (
        4.0 * daily_rate * ((mean_balance - LOWER_LIMIT) / 3.0) ** 3 / (3.0 * deviation**2)
    )

    by_day = net_flow.groupby(net_flow.index.day).mean()
    worst_day = int(by_day.idxmin())

    return (
        int(len(account)),
        float(net_flow.mean()),
        deviation,
        int(closing.iloc[-1]),
        rate_pct,
        return_point,
        mean_balance,
        implied_cost,
        _variance_ratio(net_flow, SHORT_HORIZON),
        _variance_ratio(net_flow, LONG_HORIZON),
        worst_day,
        float(by_day.loc[worst_day]),
        int(closing.min()),
        str(closing.idxmin().date()),
        int(closing.max()),
        int(net_flow.abs().gt(3.0 * (return_point - LOWER_LIMIT)).sum()),
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


def validate_flow_calibration(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_control_band(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_maintained_assumption(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_observed_behaviour(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_flow_calibration": turn_validator(validate_flow_calibration),
    "validate_control_band": turn_validator(validate_control_band),
    "validate_maintained_assumption": turn_validator(validate_maintained_assumption),
    "validate_observed_behaviour": turn_validator(validate_observed_behaviour),
}
