"""What a sector transacted and what its holdings are worth are two different series.

The Financial Accounts publish one instrument as several series that share nine
digits and differ only in a two-letter prefix. Corporate equities on the
nonfinancial corporate balance sheet carry LM for the market-value level, FU for
the quarter's own transactions, FA for those same transactions restated at a
seasonally adjusted annual rate, and FR for the revaluation of what was already
held. The change in the level is the sum of the last two, not any one of them:

    LM(t) - LM(t-1) = FU(t) + FR(t)

It closes on all 41 quarter-over-quarter changes the newest release supports,
with a largest absolute residual of 0.0000 million USD. Only four quarterly
series in that release carry an other-volume-change leg at all, so the two-term
form is the ordinary case rather than a simplification.

The trap is which series answers "how much did the sector transact", and it
fires in the first turn. The pinned answer sums FU to -1,050,755.0000 million
USD. Summing FA instead returns -4,203,020.0000 million, exactly four times as
much, because the annual-rate series scales one quarter of activity to a yearly
pace rather than accumulating four quarters of it. Reading the change in the
level as the transacted amount returns +1,689,557.0000 million and carries the
opposite sign. At four requested decimals the rounding-boundary tolerance is
0.00006 million USD against separations of 3,152,265.0000 and 2,740,312.0000, so
neither wrong figure can enter the accepted band. The wrong path changes the
summed-issuance output in the first turn; the second turn's counts and the third
turn's comparison are computed per quarter from the same legs and are reached
through the identity, which is what exposes the substitution. The fourth turn
names the ratio openly, after the first turn has been scored.

The financial point is that the sector was shrinking its equity while the market
value of that equity grew. Net issuance is negative in 33 of the 41 quarters --
buybacks and cash acquisitions exceeding placements -- and net issuance and the
market-value change carry opposite signs in 21 of them. The revaluation is the
larger term in 37. Anyone reading the balance-sheet line as a measure of what
corporations did would conclude they raised 1.69 trillion USD of equity over the
decade, when they retired 1.05 trillion.

The revaluation leg is not a plug. Expressed against the market value each
quarter opened with, it correlates 0.9936 with the value-weighted market return
compounded over that quarter's trading days, differs from it by a median 0.5186
percentage points and agrees in sign in 39 of 41 quarters. That is the check the
second source exists for; the accounts alone cannot supply it.

Fragility: the largest absolute revaluation, -484,583.0000 million on
2022-06-30, stands only 5.82 percent ahead of the runner-up at -457,935.0000
million on 2020-03-31, so that selection is a two-quarter race rather than a
wide margin. The frozen release fixes it.

Only the newest release is read, and that pin is load bearing rather than
decorative: every earlier release in the extract disagrees with it on between 16
and 32 of this instrument's observations, by as much as 650,435 million USD.
Differencing the same series across releases measures revision, which four cases
in this suite already ask about and which this one deliberately avoids.

Two further conventions are left out of the query because they change nothing.
The level appears in three or four statement tables at once and every copy
carries the same value, so which copy is read cannot move an answer. And no two
quarters share a revaluation magnitude, so the tie rule the query states has
nothing to break here.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for N
requested decimals; counts match exactly and the quarter end is compared as a
string. Summed money figures are stable to 0.0000000000 across sixteen
summation orders.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, load_ken_french_table
from core.validation import turn_validator, validate_ordered_outputs


SERIES_STEM = "103064103.Q"
LEVEL_PREFIX = "LM"
FLOW_PREFIX = "FU"
ANNUAL_RATE_PREFIX = "FA"
REVALUATION_PREFIX = "FR"
IDENTITY_TOLERANCE_USD_MILLIONS = 1.0


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "covered_quarter_count",
    "closing_market_value_usd_millions",
    "summed_net_issuance_usd_millions",
    "net_retirement_quarter_count",
]
TURN_2_NAMES = [
    "identity_closing_quarter_count",
    "maximum_absolute_residual_usd_millions",
    "opposite_sign_quarter_count",
    "revaluation_exceeds_issuance_quarter_count",
]
TURN_3_NAMES = [
    "matched_quarter_count",
    "revaluation_market_return_correlation",
    "median_absolute_gap_percentage_points",
    "sign_agreement_quarter_count",
]
TURN_4_NAMES = [
    "annual_rate_to_quarterly_ratio",
    "latest_quarter_annual_rate_usd_millions",
    "largest_revaluation_quarter_end",
    "largest_revaluation_usd_millions",
]

variables = [
    _v(TURN_1_NAMES[0],
       "Store the number of quarter-over-quarter changes the window supports as an integer."),
    _v(TURN_1_NAMES[1],
       "Store the market value at the final quarter end in USD millions at the published "
       "precision, rounded to 4 decimals."),
    _v(TURN_1_NAMES[2],
       "Store the summed net issuance across those quarters in USD millions, rounded to 4 "
       "decimals."),
    _v(TURN_1_NAMES[3],
       "Store how many of those quarters carry a negative net issuance as an integer."),
    _v(TURN_2_NAMES[0],
       "Store how many quarters satisfy the decomposition as an integer."),
    _v(TURN_2_NAMES[1],
       "Store the largest absolute residual across them in USD millions, rounded to 4 "
       "decimals."),
    _v(TURN_2_NAMES[2],
       "Store how many quarters carry a net issuance and a market-value change of opposite "
       "sign as an integer."),
    _v(TURN_2_NAMES[3],
       "Store how many carry a revaluation larger in absolute size than the net issuance as an "
       "integer."),
    _v(TURN_3_NAMES[0],
       "Store how many quarters match a compounded market return as an integer."),
    _v(TURN_3_NAMES[1],
       "Store the correlation between the two series across them, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2],
       "Store the median absolute difference between them in percentage points, rounded to 4 "
       "decimals."),
    _v(TURN_3_NAMES[3],
       "Store how many of those quarters carry the same sign on both, as an integer."),
    _v(TURN_4_NAMES[0],
       "Store the annual-rate figure divided by the quarterly one for the final quarter, "
       "rounded to 4 decimals."),
    _v(TURN_4_NAMES[1],
       "Store the annual-rate figure for the final quarter in USD millions at the published "
       "precision, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2],
       "Store that quarter's end date as an ISO YYYY-MM-DD string."),
    _v(TURN_4_NAMES[3],
       "Store its revaluation in USD millions at the published precision, rounded to 4 "
       "decimals."),
]

DECIMALS = [
    0, 4, 4, 0,
    0, 4, 0, 0,
    0, 4, 4, 0,
    4, 4, None, 4,
]


@lru_cache(maxsize=1)
def _series():
    """The four legs of one instrument, from the newest release only.

    Only the newest vintage is read. Comparing releases measures revision, which
    is a different question and is not what this case asks.
    """
    accounts = load_expansion_table("fed_z1_vintages")
    frame = accounts.loc[:, ["vintage", "series", "date", "value"]].copy()
    del accounts
    frame["series"] = frame.series.astype(str)
    frame = frame.loc[
        frame.vintage.astype(str).eq(frame.vintage.astype(str).max())
        & frame.series.str.slice(2).eq(SERIES_STEM)
    ].copy()
    frame["prefix"] = frame.series.str.slice(0, 2)
    frame["quarter_end"] = pd.to_datetime(frame.date)
    frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    legs = frame.pivot_table(
        index="quarter_end", columns="prefix", values="value", aggfunc="first"
    ).sort_index()
    missing = {LEVEL_PREFIX, FLOW_PREFIX, ANNUAL_RATE_PREFIX, REVALUATION_PREFIX} - set(legs.columns)
    if missing:
        raise ValueError(f"the instrument no longer publishes every leg: {sorted(missing)}")
    return legs


@lru_cache(maxsize=1)
def _changes():
    """One row per quarter-over-quarter change, with the decomposition attached."""
    legs = _series().copy()
    legs["opening"] = legs[LEVEL_PREFIX].shift(1)
    legs = legs.dropna(subset=["opening", FLOW_PREFIX, REVALUATION_PREFIX])
    legs["market_value_change"] = legs[LEVEL_PREFIX] - legs.opening
    legs["residual"] = legs.market_value_change - legs[FLOW_PREFIX] - legs[REVALUATION_PREFIX]
    legs["revaluation_pct"] = 100.0 * legs[REVALUATION_PREFIX] / legs.opening
    return legs


@lru_cache(maxsize=1)
def _quarterly_market_return():
    """Value-weighted market return including the risk-free rate, compounded by quarter."""
    factors = load_ken_french_table("daily_factors")
    frame = factors.loc[:, ["date", "mkt_rf_pct", "rf_pct"]].copy()
    del factors
    frame["observed_on"] = pd.to_datetime(frame.date)
    frame["total"] = (
        pd.to_numeric(frame.mkt_rf_pct, errors="coerce")
        + pd.to_numeric(frame.rf_pct, errors="coerce")
    ) / 100.0
    frame = frame.loc[frame.total.notna()]
    grouped = frame.groupby(frame.observed_on.dt.to_period("Q")).total
    return 100.0 * (grouped.apply(lambda daily: float(np.prod(1.0 + daily))) - 1.0)


@lru_cache(maxsize=1)
def ground_truth():
    changes = _changes()
    closes = changes.residual.abs().le(IDENTITY_TOLERANCE_USD_MILLIONS)
    opposite = np.sign(changes[FLOW_PREFIX]) * np.sign(changes.market_value_change) < 0

    returns = _quarterly_market_return()
    quarters = pd.PeriodIndex(changes.index, freq="Q")
    matched = changes.assign(market_return=quarters.map(returns)).dropna(subset=["market_return"])
    gap = matched.revaluation_pct - matched.market_return

    final = changes.index.max()
    # Largest absolute revaluation, earliest quarter on a tie.
    magnitudes = changes[REVALUATION_PREFIX].abs()
    largest_quarter = magnitudes.loc[magnitudes.eq(magnitudes.max())].index.min()
    largest = changes.loc[largest_quarter]

    return (
        int(len(changes)),
        float(changes.loc[final, LEVEL_PREFIX]),
        float(changes[FLOW_PREFIX].sum()),
        int(changes[FLOW_PREFIX].lt(0).sum()),
        int(closes.sum()),
        float(changes.residual.abs().max()),
        int(opposite.sum()),
        int((changes[REVALUATION_PREFIX].abs() > changes[FLOW_PREFIX].abs()).sum()),
        int(len(matched)),
        float(matched.revaluation_pct.corr(matched.market_return)),
        float(gap.abs().median()),
        int((np.sign(matched.revaluation_pct) == np.sign(matched.market_return)).sum()),
        float(changes.loc[final, ANNUAL_RATE_PREFIX] / changes.loc[final, FLOW_PREFIX]),
        float(changes.loc[final, ANNUAL_RATE_PREFIX]),
        str(largest_quarter.date()),
        float(largest[REVALUATION_PREFIX]),
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


def validate_issuance_window(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_decomposition_identity(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_revaluation_against_market(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_annual_rate_and_extreme(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_issuance_window": turn_validator(validate_issuance_window),
    "validate_decomposition_identity": turn_validator(validate_decomposition_identity),
    "validate_revaluation_against_market": turn_validator(validate_revaluation_against_market),
    "validate_annual_rate_and_extreme": turn_validator(validate_annual_rate_and_extreme),
}
