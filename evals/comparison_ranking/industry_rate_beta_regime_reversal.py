"""When a yield rise stopped being good news, and which industries never agreed.

Campbell, Pflueger and Viceira read the sign of the stock-bond correlation as a
statement about which shocks dominate. Demand shocks move activity and inflation
the same way, so stocks and bond prices move apart; supply and policy shocks move
them together. The sign is therefore not a portfolio constant, and the window this
case covers contains a change in it.

Correlations are scale free, so the return on a ten-year position is taken as the
negative of the day's yield change and no duration assumption is needed or made.
Betas are not scale free and are stated per percentage point of yield change.

All three sources report on 1,747 days and 1,746 of those carry a yield change,
which is the population every figure below rests on; the query asks for the
second count, not the first. Across those days the yearly correlation
runs -0.5301, -0.4693, -0.1072, then 0.1798, 0.0668, 0.0665 and -0.1396. The first
calendar year in which it is positive is 2022. Before it the correlation is
-0.3864 over 750 days; from it, 0.0731 over 996.

The cross-section is the second half of the case. Over the whole window utilities
are the only industry whose return falls when the ten-year yield rises, at -0.4645
per percentage point, ahead of the next lowest by 0.8644; energy is the highest at
8.2132, ahead of the next by 4.1802. Split by regime, every one of the ten
industries has a lower beta in the later one: the smallest early beta is 8.4159
and nine of the ten late betas are negative, with energy falling the furthest at
-19.2693, ahead of the next largest fall by 2.0140. The fall is stored as the later
slope minus the earlier one, so it is negative; reported as a magnitude it would read
19.2693.

The last turn asks what the univariate beta was measuring. Adding the market
return to the regression flips the sign of the rate beta for six of the ten
industries -- health, non-durables, shops, telecoms, technology and durables all
move from positive to negative -- while the rank ordering survives at a Spearman
0.8303 and the two extremes do not move: utilities stay lowest at -2.0250 and
energy highest at 6.0011. The univariate figure is largely each industry's market
beta multiplied by the market's own rate sensitivity; the controlled figure is
what is left.

Measured alternative readings, all pinned in the query because all of them clear
the tolerance. Using the excess market return rather than the total return returns
-0.3866 and 0.0728 rather than -0.3864 and 0.0731. Differencing the yield along
forward-filled calendar days rather than consecutive paired observations returns
-0.3859 and 0.0732. Reading the two-year yield returns -0.3239 and -0.0051, so the
later correlation never turns positive at all; the five-year returns 0.0438 and the
thirty-year 0.0748. The flip year is 2022 under every one of those readings, which
is why it is derived rather than stated.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for N
requested decimals; counts, the year and the industry labels match exactly.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, load_ken_french_table
from core.validation import turn_validator, validate_ordered_outputs


TENOR = "10 Yr"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "paired_trading_day_count",
    "first_positive_correlation_year",
    "early_regime_correlation",
    "late_regime_correlation",
]
TURN_2_NAMES = [
    "most_negative_rate_beta_industry",
    "most_negative_rate_beta",
    "most_positive_rate_beta_industry",
    "most_positive_rate_beta",
]
TURN_3_NAMES = [
    "late_regime_negative_beta_count",
    "largest_beta_decline_industry",
    "largest_beta_decline",
    "early_regime_smallest_beta",
]
TURN_4_NAMES = [
    "sign_change_industry_count",
    "controlled_most_negative_rate_beta",
    "controlled_most_positive_rate_beta",
    "beta_ordering_rank_correlation",
]

variables = [
    _v(TURN_1_NAMES[0],
       "Store how many days carry a yield change under those rules as an integer."),
    _v(TURN_1_NAMES[1],
       "Store the first calendar year whose correlation is positive as a four-digit integer."),
    _v(TURN_1_NAMES[2],
       "Store the correlation across the days before that year, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3],
       "Store the correlation across that year and the days after it, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0],
       "Store the source column label of the lowest-sensitivity industry verbatim."),
    _v(TURN_2_NAMES[1],
       "Store its sensitivity in percent per percentage point, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2],
       "Store the source column label of the highest-sensitivity industry verbatim."),
    _v(TURN_2_NAMES[3],
       "Store its sensitivity in percent per percentage point, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0],
       "Store how many industries carry a negative later-regime sensitivity as an integer."),
    _v(TURN_3_NAMES[1],
       "Store the source column label of the industry whose sensitivity falls the most "
       "verbatim."),
    _v(TURN_3_NAMES[2],
       "Store that fall as the later-regime slope minus the earlier-regime slope, signed so "
       "that a fall is negative, in percent per percentage point, computed from the unrounded "
       "regime figures, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3],
       "Store the smallest earlier-regime sensitivity across the ten industries, rounded to 4 "
       "decimals."),
    _v(TURN_4_NAMES[0],
       "Store how many industries change the sign of their sensitivity once the market return "
       "joins the regression as an integer."),
    _v(TURN_4_NAMES[1],
       "Store the lowest market-controlled sensitivity, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2],
       "Store the highest market-controlled sensitivity, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3],
       "Store the Spearman rank correlation between the two sets of ten sensitivities, rounded "
       "to 4 decimals."),
]

DECIMALS = [
    0, 0, 4, 4,
    None, 4, None, 4,
    0, None, 4, 4,
    0, 4, 4, 4,
]


@lru_cache(maxsize=1)
def _panel():
    """One row per day on which the curve, the industries and the factors all report."""
    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["observed_on"] = pd.to_datetime(curve.Date)
    yields = (
        pd.to_numeric(curve[TENOR], errors="coerce")
        .set_axis(curve.observed_on)
        .dropna()
        .sort_index()
    )
    del curve

    industries = load_ken_french_table("daily_industry_returns").copy()
    industries["observed_on"] = pd.to_datetime(industries.date)
    industries = industries.set_index("observed_on").sort_index()
    columns = [name for name in industries.columns if name.endswith("_pct")]

    factors = load_ken_french_table("daily_factors").copy()
    factors["observed_on"] = pd.to_datetime(factors.date)
    factors = factors.set_index("observed_on").sort_index()
    # The total market return, not the excess return: the correlation this case
    # reads is between holding the market and holding a bond, both unlevered.
    market = (
        pd.to_numeric(factors.mkt_rf_pct, errors="coerce")
        + pd.to_numeric(factors.rf_pct, errors="coerce")
    ).rename("market")

    panel = (
        industries[columns]
        .join(yields.rename("tenor_yield"), how="inner")
        .join(market, how="inner")
        .dropna()
    )
    # The yield change is taken between consecutive rows of this panel, so a day
    # that any source skips is never bridged by carrying a stale yield across it.
    panel["yield_change"] = panel.tenor_yield.diff()
    panel = panel.dropna(subset=["yield_change"])
    if panel.empty:
        raise ValueError("no day is reported by all three sources")
    return panel, columns


def _slope(frame, column):
    """Sensitivity of an industry return to the yield change, no other regressor."""
    change = frame.yield_change
    return float(np.cov(frame[column], change, ddof=1)[0, 1] / change.var(ddof=1))


def _partial_slope(frame, column):
    """The same sensitivity with the market return alongside it, in closed form."""
    returns, change, market = frame[column], frame.yield_change, frame.market
    var_change = change.var(ddof=1)
    var_market = market.var(ddof=1)
    cov_change_market = float(np.cov(change, market, ddof=1)[0, 1])
    numerator = (
        float(np.cov(returns, change, ddof=1)[0, 1]) * var_market
        - float(np.cov(returns, market, ddof=1)[0, 1]) * cov_change_market
    )
    denominator = var_change * var_market - cov_change_market**2
    return float(numerator / denominator)


def _correlation(frame):
    return float(np.corrcoef(frame.market, -frame.yield_change)[0, 1])


@lru_cache(maxsize=1)
def ground_truth():
    panel, columns = _panel()

    yearly = {
        int(year): _correlation(group)
        for year, group in panel.groupby(panel.index.year)
    }
    positive = sorted(year for year, value in yearly.items() if value > 0)
    if not positive:
        raise ValueError("no calendar year carries a positive correlation")
    flip_year = positive[0]

    early = panel.loc[panel.index.year < flip_year]
    late = panel.loc[panel.index.year >= flip_year]
    if early.empty or late.empty:
        raise ValueError("the flip year leaves one regime with no days")

    whole = {name: _slope(panel, name) for name in columns}
    early_betas = {name: _slope(early, name) for name in columns}
    late_betas = {name: _slope(late, name) for name in columns}
    controlled = {name: _partial_slope(panel, name) for name in columns}
    declines = {name: late_betas[name] - early_betas[name] for name in columns}

    ranked = sorted(whole, key=lambda name: whole[name])
    falls = sorted(declines, key=lambda name: declines[name])
    controlled_ranked = sorted(controlled, key=lambda name: controlled[name])

    ordering = float(
        pd.Series([whole[name] for name in columns]).corr(
            pd.Series([controlled[name] for name in columns]), method="spearman"
        )
    )

    return (
        int(len(panel)),
        int(flip_year),
        _correlation(early),
        _correlation(late),
        str(ranked[0]),
        whole[ranked[0]],
        str(ranked[-1]),
        whole[ranked[-1]],
        int(sum(1 for name in columns if late_betas[name] < 0)),
        str(falls[0]),
        declines[falls[0]],
        min(early_betas.values()),
        int(sum(1 for name in columns if np.sign(whole[name]) != np.sign(controlled[name]))),
        controlled[controlled_ranked[0]],
        controlled[controlled_ranked[-1]],
        ordering,
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


def validate_regime_reversal(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_industry_cross_section(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_regime_shift(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_market_control(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_regime_reversal": turn_validator(validate_regime_reversal),
    "validate_industry_cross_section": turn_validator(validate_industry_cross_section),
    "validate_regime_shift": turn_validator(validate_regime_shift),
    "validate_market_control": turn_validator(validate_market_control),
}
