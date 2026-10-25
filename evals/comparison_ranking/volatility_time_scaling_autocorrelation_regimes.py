"""Square-root-of-time scaling is exact only for serially uncorrelated returns.

Annualising a daily standard deviation by the square root of 252 assumes that
one day's return carries no information about the next. Where daily returns
are positively autocorrelated, a multi-day move is larger than independent
days would produce and the scaled figure understates annual risk; where they
are negatively autocorrelated, the scaled figure overstates it. The market
portfolio shows both, in different eras.

From 1963 through 1999 the lag-one autocorrelation of the daily market return
is 0.1681, and the annualised volatility is 0.1319 scaled up from daily
returns against 0.1516 scaled up from compounded calendar-month returns. All
ten industry portfolios show the same ordering, the utilities portfolio most
of all at a ratio of 1.4213 between the monthly-scaled and daily-scaled
figures. From 2000 through 2025 the autocorrelation is -0.0834 and the
ordering reverses, 0.1966 against 0.1567; only one industry keeps a ratio
above one, and the utilities portfolio now sits furthest below it at 0.7766.

The distortion passes straight into a Sharpe ratio, in the opposite direction
to the volatility bias. Annualised from daily excess returns the market's
ratio is 0.5192 in the earlier window against 0.4699 from monthly excess
returns, and 0.4088 against 0.4675 in the later one.

Seven conventions the query states because each moves an output:

1. The market return is the factor table's excess return plus its risk-free
   return, both converted from percent. The industry portfolios are total
   returns and need no such step.
2. Returns are simple returns throughout. Log returns are a common alternative
   for volatility work and move every market output here: 0.1679, 0.1326 and
   0.1522 in the earlier window against the reported 0.1681, 0.1319 and
   0.1516, and -0.0818, 0.1969 and 0.1578 in the later one against -0.0834,
   0.1966 and 0.1567. The square root of 252 is likewise stated rather than
   the window's own average count of trading days a year, which would read
   0.1964 for the later daily-scaled figure.
3. Standard deviations are sample standard deviations with an n - 1 divisor.
   With a divisor of n the monthly-scaled market figure reads 0.1514 rather
   than 0.1516 in the earlier window and 0.1564 rather than 0.1567 in the
   later one.
4. A monthly return is the compounded product of the daily returns inside the
   calendar month, not their sum.
5. The autocorrelation is the Pearson correlation between each day's return
   and the previous trading day's return inside the window. The textbook
   estimator with one full-sample mean and a full-sample denominator differs
   by under 1e-5 here and reads the same at four decimals; the rule is stated
   rather than relied on.
6. The daily excess return for the Sharpe ratio is the factor table's market
   column as it stands, which is already net of the risk-free rate. Taking the
   risk-free rate off it a second time gives 0.0631 in the earlier window and
   0.3125 in the later one.
7. The monthly excess return is the compounded market total return in the
   month less the compounded risk-free return in the month. Compounding the
   daily excess returns directly instead gives 0.4713 and 0.4680.

Near one the ratio is not a clean reading of serial correlation. The later
window's one industry above one, at 1.0199, and the earlier window's lowest,
at 1.0038, both fall below one, to 0.9875 and 0.9928, when monthly returns are
summed rather than compounded: compounding within the month, not serial
correlation, lifts them over one. The claims therefore tie serial correlation
to the ratios furthest from one, where it holds in both windows.

Industry rankings carry a stated tie rule, by column label ascending, which
does not currently bind: the leading ratio is clear of the runner-up by at
least 0.005 in each ranking, and the case test asserts that no two ratios
tie.
"""

from functools import lru_cache
from math import sqrt

import pandas as pd
from cave_agent import Variable

from core.data import load_ken_french_table
from core.validation import turn_validator, validate_ordered_outputs


EARLY = ("1963-01-01", "1999-12-31")
LATE = ("2000-01-01", "2025-12-31")
TRADING_DAYS = 252
MONTHS = 12

OUTPUTS = (
    ("early_market_autocorrelation", 4, "the earlier window's lag-one autocorrelation of daily market returns"),
    ("early_market_daily_scaled_volatility", 4, "the earlier window's market volatility scaled from daily returns, as a decimal fraction"),
    ("early_market_monthly_scaled_volatility", 4, "the earlier window's market volatility scaled from monthly returns, as a decimal fraction"),
    ("late_market_autocorrelation", 4, "the later window's lag-one autocorrelation of daily market returns"),
    ("late_market_daily_scaled_volatility", 4, "the later window's market volatility scaled from daily returns, as a decimal fraction"),
    ("late_market_monthly_scaled_volatility", 4, "the later window's market volatility scaled from monthly returns, as a decimal fraction"),
    ("early_largest_ratio_industry", None, "the column label of the earlier window's industry with the largest ratio, verbatim as it appears in the industry table provided"),
    ("early_largest_ratio", 4, "that industry's ratio of monthly-scaled to daily-scaled volatility"),
    ("late_smallest_ratio_industry", None, "the column label of the later window's industry with the smallest ratio, verbatim as it appears in the industry table provided"),
    ("late_smallest_ratio", 4, "that industry's ratio of monthly-scaled to daily-scaled volatility"),
    ("late_ratio_above_one_count", 0, "how many industries have a ratio above one in the later window, as an integer"),
    ("early_daily_sharpe", 4, "the earlier window's market Sharpe ratio annualised from daily excess returns"),
    ("early_monthly_sharpe", 4, "the earlier window's market Sharpe ratio annualised from monthly excess returns"),
    ("late_daily_sharpe", 4, "the later window's market Sharpe ratio annualised from daily excess returns"),
    ("late_monthly_sharpe", 4, "the later window's market Sharpe ratio annualised from monthly excess returns"),
)


def _description(precision, text):
    if precision is None:
        return f"Store {text}."
    if precision == 0:
        return f"Store {text}."
    return f"Store {text}, rounded to {precision} decimals from unrounded source rows."


variables = [Variable(name, None, _description(p, text)) for name, p, text in OUTPUTS]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:3]),
    tuple(n for n, _, _ in OUTPUTS[3:6]),
    tuple(n for n, _, _ in OUTPUTS[6:11]),
    tuple(n for n, _, _ in OUTPUTS[11:]),
)


@lru_cache(maxsize=1)
def _daily():
    """Industry and factor rows on the same trading dates, returns in decimals."""
    industries = load_ken_french_table("daily_industry_returns")
    factors = load_ken_french_table("daily_factors")
    frame = industries.merge(factors, on="date", how="inner", validate="one_to_one")
    if len(frame) != len(industries) or len(frame) != len(factors):
        raise ValueError("industry and factor tables do not share every trading date")
    frame["date"] = frame.date.astype(str)
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["month"] = frame.date.str[:7]
    frame["market"] = (frame.mkt_rf_pct + frame.rf_pct) / 100
    frame["market_excess"] = frame.mkt_rf_pct / 100
    frame["risk_free"] = frame.rf_pct / 100
    return frame


def industry_columns():
    return [c for c in load_ken_french_table("daily_industry_returns").columns if c != "date"]


def _window(bounds):
    frame = _daily()
    window = frame.loc[frame.date.between(*bounds)].reset_index(drop=True)
    if window.empty:
        raise ValueError(f"no trading day falls in {bounds}")
    return window


def _monthly(window, returns):
    return (1 + returns).groupby(window.month).prod() - 1


def _scaled(window, returns):
    daily = returns.std(ddof=1) * sqrt(TRADING_DAYS)
    monthly = _monthly(window, returns).std(ddof=1) * sqrt(MONTHS)
    return daily, monthly


def _ratios(window):
    ratios = {}
    for column in industry_columns():
        daily, monthly = _scaled(window, window[column] / 100)
        ratios[column] = monthly / daily
    return pd.Series(ratios)


def _sharpe(window):
    excess = window.market_excess
    daily = excess.mean() * TRADING_DAYS / (excess.std(ddof=1) * sqrt(TRADING_DAYS))
    monthly_excess = _monthly(window, window.market) - _monthly(window, window.risk_free)
    monthly = monthly_excess.mean() * MONTHS / (monthly_excess.std(ddof=1) * sqrt(MONTHS))
    return daily, monthly


@lru_cache(maxsize=1)
def analysis():
    early, late = _window(EARLY), _window(LATE)
    answers = {}
    for label, window in (("early", early), ("late", late)):
        daily, monthly = _scaled(window, window.market)
        answers[f"{label}_market_autocorrelation"] = float(window.market.autocorr(1))
        answers[f"{label}_market_daily_scaled_volatility"] = float(daily)
        answers[f"{label}_market_monthly_scaled_volatility"] = float(monthly)

    # Rank by ratio, breaking a tie by column label ascending.
    early_ratios = _ratios(early).rename_axis("label").reset_index(name="ratio")
    largest = early_ratios.sort_values(["ratio", "label"], ascending=[False, True]).iloc[0]
    late_ratios = _ratios(late).rename_axis("label").reset_index(name="ratio")
    smallest = late_ratios.sort_values(["ratio", "label"], ascending=[True, True]).iloc[0]
    answers["early_largest_ratio_industry"] = str(largest.label)
    answers["early_largest_ratio"] = float(largest.ratio)
    answers["late_smallest_ratio_industry"] = str(smallest.label)
    answers["late_smallest_ratio"] = float(smallest.ratio)
    answers["late_ratio_above_one_count"] = int(late_ratios.ratio.gt(1).sum())

    for label, window in (("early", early), ("late", late)):
        daily, monthly = _sharpe(window)
        answers[f"{label}_daily_sharpe"] = float(daily)
        answers[f"{label}_monthly_sharpe"] = float(monthly)
    return answers


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


def validate_turn_1(outputs):
    return _check(outputs, TURNS[0])


def validate_turn_2(outputs):
    return _check(outputs, TURNS[1])


def validate_turn_3(outputs):
    return _check(outputs, TURNS[2])


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
