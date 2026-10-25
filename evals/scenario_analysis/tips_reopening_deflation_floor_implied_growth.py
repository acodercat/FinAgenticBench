"""Price a reopened TIPS under a stated hypothetical nominal-measure index model.

The issue-date adjusted auction price already reflects indexation. Settlement
accrual is converted from per 1,000 original face. Coupons follow expected
indexed principal; only final principal has an original-face floor. In the
stated lognormal model, arithmetic index growth fixes the coupon expectation
independently of volatility, while terminal floor value is nonlinear.

A wrong purchased-cost floor, double indexation of the adjusted price, or
substitution of log drift for arithmetic drift changes the price-matching root.
The calibrated parameter is model-implied, not an observed CPI forward.
"""

from datetime import date
from functools import lru_cache
from math import log, sqrt

import numpy as np
from cave_agent import Variable
from scipy.optimize import brentq
from scipy.stats import norm

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


CUSIP = "91282CNB3"
AUCTION_DATE = "2025-06-17"
SETTLEMENT = date(2025, 6, 30)
PAR_TENORS = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
PAR_COLUMNS = ["6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr"]
HALF_YEARS = np.arange(0.5, 5.01, 0.5)
BASE_GROWTH = 0.02
LOW_VOL = 0.03
HIGH_VOL = 0.05


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("dirty_invoice_per_100", "Store the June 30 dirty settlement invoice per USD 100 original face, in USD rounded to 6 decimals."),
    _v("settlement_index_ratio", "Store the issue-date index ratio as a dimensionless number, rounded to 5 decimals."),
    _v("next_coupon_date", "Store the next coupon date as an ISO YYYY-MM-DD string."),
    _v("maturity_discount_factor", "Store the nominal discount factor to maturity as a dimensionless number, rounded to 6 decimals."),
    _v("unfloored_price_per_100", "Store the no-principal-floor present value at 2% arithmetic index growth per USD 100 original face, in USD rounded to 4 decimals."),
    _v("fitted_growth_3pct", "Store the annual arithmetic index growth calibrated at 3% volatility, in percent rounded to 4 decimals."),
    _v("floor_value_3pct_per_100", "Store the terminal principal-floor incremental present value at 3% volatility per USD 100 original face, in USD rounded to 4 decimals."),
    _v("floor_probability_3pct", "Store the model probability that final indexed principal is below original face at 3% volatility, in percent rounded to 3 decimals."),
    _v("fixed_growth_price_5pct_per_100", "Store the full bond present value at 5% volatility with the previously fitted growth fixed, per USD 100 original face in USD rounded to 3 decimals."),
    _v("fixed_growth_floor_5pct_per_100", "Store the terminal principal-floor incremental present value at 5% volatility with the previously fitted growth fixed, per USD 100 original face in USD rounded to 4 decimals."),
    _v("refitted_growth_5pct", "Store the annual arithmetic index growth recalibrated at 5% volatility, in percent rounded to 4 decimals."),
    _v("refitted_floor_probability_5pct", "Store the model probability that final indexed principal is below original face after 5% volatility recalibration, in percent rounded to 3 decimals."),
]

TURN_NAMES = [
    ["dirty_invoice_per_100", "settlement_index_ratio", "next_coupon_date"],
    ["maturity_discount_factor", "unfloored_price_per_100"],
    ["fitted_growth_3pct", "floor_value_3pct_per_100", "floor_probability_3pct"],
    ["fixed_growth_price_5pct_per_100", "fixed_growth_floor_5pct_per_100"],
    ["refitted_growth_5pct", "refitted_floor_probability_5pct"],
]
DECIMALS = [6, 5, None, 6, 4, 4, 4, 3, 3, 4, 4, 3]


def _inputs():
    auctions = load_expansion_table("treasury_auctions")
    auction = auctions.loc[auctions.cusip.eq(CUSIP) & auctions.auction_date.eq(AUCTION_DATE)]
    if len(auction) != 1:
        raise ValueError("The reopening auction is not unique")
    a = auction.iloc[0]
    if a.reopening != "Yes" or a.inflation_index_security != "Yes":
        raise ValueError("Auction is not a TIPS reopening")
    if str(a.issue_date) != SETTLEMENT.isoformat() or str(a.maturity_date) != "2030-04-15":
        raise ValueError("Frozen auction dates changed")
    curve = load_expansion_table("treasury_yield_curve")
    curve = curve.loc[curve.Date.astype(str).eq(SETTLEMENT.isoformat())]
    if len(curve) != 1:
        raise ValueError("Nominal curve date is not unique")
    return a, curve.iloc[0]


def _cashflow_state(auction, curve):
    par = np.array([float(curve[column]) for column in PAR_COLUMNS]) / 100
    half_year_par = np.interp(HALF_YEARS, PAR_TENORS, par)
    discount_grid = []
    for annual_par in half_year_par:
        semiannual_coupon = annual_par / 2
        discount_grid.append(
            (1 - semiannual_coupon * sum(discount_grid)) / (1 + semiannual_coupon)
        )
    grid = np.r_[0.0, HALF_YEARS]
    grid_df = np.r_[1.0, discount_grid]
    maturity = date.fromisoformat(str(auction.maturity_date))
    coupon_dates = [
        date(year, month, 15)
        for year in range(2025, 2031)
        for month in (4, 10)
        if SETTLEMENT < date(year, month, 15) <= maturity
    ]
    if coupon_dates[0].isoformat() != str(auction.first_int_payment_date):
        raise ValueError("Published first coupon does not match schedule")
    times = np.array([(day - SETTLEMENT).days / 365 for day in coupon_dates])
    cashflow_df = np.exp(np.interp(times, grid, np.log(grid_df)))
    return times, cashflow_df


def _values(growth, volatility, ratio, coupon_pct, times, discount):
    expected_ratio = ratio * np.exp(growth * times)
    coupon_pv = float(coupon_pct / 2 * np.dot(discount, expected_ratio))
    unfloored = coupon_pv + 100 * discount[-1] * expected_ratio[-1]
    terminal_t = times[-1]
    z = (log(1 / ratio) - (growth - volatility * volatility / 2) * terminal_t) / (
        volatility * sqrt(terminal_t)
    )
    probability = norm.cdf(z)
    floor_expectation = probability - expected_ratio[-1] * norm.cdf(
        z - volatility * sqrt(terminal_t)
    )
    floor_pv = 100 * discount[-1] * floor_expectation
    return float(unfloored), float(floor_pv), float(probability)


def analysis():
    a, curve = _inputs()
    times, discount = _cashflow_state(a, curve)
    ratio = float(a.index_ratio_on_issue_date)
    coupon = float(a.int_rate)
    invoice = float(a.adj_price) + float(a.adj_accrued_int_per1000) / 10

    def price(growth, volatility):
        base, floor, _ = _values(growth, volatility, ratio, coupon, times, discount)
        return base + floor

    def fit(volatility):
        residual = lambda growth: price(growth, volatility) - invoice
        if residual(-0.10) >= 0 or residual(0.15) <= 0:
            raise ValueError("Stated domain does not bracket unique growth root")
        root = brentq(residual, -0.10, 0.15, xtol=1e-14)
        if abs(residual(root)) > 1e-9:
            raise ValueError("Price calibration failed")
        return root

    initial_base, _, _ = _values(BASE_GROWTH, LOW_VOL, ratio, coupon, times, discount)
    first_growth = fit(LOW_VOL)
    _, first_floor, first_probability = _values(
        first_growth, LOW_VOL, ratio, coupon, times, discount
    )
    fixed_base, fixed_floor, _ = _values(
        first_growth, HIGH_VOL, ratio, coupon, times, discount
    )
    second_growth = fit(HIGH_VOL)
    _, _, second_probability = _values(
        second_growth, HIGH_VOL, ratio, coupon, times, discount
    )
    return {
        "dirty_invoice_per_100": invoice,
        "settlement_index_ratio": ratio,
        "next_coupon_date": str(a.first_int_payment_date),
        "maturity_discount_factor": float(discount[-1]),
        "unfloored_price_per_100": initial_base,
        "fitted_growth_3pct": 100 * first_growth,
        "floor_value_3pct_per_100": first_floor,
        "floor_probability_3pct": 100 * first_probability,
        "fixed_growth_price_5pct_per_100": fixed_base + fixed_floor,
        "fixed_growth_floor_5pct_per_100": fixed_floor,
        "refitted_growth_5pct": 100 * second_growth,
        "refitted_floor_probability_5pct": 100 * second_probability,
    }


@lru_cache(maxsize=1)
def ground_truth():
    result = analysis()
    return tuple(result[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    precision = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[n] for n in names], [truth[n] for n in names],
        [precision[n] for n in names],
    )


def validate(outputs):
    return _validate_subset(outputs, [v.name for v in variables])


validators = {
    f"validate_turn_{i}": turn_validator(
        lambda outputs, names=names: _validate_subset(outputs, names)
    )
    for i, names in enumerate(TURN_NAMES, start=1)
}
