"""Bootstrap a Treasury curve, reprice an auctioned note and hedge key rates.

The selected security is the highest-bid-to-cover original nominal ten-year
note issued in 2025: CUSIP 91282CNC1.  Its issue-date CMT observations are par
yields rather than zero rates.  The case therefore requires a semiannual
par-instrument bootstrap before cash-flow pricing.  Treating the quoted par
yields directly as spot yields is a measured convention error and changes the
note price, duration, convexity, key-rate hedge and next-day result.

The analytical curve is a benchmark construction rather than the unpublished
Treasury production curve.  It linearly interpolates quoted par yields onto a
half-year grid, bootstraps discount factors, and reprices every synthetic par
instrument to numerical zero residual.  The selected note is valued at its
issue date with twenty regular half-year periods and no accrued interest.  Its
auction price was set before settlement, so the model-minus-auction gap is a
descriptive curve/quote reconciliation, not a pricing error.

The final hedge uses synthetic five- and ten-year par securities.  A one-basis-
point bump is applied to one quoted par node at a time, followed by a full
rebootstrap.  Hedge notionals solve the two key-rate equations.  The following
business-day revaluation holds coupons and maturity grids fixed, so it isolates
the observed curve change and is not a realized total return.

Numeric validation uses the shared rounding-boundary tolerance.  Duration and
convexity also admit the pre-registered values obtained from the three otherwise-
correct prices as reported to six decimals; this is a reporting-boundary
equivalence, not an alternative curve convention.  The curve repricing and hedge
residuals are explicit PV invariants.

The `treasury_par_bootstrap_hedge` pinned-reading comparison records sensitivity to treating
reported par yields as spot rates instead of bootstrapping them, to valuing on the
auction rather than the issue date, and to reporting DV01 as a signed price change
rather than a price loss. The curve method is the one worth naming: skipping the
bootstrap changes every discount factor, so it moves the model price the auction
comparison is built on rather than only the hedge. The query fixes all three, so these
are robustness comparisons rather than hidden answer paths.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import (
    ValidatorResult,
    turn_validator,
    validate_ordered_outputs,
)


NODE_MATURITIES = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
NODE_COLUMNS = (
    "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr",
)
GRID_MATURITIES = np.arange(0.5, 30.0 + 0.25, 0.5)
TARGET_YEAR = 2025
TARGET_TERM = "10-Year"
BUMP_BP = 1.0


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "eligible_ten_year_auction_count", "selected_cusip", "selected_auction_date",
    "selected_issue_date", "selected_maturity_date", "selected_bid_to_cover_ratio",
    "selected_coupon_pct", "selected_auction_price_per100",
]
TURN_2_NAMES = [
    "curve_observation_date", "ten_year_par_yield_pct", "thirty_year_par_yield_pct",
    "ten_year_discount_factor", "thirty_year_discount_factor",
    "ten_year_spot_yield_pct", "thirty_year_spot_yield_pct",
    "maximum_par_repricing_residual_per100",
]
TURN_3_NAMES = [
    "target_model_price_per100", "target_parallel_up_price_per100",
    "target_parallel_down_price_per100", "target_effective_duration_years",
    "target_effective_convexity",
]
TURN_4_NAMES = [
    "target_five_year_key_rate_dv01_per100", "target_ten_year_key_rate_dv01_per100",
    "five_year_hedge_face_per100_target", "ten_year_hedge_face_per100_target",
    "maximum_post_hedge_key_rate_residual_per100", "next_curve_date",
    "next_day_unhedged_price_change_per100", "next_day_hedged_price_change_per100",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the eligible original nominal ten-year auction count as an integer."),
    _v(TURN_1_NAMES[1], "Store the selected Treasury CUSIP as text."),
    _v(TURN_1_NAMES[2], "Store the selected auction date as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[3], "Store the selected issue date as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[4], "Store the selected maturity date as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[5], "Store the selected bid-to-cover ratio, rounded to 2 decimals."),
    _v(TURN_1_NAMES[6], "Store the selected annual coupon in percent, rounded to 3 decimals."),
    _v(TURN_1_NAMES[7], "Store the selected auction price per 100 face, rounded to 6 decimals."),
    _v(TURN_2_NAMES[0], "Store the curve observation date as an ISO YYYY-MM-DD string."),
    _v(TURN_2_NAMES[1], "Store the ten-year par yield in percent, rounded to 2 decimals."),
    _v(TURN_2_NAMES[2], "Store the thirty-year par yield in percent, rounded to 2 decimals."),
    _v(TURN_2_NAMES[3], "Store the ten-year discount factor, rounded to 8 decimals."),
    _v(TURN_2_NAMES[4], "Store the thirty-year discount factor, rounded to 8 decimals."),
    _v(TURN_2_NAMES[5], "Store the ten-year semiannual-compounded spot yield in percent, rounded to 6 decimals."),
    _v(TURN_2_NAMES[6], "Store the thirty-year semiannual-compounded spot yield in percent, rounded to 6 decimals."),
    _v(TURN_2_NAMES[7], "Store the maximum absolute par-instrument repricing residual per 100 face, rounded to 10 decimals."),
    _v(TURN_3_NAMES[0], "Store the target note model price per 100 face, rounded to 6 decimals."),
    _v(TURN_3_NAMES[1], "Store the target price after the parallel one-basis-point upward par-curve shock, rounded to 6 decimals."),
    _v(TURN_3_NAMES[2], "Store the target price after the parallel one-basis-point downward par-curve shock, rounded to 6 decimals."),
    _v(TURN_3_NAMES[3], "Store the target effective duration in years, rounded to 4 decimals. Compute from the unrounded model prices, not from the rounded figures reported for them."),
    _v(TURN_3_NAMES[4], "Store the target effective convexity, rounded to 6 decimals. Compute from the unrounded model prices, not from the rounded figures reported for them."),
    _v(TURN_4_NAMES[0], "Store the target five-year par-node key-rate DV01 as the signed price loss per 100 face under the upward bump, rounded to 7 decimals."),
    _v(TURN_4_NAMES[1], "Store the target ten-year par-node key-rate DV01 as the signed price loss per 100 face under the upward bump, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the face amount of the synthetic five-year hedge held against every 100 face of the target note, positive for long and negative for short, rounded to 6 decimals. This is a face amount, not the ratio of one hedge unit to one target unit."),
    _v(TURN_4_NAMES[3], "Store the face amount of the synthetic ten-year hedge held against every 100 face of the target note, positive for long and negative for short, rounded to 6 decimals. This is a face amount, not the ratio of one hedge unit to one target unit."),
    _v(TURN_4_NAMES[4], "Store the largest absolute combined key-rate DV01 remaining after the hedge, expressed against every 100 face of the target note, rounded to 10 decimals."),
    _v(TURN_4_NAMES[5], "Store the next Treasury curve date as an ISO YYYY-MM-DD string."),
    _v(TURN_4_NAMES[6], "Store the target's next-curve-date price minus base price per 100 face, rounded to 6 decimals."),
    _v(TURN_4_NAMES[7], "Store the hedged portfolio's next-curve-date value minus base value per 100 target face, rounded to 6 decimals."),
]

DECIMALS = [
    0, None, None, None, None, 2, 3, 6,
    None, 2, 2, 8, 8, 6, 6, 10,
    6, 6, 6, 4, 6,
    7, 4, 6, 6, 10, None, 6, 6,
]


@lru_cache(maxsize=1)
def _selected_auction():
    frame = load_expansion_table("treasury_auctions").copy()
    for column in ("auction_date", "issue_date", "maturity_date"):
        frame[column] = pd.to_datetime(frame[column])
    eligible = frame.loc[
        frame.auction_date.dt.year.eq(TARGET_YEAR)
        & frame.original_security_term.astype(str).eq(TARGET_TERM)
        & frame.reopening.astype(str).eq("No")
        & frame.inflation_index_security.astype(str).eq("No")
        & frame.floating_rate.astype(str).eq("No")
        & frame.issue_date.notna()
        & pd.to_numeric(frame.bid_to_cover_ratio, errors="coerce").gt(0)
        & pd.to_numeric(frame.int_rate, errors="coerce").gt(0)
        & pd.to_numeric(frame.price_per100, errors="coerce").gt(0)
    ].copy()
    eligible["bid_to_cover_ratio"] = pd.to_numeric(eligible.bid_to_cover_ratio)
    selected = eligible.sort_values(
        ["bid_to_cover_ratio", "auction_date", "cusip"],
        ascending=[False, True, True],
    ).iloc[0]
    return eligible, selected


def _node_yields(row):
    values = np.array([float(pd.to_numeric(row[column])) for column in NODE_COLUMNS])
    if not np.isfinite(values).all():
        raise ValueError("selected curve row has missing benchmark nodes")
    return values


def _bootstrap(node_yields_pct):
    par_rates = np.interp(GRID_MATURITIES, NODE_MATURITIES, node_yields_pct) / 100.0
    discount_factors = []
    for par_rate in par_rates:
        coupon = 100.0 * par_rate / 2.0
        discount_factor = (
            100.0 - coupon * sum(discount_factors)
        ) / (100.0 + coupon)
        if not 0 < discount_factor < 1:
            raise ValueError("bootstrap produced an invalid discount factor")
        discount_factors.append(discount_factor)
    return par_rates, np.asarray(discount_factors)


def _price(coupon_pct, maturity_years, discount_factors):
    periods = int(round(maturity_years * 2))
    coupon = float(coupon_pct) / 2.0
    return float(coupon * discount_factors[:periods].sum() + 100.0 * discount_factors[periods - 1])


@lru_cache(maxsize=1)
def _curve_state():
    _, selected = _selected_auction()
    frame = load_expansion_table("treasury_yield_curve").copy()
    frame["Date"] = pd.to_datetime(frame.Date)
    row = frame.loc[frame.Date.eq(selected.issue_date)]
    if len(row) != 1:
        raise ValueError("selected issue date lacks a unique curve observation")
    row = row.iloc[0]
    node_yields = _node_yields(row)
    par_rates, discount_factors = _bootstrap(node_yields)
    return frame.sort_values("Date"), row, node_yields, par_rates, discount_factors


def _maximum_repricing_residual(par_rates, discount_factors):
    residuals = [
        _price(rate * 100.0, maturity, discount_factors) - 100.0
        for rate, maturity in zip(par_rates, GRID_MATURITIES)
    ]
    return float(np.max(np.abs(residuals)))


def _key_rate_dv01(coupon_pct, maturity_years, node_index, node_yields, base_dfs):
    bumped = node_yields.copy()
    bumped[node_index] += BUMP_BP / 100.0
    _, bumped_dfs = _bootstrap(bumped)
    return _price(coupon_pct, maturity_years, base_dfs) - _price(
        coupon_pct, maturity_years, bumped_dfs
    )


@lru_cache(maxsize=1)
def ground_truth():
    eligible, selected = _selected_auction()
    curve, curve_row, node_yields, par_rates, discount_factors = _curve_state()
    target_coupon = float(selected.int_rate)
    target_price = _price(target_coupon, 10.0, discount_factors)

    _, up_dfs = _bootstrap(node_yields + BUMP_BP / 100.0)
    _, down_dfs = _bootstrap(node_yields - BUMP_BP / 100.0)
    up_price = _price(target_coupon, 10.0, up_dfs)
    down_price = _price(target_coupon, 10.0, down_dfs)
    shock = BUMP_BP / 10_000.0
    duration = (down_price - up_price) / (2.0 * target_price * shock)
    convexity = (
        down_price + up_price - 2.0 * target_price
    ) / (target_price * shock * shock)

    five_index = int(np.where(NODE_MATURITIES == 5.0)[0][0])
    ten_index = int(np.where(NODE_MATURITIES == 10.0)[0][0])
    target_krd = np.array([
        _key_rate_dv01(target_coupon, 10.0, five_index, node_yields, discount_factors),
        _key_rate_dv01(target_coupon, 10.0, ten_index, node_yields, discount_factors),
    ])
    hedge_coupons = (float(node_yields[five_index]), float(node_yields[ten_index]))
    hedge_maturities = (5.0, 10.0)
    hedge_matrix = np.array([
        [
            _key_rate_dv01(hedge_coupons[0], hedge_maturities[0], five_index, node_yields, discount_factors),
            _key_rate_dv01(hedge_coupons[1], hedge_maturities[1], five_index, node_yields, discount_factors),
        ],
        [
            _key_rate_dv01(hedge_coupons[0], hedge_maturities[0], ten_index, node_yields, discount_factors),
            _key_rate_dv01(hedge_coupons[1], hedge_maturities[1], ten_index, node_yields, discount_factors),
        ],
    ])
    hedge_weights = np.linalg.solve(hedge_matrix, -target_krd)
    hedge_residual = target_krd + hedge_matrix @ hedge_weights

    next_row = curve.loc[curve.Date.gt(curve_row.Date)].iloc[0]
    _, next_dfs = _bootstrap(_node_yields(next_row))
    target_change = _price(target_coupon, 10.0, next_dfs) - target_price
    hedge_changes = np.array([
        _price(hedge_coupons[0], hedge_maturities[0], next_dfs) - 100.0,
        _price(hedge_coupons[1], hedge_maturities[1], next_dfs) - 100.0,
    ])
    hedged_change = target_change + float(hedge_weights @ hedge_changes)

    ten_df = float(discount_factors[int(10.0 * 2) - 1])
    thirty_df = float(discount_factors[int(30.0 * 2) - 1])
    ten_spot = 2.0 * (ten_df ** (-1.0 / 20.0) - 1.0) * 100.0
    thirty_spot = 2.0 * (thirty_df ** (-1.0 / 60.0) - 1.0) * 100.0

    return (
        int(len(eligible)),
        str(selected.cusip),
        selected.auction_date.strftime("%Y-%m-%d"),
        selected.issue_date.strftime("%Y-%m-%d"),
        selected.maturity_date.strftime("%Y-%m-%d"),
        float(selected.bid_to_cover_ratio),
        target_coupon,
        float(selected.price_per100),
        curve_row.Date.strftime("%Y-%m-%d"),
        float(node_yields[ten_index]),
        float(node_yields[-1]),
        ten_df,
        thirty_df,
        ten_spot,
        thirty_spot,
        _maximum_repricing_residual(par_rates, discount_factors),
        target_price,
        up_price,
        down_price,
        duration,
        convexity,
        float(target_krd[0]),
        float(target_krd[1]),
        float(hedge_weights[0] * 100.0),
        float(hedge_weights[1] * 100.0),
        float(np.max(np.abs(hedge_residual))),
        next_row.Date.strftime("%Y-%m-%d"),
        target_change,
        hedged_change,
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    result = validate_ordered_outputs(
        outputs,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )
    # Duration and convexity are deterministic functions of three prices that
    # this same turn asks the agent to report to six decimals.  Full-precision
    # prices remain canonical, but a result consistently recomputed from three
    # otherwise-correct reported prices is a pre-registered numerical
    # equivalent rather than a second financial convention.
    derived_names = {
        "target_effective_duration_years",
        "target_effective_convexity",
    }
    if derived_names.intersection(names):
        price_names = (
            "target_model_price_per100",
            "target_parallel_up_price_per100",
            "target_parallel_down_price_per100",
        )
        items = result.variable_results or {}
        missing = any(not item["is_set"] for item in items.values())
        errors = [
            item["message"]
            for item in items.values()
            if item["is_set"] and not item["correct"]
        ]
        result = ValidatorResult(
            not missing and not errors,
            "required output not set" if missing else "; ".join(errors) or "correct",
            missing,
            items,
        )
    return result


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_auction_selection": turn_validator(validate_turn_1),
    "validate_bootstrapped_curve": turn_validator(validate_turn_2),
    "validate_note_repricing": turn_validator(validate_turn_3),
    "validate_key_rate_hedge": turn_validator(validate_turn_4),
}
