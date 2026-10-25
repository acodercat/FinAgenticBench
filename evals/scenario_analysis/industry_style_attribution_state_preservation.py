"""A factor-neutral HiTec program, a stress-regime comparison, and attribution of the original.

A three-factor model for the value-weighted HiTec portfolio is fitted on nonpositive OFR
Financial Stress Index days and turned into a daily-reset factor-neutral program. A
comparison is fitted on positive-stress days and evaluated out of sample; the held-out
evaluation and the drawdown attribution then return to the original program and its own
fitted exposures.

Rejected alternative conventions, each measured against the pinned reading:

- Evaluating the original program with the comparison's exposures gives a held-out return
  of 52.1001 instead of 46.3760 percent and makes SMB rather than Mkt-RF the weakest
  factor leg.
- Charging the fitted intercept against returns each day cuts the original held-out
  return from 46.3760 to 29.6649 percent and deepens its drawdown to 8.4211 percent.
- Dividing residual variance by the observation count instead of the count minus four
  gives a residual volatility of 5.0996 instead of 5.1086 percent.
- Starting the held-out window in 2021 instead of 2020 gives an original return of 48.4500
  percent.
"""

from functools import lru_cache
import numpy as np
from core.data import load_expansion_table, load_ken_french_table

FACTORS = ["mkt_rf_pct", "smb_pct", "hml_pct"]
TOKENS = ["Mkt-RF", "SMB", "HML"]


@lru_cache(None)
def inputs():
    a = load_ken_french_table("daily_industry_returns")
    b = load_ken_french_table("daily_factors")
    o = load_expansion_table("ofr_market_stress")
    z = (
        a[["date", "hitec_pct"]]
        .merge(b, on="date", validate="one_to_one")
        .merge(o[["date", "ofr_fsi"]], on="date", validate="one_to_one")
        .sort_values("date")
    )
    z = z.dropna(subset=["hitec_pct", "rf_pct", "ofr_fsi"] + FACTORS)
    assert np.isfinite(z[["hitec_pct", "rf_pct", "ofr_fsi"] + FACTORS].to_numpy(float)).all()
    return z


def fit(t, ddof=4):
    x = np.column_stack([np.ones(len(t)), t[FACTORS].to_numpy(float) / 100])
    y = (t.hitec_pct - t.rf_pct).to_numpy(float) / 100
    q, r = np.linalg.qr(x, mode="reduced")
    coef = np.linalg.solve(r, q.T @ y)
    res = y - x @ coef
    return (
        coef,
        float(np.sqrt(res @ res / (len(t) - ddof)) * np.sqrt(252) * 100),
        dict(
            count=len(t),
            rank=int(np.linalg.matrix_rank(x)),
            condition=float(np.linalg.cond(x)),
            orthogonality=float(np.max(np.abs(x.T @ res))),
            residual_sum=float(res.sum()),
        ),
    )


def performance(t, b, subtract_alpha=False):
    f = t[FACTORS].to_numpy(float) / 100
    raw = t.hitec_pct.to_numpy(float) / 100
    r = raw - f @ b[1:] - (b[0] if subtract_alpha else 0)
    assert np.min(1 + r) > 0
    nav = np.r_[1, np.cumprod(1 + r)]
    dd = 1 - nav / np.maximum.accumulate(nav)
    trough = int(np.argmax(dd))
    peak = int(np.argmax(nav[: trough + 1]))
    leg = -nav[peak:trough, None] * f[peak:trough] * b[None, 1:] * 100
    contributions = leg.sum(axis=0)
    winner = sorted(range(3), key=lambda i: (contributions[i], TOKENS[i]))[0]
    industry = float(nav[peak:trough] @ raw[peak:trough] * 100)
    alpha = float(-b[0] * nav[peak:trough].sum() * 100) if subtract_alpha else 0
    return dict(
        return_pct=float((nav[-1] - 1) * 100),
        drawdown_pct=float(dd[trough] * 100),
        factor=TOKENS[winner],
        factor_pnl=float(contributions[winner]),
    ), dict(
        peak=peak,
        trough=trough,
        peak_date=str(t.date.iloc[peak - 1]) if peak else "initial",
        trough_date=str(t.date.iloc[trough - 1]),
        terminal_nav=float(nav[-1]),
        minimum_gross_return=float(min(1 + r)),
        factor_pnl_million=contributions.tolist(),
        industry_pnl_million=industry,
        episode_pnl_million=float((nav[trough] - nav[peak]) * 100),
        attribution_residual=float(
            industry + contributions.sum() + alpha - (nav[trough] - nav[peak]) * 100
        ),
        drawdown_runnerup_gap=float(np.sort(dd)[-1] - np.sort(dd)[-2]),
    )


def calculate(coefficient_state="original", intercept_cash="none", variance_ddof=4, holdout="full"):
    z = inputs()
    tr = z[z.date.between("2015-01-01", "2019-12-31")]
    calm = tr[tr.ofr_fsi.le(0)]
    stress = tr[tr.ofr_fsi.gt(0)]
    b, vol, be = fit(calm, variance_ddof)
    alt, av, ae = fit(stress, variance_ddof)
    t = z[z.date.between("2020-01-01" if holdout == "full" else "2021-01-01", "2024-12-31")]
    ap, ape = performance(t, alt, intercept_cash == "subtract")
    op, ope = performance(
        t, b if coefficient_state == "original" else alt, intercept_cash == "subtract"
    )
    out = dict(
        original_calibration_count=len(calm),
        original_market_beta=float(b[1]),
        original_value_beta=float(b[3]),
        original_arithmetic_alpha_pct=float(b[0] * 252 * 100),
        original_residual_volatility_pct=vol,
        comparison_market_beta=float(alt[1]),
        comparison_gross_factor_notional=float(abs(alt[1:]).sum()),
        comparison_holdout_return_pct=ap["return_pct"],
        comparison_max_drawdown_pct=ap["drawdown_pct"],
        original_holdout_return_pct=op["return_pct"],
        original_max_drawdown_pct=op["drawdown_pct"],
        original_worst_factor=op["factor"],
        original_worst_factor_pnl_million=op["factor_pnl"],
    )
    return out, dict(
        original_coefficients=b.tolist(),
        comparison_coefficients=alt.tolist(),
        original_fit=be,
        comparison_fit=ae,
        holdout_count=len(t),
        original_path=ope,
        comparison_path=ape,
    )


from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = ["original_calibration_count", "original_market_beta", "original_value_beta"]
TURN_2_NAMES = ["original_arithmetic_alpha_pct", "original_residual_volatility_pct"]
TURN_3_NAMES = ["comparison_market_beta", "comparison_gross_factor_notional"]
TURN_4_NAMES = ["comparison_holdout_return_pct", "comparison_max_drawdown_pct"]
TURN_5_NAMES = ["original_holdout_return_pct", "original_max_drawdown_pct"]
TURN_6_NAMES = ["original_worst_factor", "original_worst_factor_pnl_million"]

variables = [
    Variable(
        "original_calibration_count",
        None,
        "Store original calibration observation count as an integer.",
    ),
    Variable(
        "original_market_beta",
        None,
        "Store original market loading as a dimensionless coefficient, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_value_beta",
        None,
        "Store original HML loading as a dimensionless coefficient, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_arithmetic_alpha_pct",
        None,
        "Store original arithmetic annual alpha in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_residual_volatility_pct",
        None,
        "Store original annualized regression residual standard error in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "comparison_market_beta",
        None,
        "Store comparison market loading as a dimensionless coefficient, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "comparison_gross_factor_notional",
        None,
        "Store comparison gross factor-contract exposure as a dimensionless NAV multiple, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "comparison_holdout_return_pct",
        None,
        "Store comparison compounded held-out total return in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "comparison_max_drawdown_pct",
        None,
        "Store comparison maximum drawdown as a positive loss percentage, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_holdout_return_pct",
        None,
        "Store original compounded held-out total return in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_max_drawdown_pct",
        None,
        "Store original maximum drawdown as a positive loss percentage, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_worst_factor",
        None,
        "Store selected factor as exactly one token from {'Mkt-RF', 'SMB', 'HML'}.",
    ),
    Variable(
        "original_worst_factor_pnl_million",
        None,
        "Store selected factor’s signed cumulative contribution in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, None, 4]


@lru_cache(maxsize=1)
def ground_truth():
    out = calculate()[0]
    return tuple(out[v.name] for v in variables)


def _validate_subset(outputs, names):
    by = {v.name: v for v in variables}
    truth = dict(zip(by, ground_truth()))
    places = dict(zip(by, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by[n] for n in names], [truth[n] for n in names], [places[n] for n in names]
    )


def validate(outputs):
    return _validate_subset(outputs, [v.name for v in variables])


def validate_turn_1(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_turn_2(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_turn_3(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_turn_4(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate_turn_5(outputs):
    return _validate_subset(outputs, TURN_5_NAMES)


def validate_turn_6(outputs):
    return _validate_subset(outputs, TURN_6_NAMES)


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1, 7)
}
