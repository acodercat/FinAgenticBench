"""Author A: scalar cumulant reduction and conditional gamma-normal integration."""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.special import gammaln, ndtr
from functools import lru_cache
from core.data import load_expansion_table, load_ken_french_table


@lru_cache(None)
def inputs(start="2020-01-01", simple=False, rate_type="SOFR", cash="effective"):
    f = load_ken_french_table("daily_industry_returns")
    d = f[f.date.between(start, "2024-12-31")].sort_values("date")
    assert not d.date.duplicated().any()
    a = d.hitec_pct.to_numpy(float) / 100
    assert np.isfinite(a).all() and min(a) > -1
    x = a if simple else np.log1p(a)
    q = load_expansion_table("nyfed_reference_rates")
    q = q[
        q["Rate Type"].eq(rate_type)
        & q["Effective Date"].between("2024-01-01", "2024-12-31")
    ]
    assert not q["Effective Date"].duplicated().any()
    rate = float(q["Rate (%)"].mean()) / 100
    r = np.log1p(rate) if cash == "effective" else rate
    return (
        x,
        r,
        {
            "n": len(x),
            "date_first": d.date.iloc[0],
            "date_last": d.date.iloc[-1],
            "fixing_count": len(q),
            "cash_annual_effective": rate,
        },
    )


def fit(x):
    m = x.mean()
    v = np.mean((x - m) ** 2)
    c3 = np.mean((x - m) ** 3)
    c4 = np.mean((x - m) ** 4) - 3 * v * v
    assert v > 0 and c4 > 0
    rat = c3 * c3 / (c4 * v)
    assert v > 0 and c4 > 0 and 0 < rat < 2 / 3
    z = brentq(
        lambda z: z * (3 - z) ** 2 / (3 * (1 + 2 * z - z * z)) - rat, 0, 1, xtol=1e-15
    )
    vr = v * 252
    nu = c4 * 252 / (3 * vr * vr * (1 + 2 * z - z * z))
    theta = np.sign(c3) * np.sqrt(z * vr / nu)
    sigma = np.sqrt(vr * (1 - z))
    mu = m * 252 - theta
    return np.array([mu, theta, sigma, nu]), np.array([m, v, c3, c4]) * 252


def compute(x, r, mean_correction=False, days=5):
    pars, ks = fit(x)
    mu, theta, sigma, nu = pars
    T = days / 252
    S0 = 100.0
    K = 102.0

    def d(u):
        return 1 - theta * nu * u - 0.5 * sigma * sigma * nu * u * u

    roots = np.sort(np.roots([-0.5 * sigma * sigma * nu, -theta * nu, 1.0]))
    lower, upper = roots

    def cumulant(u):
        return mu * u - np.log(d(u)) / nu

    h = brentq(
        lambda h: cumulant(h + 1) - cumulant(h) - r,
        lower + 1e-8,
        upper - 1 - 1e-8,
        xtol=1e-12,
    )
    if mean_correction:
        h = 0.0
        mu = r + np.log(d(1)) / nu
    D = d(h)
    th = (theta + sigma * sigma * h) / D
    sig = sigma / np.sqrt(D)
    assert d(h + 2) > 0
    shape = T / nu

    def tail(a, threshold):
        if threshold <= 0:
            return float(
                S0**a
                * np.exp(
                    T
                    * (
                        mu * a
                        - np.log(1 - th * nu * a - 0.5 * sig * sig * nu * a * a) / nu
                    )
                )
            ), 0.0
        logk = np.log(threshold / S0)

        # Integrate unit-scale gamma clock to avoid tiny-clock conditioning.
        def fun(y):
            if y <= 0:
                return 0.0
            g = nu * y
            z = (mu * T + th * g + a * sig * sig * g - logk) / (sig * np.sqrt(g))
            lg = (
                (shape - 1) * np.log(y)
                - y
                - gammaln(shape)
                + a * mu * T
                + (a * th + 0.5 * a * a * sig * sig) * g
            )
            return np.exp(lg) * ndtr(z)

        val, err = quad(fun, 0, np.inf, epsabs=2e-12, epsrel=2e-12, limit=500)
        return S0**a * val, S0**a * err

    m0, e0 = tail(0, K)
    m1, e1 = tail(1, K)
    m2, e2 = tail(2, K)
    C = m1 - K * m0
    C2 = m2 - 2 * K * m1 + K * K * m0
    ES = tail(1, 0)[0]
    ES2 = tail(2, 0)[0]
    var = ES2 - ES * ES
    SC = m2 - K * m1
    a = (SC - ES * C) / var
    B = C - a * ES
    resvar = C2 - C * C - a * a * var
    assert 0 < a < 1 and resvar > 0
    low = (-1 - B) / a
    high = (K + B + 1) / (1 - a)
    assert low < K < high
    pshort = 1 - tail(0, low)[0] + tail(0, high)[0]
    vals = [
        len(x),
        pars[0] * 100,
        theta * 100,
        sigma * 100,
        nu * 252,
        h,
        r * 100,
        np.exp(-r * T) * C,
        C2,
        a,
        np.sqrt(resvar),
        pshort,
    ]
    diag = {
        "physical_parameters": pars.tolist(),
        "cumulants_per_year": ks.tolist(),
        "mgf_domain": roots.tolist(),
        "esscher_h": h,
        "pricing_theta": th,
        "pricing_sigma": sig,
        "T": T,
        "price": np.exp(-r * T) * C,
        "call_mean": C,
        "call_second_moment": C2,
        "stock_mean": ES,
        "stock_second_moment": ES2,
        "stock_call_cross_moment": SC,
        "stock_variance": var,
        "hedge_shares": a,
        "hedge_terminal_cash": B,
        "hedge_residual_variance": resvar,
        "shortfall_thresholds": [low, high],
        "tail_moments": [m0, m1, m2],
        "quadrature_error_estimates": [e0, e1, e2],
        "martingale_error": float(ES - S0 * np.exp(r * T)),
        "moment_ratio": float(ks[2] ** 2 / (ks[3] * ks[1])),
        "root_slope": float(
            (theta + sigma * sigma * (h + 1)) / d(h + 1)
            - (theta + sigma * sigma * h) / d(h)
        ),
    }
    return vals, diag


from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = [
    "return_count",
    "deterministic_log_drift_pct",
    "brownian_clock_drift_pct",
    "brownian_volatility_pct",
    "gamma_variance_trading_days",
]

TURN_2_NAMES = ["esscher_tilt", "cash_force_pct"]

TURN_3_NAMES = ["call_price_usd", "call_payoff_second_moment_usd2"]

TURN_4_NAMES = [
    "optimal_static_shares",
    "minimum_terminal_rmse_usd",
    "one_dollar_shortfall_probability",
]

variables = [
    Variable(
        "return_count", None, "Store daily log-return observation count as an integer."
    ),
    Variable(
        "deterministic_log_drift_pct",
        None,
        "Store separate deterministic log drift in percent per year rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "brownian_clock_drift_pct",
        None,
        "Store Brownian gamma-clock drift in percent per year rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "brownian_volatility_pct",
        None,
        "Store Brownian volatility in percent per square-root year rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "gamma_variance_trading_days",
        None,
        "Store unit-mean gamma-clock variance-rate parameter in model trading days, with 252 days per year rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "esscher_tilt",
        None,
        "Store log-return Esscher tilt parameter, dimensionless rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "cash_force_pct",
        None,
        "Store constant continuously compounded bank-account annual rate in percent rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "call_price_usd",
        None,
        "Store current European-call price in USD rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "call_payoff_second_moment_usd2",
        None,
        "Store undiscounted terminal call-payoff second raw moment in USD squared rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "optimal_static_shares",
        None,
        "Store optimal static fund-share position for a short one-call hedge rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "minimum_terminal_rmse_usd",
        None,
        "Store minimum RMS undiscounted terminal hedging error in USD under the carried Esscher measure rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "one_dollar_shortfall_probability",
        None,
        "Store Esscher probability that call payoff exceeds terminal hedge wealth by more than USD1, as a decimal rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


def calculate(
    start="2020-01-01",
    simple=False,
    mean_correction=False,
    days=5,
    rate_type="SOFR",
    cash="effective",
):
    x, r, e = inputs(start, simple, rate_type, cash)
    v, d = compute(x, r, mean_correction, days)
    return dict(zip((x.name for x in variables), v)), d


@lru_cache(maxsize=1)
def ground_truth():
    out = calculate()[0]
    return tuple(out[v.name] for v in variables)


def _validate_subset(outputs, names):
    by = {v.name: v for v in variables}
    truth = dict(zip(by, ground_truth()))
    d = dict(zip(by, DECIMALS))
    return validate_ordered_outputs(
        outputs,
        [by[n] for n in names],
        [truth[n] for n in names],
        [d[n] for n in names],
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


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
