"""Quantitative stochastic-control case with a single strict PV channel."""

from functools import lru_cache
import numpy as np
from scipy.optimize import root
from core.data import load_expansion_table
from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator


@lru_cache(None)
def producer_inputs(year=2024, ddof=0):
    d = load_expansion_table("eia_bulk_wti_daily")
    d = d[d.observation_date.between(f"{year}-01-01", f"{year}-12-31")].sort_values(
        "observation_date"
    )
    assert d.observation_date.is_unique
    x = d.wti_spot_price_usd_per_barrel.to_numpy(float)
    assert np.all(x > 0)
    log = np.diff(np.log(x))
    sigma = log.std(ddof=ddof) * np.sqrt(252)
    r = load_expansion_table("treasury_yield_curve")
    row = r[r.Date == "2024-12-31"]
    assert len(row) == 1
    rate = float(row["10 Yr"].iloc[0]) / 100
    return (
        sigma,
        rate,
        float(x[-1]),
        len(log),
        {
            "quote_count": len(x),
            "end_quote": float(x[-1]),
            "log_return_mean": float(log.mean()),
            "annual_volatility": float(sigma),
            "discount_force": rate,
        },
    )


def producer(sigma, r, p0, C=60.0, K=200.0, L=20.0, mu=0.0):
    if sigma <= 0 or r <= max(mu, 0) or min(C, K, L) <= 0 or C <= r * L:
        raise ValueError("producer domain")
    bn, bp = np.sort(np.roots([sigma * sigma / 2, mu - sigma * sigma / 2, -r]))
    scale = C / r
    # z=P/C, constant active-value slope r/(r-mu) in scale C/r.
    slope = r / (r - mu)
    kk = K / scale
    ll = L / scale
    abandon = bn / (bn - 1) * (1 - ll) / slope
    B_ab = -slope / bn * abandon ** (1 - bn)
    active_ab = lambda z: scale * (slope * z - 1 + B_ab * z**bn) if z > abandon else -L

    def f(v):
        lo, hi, A, B = v
        if min(lo, hi) <= 0:
            return np.full(4, 1e6)
        D = lambda z: slope * z - 1 + B * z**bn - A * z**bp
        Dp = lambda z: slope + bn * B * z ** (bn - 1) - bp * A * z ** (bp - 1)
        return [D(lo) + ll, D(hi) - kk, Dp(lo), Dp(hi)]

    sol = root(f, [0.45, 1.8, 0.2, 0.25], tol=1e-11)
    lo, hi, A, B = sol.x
    if not sol.success and max(abs(np.array(f(sol.x)))) > 1e-10:
        raise RuntimeError(sol.message)
    assert 0 < lo < (1 - ll) / slope < (1 + kk) / slope < hi and min(A, B) > 0
    low, high = lo * C, hi * C
    idle = lambda P: (
        scale * A * (P / C) ** bp
        if P < high
        else scale * (slope * P / C - 1 + B * (P / C) ** bn) - K
    )
    down = (high / low) ** bn
    up = (low / high) ** bp
    start = (p0 / high) ** bp
    assert p0 < high
    entries = start / (1 - down * up)
    exits = entries * down
    occupation = entries * (1 - down) / r
    revenue = entries * (high - down * low) / (r - mu)
    recon = revenue - C * occupation - K * entries - L * exits - idle(p0)
    out = {
        "increment_count": 0,
        "diffusion_volatility_pct": sigma * 100,
        "discount_force_pct": r * 100,
        "permanent_exit_price": abandon * C,
        "permanent_active_value": active_ab(p0 / C),
        "switching_exit_price": low,
        "switching_entry_price": high,
        "idle_project_value": idle(p0),
        "discounted_operating_years": occupation,
        "entry_outlay_pv": K * entries,
        "exit_outlay_pv": L * exits,
    }
    diag = {
        "bp": bp,
        "bn": bn,
        "A": A,
        "B": B,
        "root_residual": max(abs(np.array(f(sol.x)))),
        "down_transform": down,
        "up_transform": up,
        "renewal_product": down * up,
        "cashflow_reconciliation": recon,
        "revenue_pv": revenue,
        "qvi_exit_margin": C - r * L - low,
        "qvi_entry_margin": high - C - r * K,
        "initial_entry_margin": high - p0,
        "solver_success": sol.success,
        "root_nfev": sol.nfev,
    }
    return out, diag


def calculate(*, year=2024, ddof=0, clock=252, mu=0.0):
    sigma, r, p0, n, _ = producer_inputs(year, ddof)
    out, diag = producer(sigma * np.sqrt(clock / 252), r, p0, mu=mu)
    out["increment_count"] = n
    return out, diag


TURN_1_NAMES = ["increment_count", "diffusion_volatility_pct", "discount_force_pct"]
TURN_2_NAMES = ["permanent_exit_price", "permanent_active_value"]
TURN_3_NAMES = ["switching_exit_price", "switching_entry_price", "idle_project_value"]
TURN_4_NAMES = ["discounted_operating_years", "entry_outlay_pv", "exit_outlay_pv"]

variables = [
    Variable(
        "increment_count",
        None,
        "Store number of consecutive published within-year log increments as an integer.",
    ),
    Variable(
        "diffusion_volatility_pct",
        None,
        "Store annual GBM diffusion volatility in percent rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "discount_force_pct",
        None,
        "Store hypothetical annual force of discount in percent rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "permanent_exit_price",
        None,
        "Store optimal permanent-exit threshold in USD per barrel rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "permanent_active_value",
        None,
        "Store active project value with permanent abandonment at the initial spot in USD rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "switching_exit_price",
        None,
        "Store optimal recurring-switching exit threshold in USD per barrel rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "switching_entry_price",
        None,
        "Store optimal recurring-switching entry threshold in USD per barrel rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "idle_project_value",
        None,
        "Store idle project value at the initial spot in USD rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "discounted_operating_years",
        None,
        "Store expected continuously discounted cumulative active time in years rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "entry_outlay_pv",
        None,
        "Store expected present value of all entry and restart outlays in USD, positive cost rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "exit_outlay_pv",
        None,
        "Store expected present value of all shutdown outlays in USD, positive cost rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def ground_truth():
    out = calculate()[0]
    return tuple(out[v.name] for v in variables)


def _validate_subset(outputs, names):
    by = {v.name: v for v in variables}
    truth = dict(zip(by, ground_truth()))
    decimals = dict(zip(by, DECIMALS))
    return validate_ordered_outputs(
        outputs,
        [by[n] for n in names],
        [truth[n] for n in names],
        [decimals[n] for n in names],
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
