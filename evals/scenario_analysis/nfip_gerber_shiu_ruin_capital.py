"""Formula-free hard case; one strict PV channel, private invariants. Raw certification: tests/test_raw_certification_quant_cases.py rebuilds this case's inputs from the source-native archive and asserts they match the governed runtime tables value for value."""

from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from core.data import load_expansion_table
from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator


@lru_cache(None)
def ruin_inputs(scope="gross", end="2024-12-31", frequency="disasters"):
    d = load_expansion_table("fema_nfip")
    cols = [f"{scope}_{k}_payment_usd" for k in ["building", "contents", "icc"]]
    q = (
        d[d.state_usps.eq("FL") & d.date_of_loss.between("2019-01-01", end)]
        .dropna(subset=cols)
        .copy()
    )
    assert q[cols].notna().all().all() and not q.claim_record_id.duplicated().any()
    x = q[cols].astype(float).sum(axis=1)
    eligible = x > 0
    x = x[eligible].to_numpy() / 100000
    f = load_expansion_table("fema_disasters")
    g = f[
        f.state_usps.eq("FL")
        & f.declaration_type.eq("DR")
        & f.incident_type.isin(["Hurricane", "Flood"])
        & f.declaration_date.between("2019-01-01", end)
    ]
    lam = (g.disaster_number.nunique() if frequency == "disasters" else len(g)) / 6
    return x, lam, q.loc[eligible], g


def gs_roots(p, beta, lam, c, delta, penalty=0):
    kernel = lambda s: c * s - lam - delta + lam * np.sum(p * beta / (beta + s))
    derivative = lambda s: c - lam * np.sum(p * beta / (beta + s) ** 2)
    rho = brentq(kernel, 1e-12, max(100, 2 * (lam + delta) / c), xtol=1e-14)
    poles = np.array(
        [
            brentq(kernel, -beta.max() + 1e-11, -beta.min() - 1e-11, xtol=1e-14),
            brentq(kernel, -beta.min() + 1e-11, -1e-12, xtol=1e-14),
        ]
    )
    bt = lambda s: np.sum(p / beta**penalty / (beta + s))
    weights = np.array([lam * (bt(rho) - bt(s)) / derivative(s) for s in poles])
    assert max(poles) < 0
    return lambda u: float(weights @ np.exp(poles * u)), dict(
        rho=rho,
        poles=poles,
        weights=weights,
        kernel_residual=max(abs(kernel(s)) for s in [rho, *poles]),
    )


def ruin_a(scope="gross", end="2024-12-31", frequency="disasters", premium="fixed"):
    x, lam, _, _ = ruin_inputs(scope, end, frequency)
    mean = float(x.mean())
    cv = float(x.var(ddof=0) / mean**2)
    if cv <= 1:
        raise ValueError("H2 balanced means unidentified")
    prob = (1 + np.sqrt((cv - 1) / (cv + 1))) / 2
    p = np.array([prob, 1 - prob])
    beta = 2 * p / mean
    c = 1.25 * lam * mean
    mgf = lambda t: float(np.sum(p * beta / (beta - t)))
    R = brentq(
        lambda t: lam * (mgf(t) - 1) - c * t, 1e-9, beta.min() * (1 - 1e-10), xtol=1e-14
    )
    bound = -np.log(0.01) / R
    u = 0.5 * bound
    g, gd = gs_roots(p, beta, lam, c, 0.05)
    h, hd = gs_roots(p, beta, lam, c, 0.05, 1)
    cs = c if premium == "fixed" else 1.5 * c
    gs, sd = gs_roots(p, beta, 1.5 * lam, cs, 0.05)
    hs, shd = gs_roots(p, beta, 1.5 * lam, cs, 0.05, 1)
    root_u = brentq(lambda v: gs(v) - g(u), 0, 1000, xtol=1e-12)
    out = dict(
        severity_record_count=len(x),
        fast_phase_probability=float(p[0]),
        fast_phase_rate_per100k=float(beta[0]),
        slow_phase_rate_per100k=float(beta[1]),
        annual_arrival_intensity=lam,
        annual_premium_usd_thousand=c * 100,
        adjustment_coefficient_per100k=R,
        lundberg_capital_usd_million=bound / 10,
        discounted_ruin_value=g(u),
        discounted_deficit_usd_thousand=h(u) * 100,
        stress_capital_usd_million=root_u / 10,
        stress_discounted_deficit_usd_thousand=hs(root_u) * 100,
    )
    return out, dict(
        p=p,
        beta=beta,
        lam=lam,
        c=c,
        mean=mean,
        cv=cv,
        base_reserve=u,
        stress_reserve=root_u,
        gs=gd,
        deficit=hd,
        stress_gs=sd,
        stress_deficit=shd,
        moment_error=max(
            abs(np.sum(p / beta) - mean), abs(2 * np.sum(p / beta**2) - np.mean(x * x))
        ),
        adjustment_residual=abs(lam * (mgf(R) - 1) - c * R),
        capital_residual=abs(gs(root_u) - g(u)),
        capital_slope=float(
            sd["weights"] @ (sd["poles"] * np.exp(sd["poles"] * root_u))
        ),
        stress_drift=cs - 1.5 * lam * mean,
    )


TURN_1_NAMES = [
    "severity_record_count",
    "fast_phase_probability",
    "fast_phase_rate_per100k",
    "slow_phase_rate_per100k",
]
TURN_2_NAMES = [
    "annual_arrival_intensity",
    "annual_premium_usd_thousand",
    "adjustment_coefficient_per100k",
    "lundberg_capital_usd_million",
]
TURN_3_NAMES = ["discounted_ruin_value", "discounted_deficit_usd_thousand"]
TURN_4_NAMES = ["stress_capital_usd_million", "stress_discounted_deficit_usd_thousand"]

variables = [
    Variable(
        "severity_record_count",
        None,
        "Store eligible physical claim-record count as an integer.",
    ),
    Variable(
        "fast_phase_probability",
        None,
        "Store fast-phase mixture probability as a decimal rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "fast_phase_rate_per100k",
        None,
        "Store fast exponential rate per USD100000 rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "slow_phase_rate_per100k",
        None,
        "Store slow exponential rate per USD100000 rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "annual_arrival_intensity",
        None,
        "Store Poisson annual arrival intensity rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "annual_premium_usd_thousand",
        None,
        "Store annual premium in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "adjustment_coefficient_per100k",
        None,
        "Store positive adjustment coefficient per USD100000 rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "lundberg_capital_usd_million",
        None,
        "Store Lundberg one-percent-bound capital in USD millions rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "discounted_ruin_value",
        None,
        "Store present value in USD of USD1 paid at first ruin rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "discounted_deficit_usd_thousand",
        None,
        "Store expected discounted positive deficit at first ruin in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "stress_capital_usd_million",
        None,
        "Store stress capital preserving the third-turn discounted ruin value in USD millions rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "stress_discounted_deficit_usd_thousand",
        None,
        "Store stressed expected discounted positive deficit at the solved capital in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def ground_truth():
    out = ruin_a()[0]
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
