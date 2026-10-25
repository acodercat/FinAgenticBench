"""Four-turn quantitative-hard case. Exact model variants are public conventions.
Alternative window, model and unit paths are measured private regression probes.
Numerical invariants remain internal; no mechanically zero diagnostic is scored.
Raw certification: tests/test_raw_certification_quant_cases.py rebuilds this case's inputs from the source-native archive and asserts they match the governed runtime tables value for value.
"""

from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import cho_factor, cho_solve
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator

MATS = np.array([1, 2, 3, 5, 7, 10, 20, 30.0])
LABELS = ["1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"]


@lru_cache(None)
def sw_inputs(date="2023-12-29"):
    d = load_expansion_table("treasury_yield_curve")
    r = d[d.Date.eq(date)]
    assert len(r) == 1
    y = r[LABELS].to_numpy(float)[0] / 100
    s = load_expansion_table("dol_form5500_schedule_sb")
    s = s[
        s.sponsor_ein.eq("060570975")
        & s.plan_number.eq("041")
        & s.plan_year_end.eq("2024-12-31")
        & s.valuation_date.eq("2024-01-01")
        & s.filing_status.eq("FILING_RECEIVED")
        & s.date_received.le("2025-12-31")
    ].sort_values(["date_received", "filing_id"])
    row = s.iloc[-1]
    return (
        y,
        float(row.total_funding_target_usd) / 1e9,
        float(row.schedule_sb_participant_count),
        str(row.filing_id),
    )


def coupon_matrix(y):
    C = np.zeros((30, 8))
    for j, t in enumerate(MATS.astype(int)):
        C[:t, j] = y[j]
        C[t - 1, j] += 1
    return C


def sw_curve_a(y, ufr=0.035, alpha=0.1, compounding="effective", variant="wilson"):
    omega = np.log1p(ufr) if compounding == "effective" else ufr
    u = np.arange(1, 31.0)
    C = coupon_matrix(y)

    def kernel(t, v):
        a = np.minimum(t, v)
        b = np.maximum(t, v)
        return np.exp(-omega * (t + v)) * (
            alpha * a - 0.5 * (np.exp(-alpha * (b - a)) - np.exp(-alpha * (b + a)))
        )

    W = kernel(u[:, None], u[None, :])
    K = C.T @ W @ C
    mu = np.exp(-omega * u)
    beta = cho_solve(cho_factor(K, lower=True), np.ones(8) - C.T @ mu)
    coef = C @ beta

    def discount(t):
        t = np.atleast_1d(t).astype(float)
        if variant == "flat":
            # Bootstrap annual par-grid quotes by maturity interpolation, then flat-last-forward beyond30.
            q = np.interp(u, MATS, y)
            df = []
            for j, yy in enumerate(q):
                df.append((1 - yy * sum(df)) / (1 + yy))
            f = -np.log(df[-1] / df[-2])
            return np.exp(
                np.interp(np.minimum(t, 30), np.r_[0, u], np.log(np.r_[1, df]))
                - np.maximum(t - 30, 0) * f
            )
        return np.exp(-omega * t) + kernel(t[:, None], u[None, :]) @ coef

    def forward(t):
        if variant == "flat":
            q = np.interp(u, MATS, y)
            df = []
            for yy in q:
                df.append((1 - yy * sum(df)) / (1 + yy))
            return float(-np.log(df[-1] / df[-2]))
        h = alpha * u - np.exp(-alpha * t) * np.sinh(alpha * u)
        hp = alpha * np.exp(-alpha * t) * np.sinh(alpha * u)
        z = coef * np.exp(-omega * u)
        return omega - (hp @ z) / (1 + h @ z)

    return (
        discount,
        forward,
        dict(
            kernel_condition=float(np.linalg.cond(K)),
            kernel_min_eigenvalue=float(min(np.linalg.eigvalsh(K))),
            repricing_residual=float(max(abs(C.T @ discount(u) - 1))),
            coefficient=coef,
        ),
    )


def sw_trajectory(
    y,
    target,
    n,
    ufr=0.035,
    alpha=0.1,
    compounding="effective",
    variant="wilson",
    independent=False,
):
    make = sw_curve_a
    kw = {} if independent else dict(compounding=compounding, variant=variant)
    D, F, diag = make(y, ufr, alpha, **kw)
    years = np.arange(1, 51.0)
    df = D(years)
    first = n * 5000 / 1e9
    func = lambda g: float(df @ (first * (1 + g) ** (years - 1)) - target)
    if independent:
        lo, hi = -0.05, 0.05
        for _ in range(60):
            mid = (lo + hi) / 2
            if func(mid) > 0:
                hi = mid
            else:
                lo = mid
        growth = (lo + hi) / 2
    else:
        growth = brentq(func, -0.05, 0.05, xtol=1e-14)
    cf = first * (1 + growth) ** (years - 1)
    pv = cf * df
    duration = float(pv @ years / target)
    longshare = float(pv[years > 30].sum() / target * 100)
    hedgeT = np.array([1.0, 60.0])
    hedgeD = D(hedgeT)
    hold = np.linalg.solve(
        np.array([hedgeD, hedgeT * hedgeD]), [target, target * duration]
    )
    convexity = float((hold * hedgeD) @ (hedgeT**2) - pv @ (years**2))
    Ds, Fs, stressdiag = make(y, ufr - 0.01, alpha, **kw)
    l = float(cf @ Ds(years))
    assets = float(hold @ Ds(hedgeT))
    share = float((cf * Ds(years))[years > 30].sum() / l * 100)
    out = {
        "discount_factor_year40": float(D(40)[0]),
        "discount_factor_year60": float(D(60)[0]),
        "instantaneous_forward_year60_pct": float(F(60) * 100),
        "implied_benefit_growth_pct": growth * 100,
        "liability_duration_years": duration,
        "liability_after30_pv_share_pct": longshare,
        "year1_zero_face_usd_bn": float(hold[0]),
        "year60_zero_face_usd_bn": float(hold[1]),
        "surplus_convexity_usd_bn_year2": convexity,
        "ufr_stress_surplus_usd_bn": assets - l,
        "ufr_stress_after30_pv_share_pct": share,
    }
    return out, dict(
        **diag,
        growth_endpoint_values=[func(-0.05), func(0.05)],
        growth_residual=abs(func(growth)),
        immunization_residual=float(
            max(
                abs(
                    np.array([hedgeD, hedgeT * hedgeD]) @ hold
                    - [target, target * duration]
                )
            )
        ),
        minimum_discount=float(min(D(np.linspace(0, 150, 1501)))),
        minimum_holding=float(min(hold)),
        cf=cf,
        years=years,
        holdings=hold,
        discounts=df,
        growth=growth,
        stress_repricing=stressdiag.get(
            "repricing_residual", stressdiag.get("boundary_residual")
        ),
    )


def sw_a(date="2023-12-29", alpha=0.1, compounding="effective", variant="wilson"):
    y, t, n, f = sw_inputs(date)
    return sw_trajectory(y, t, n, alpha=alpha, compounding=compounding, variant=variant)


TURN_1_NAMES = [
    "discount_factor_year40",
    "discount_factor_year60",
    "instantaneous_forward_year60_pct",
]
TURN_2_NAMES = [
    "implied_benefit_growth_pct",
    "liability_duration_years",
    "liability_after30_pv_share_pct",
]
TURN_3_NAMES = [
    "year1_zero_face_usd_bn",
    "year60_zero_face_usd_bn",
    "surplus_convexity_usd_bn_year2",
]
TURN_4_NAMES = ["ufr_stress_surplus_usd_bn", "ufr_stress_after30_pv_share_pct"]

variables = [
    Variable(
        "discount_factor_year40",
        None,
        "Store the year40 discount factor, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "discount_factor_year60",
        None,
        "Store the year60 discount factor, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "instantaneous_forward_year60_pct",
        None,
        "Store the year60 instantaneous forward rate in continuously compounded annual percent, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "implied_benefit_growth_pct",
        None,
        "Store the implied annual geometric benefit growth rate in percent, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "liability_duration_years",
        None,
        "Store the Fisher–Weil liability duration in years under a parallel additive continuous spot shift, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "liability_after30_pv_share_pct",
        None,
        "Store the liability present-value share of payments strictly afteryear30 in percent, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "year1_zero_face_usd_bn",
        None,
        "Store the year1 zero-coupon bond face amount in USD billions, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "year60_zero_face_usd_bn",
        None,
        "Store the year60 zero-coupon bond face amount in USD billions, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "surplus_convexity_usd_bn_year2",
        None,
        "Store the unnormalized second derivative of asset-minus-liability PV with respect to a parallel additive annual continuously compounded spot-rate shift expressed in decimals, in USD billion-years squared, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "ufr_stress_surplus_usd_bn",
        None,
        "Store the asset-minus-liability present-value surplus after the UFR reduction in USD billions, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "ufr_stress_after30_pv_share_pct",
        None,
        "Store the stressed liability present-value share of payments strictly afteryear30 in percent, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
]
DECIMALS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def ground_truth():
    result = sw_a()[0]
    return tuple(result[v.name] for v in variables)


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
