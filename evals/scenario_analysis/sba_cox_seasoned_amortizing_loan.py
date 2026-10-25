"""Four-turn quantitative-hard case. Exact model variants are public conventions.
Alternative window, model and unit paths are measured private regression probes.
Numerical invariants remain internal; no mechanically zero diagnostic is scored.
Raw certification: tests/test_raw_certification_quant_cases.py rebuilds this case's inputs from the source-native archive and asserts they match the governed runtime tables value for value.
"""

from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


@lru_cache(None)
def cox_inputs(cutoff="2024-12-31", origin="disbursement", scope="TX"):
    d = load_expansion_table("sba_7a")
    d = d[
        d.ProjectState.eq(scope)
        & d.ApprovalFY.between(2020, 2022)
        & d.GrossApproval.gt(0)
        & d.TermInMonths.gt(0)
    ].copy()
    for c in [
        "FirstDisbursementDate",
        "ApprovalDate",
        "PaidInFullDate",
        "ChargeOffDate",
    ]:
        d[c] = pd.to_datetime(d[c])
    d = d[d.FirstDisbursementDate.notna() & d.FirstDisbursementDate.le(cutoff)].copy()
    start = d.FirstDisbursementDate if origin == "disbursement" else d.ApprovalDate
    assert not ((d.ChargeOffDate < start) | (d.PaidInFullDate < start)).any()
    co = d.ChargeOffDate.fillna(pd.Timestamp.max)
    pa = d.PaidInFullDate.fillna(pd.Timestamp.max)
    end = pd.concat(
        [co, pa, pd.Series(pd.Timestamp(cutoff), index=d.index)], axis=1
    ).min(axis=1)
    event = ((co <= pa) & (co <= pd.Timestamp(cutoff))).to_numpy(int)
    age = (end - start).dt.days.to_numpy(int)
    x = np.c_[
        np.log(d.GrossApproval.to_numpy(float) / 100000),
        d.TermInMonths.to_numpy(float) / 120,
    ]
    assert np.isfinite(x).all() and min(age) >= 0
    rr = load_expansion_table("nyfed_reference_rates")
    row = rr[rr["Effective Date"].eq("2024-12-31") & rr["Rate Type"].eq("SOFR")]
    assert len(row) == 1
    return age, event, x, float(row["Rate (%)"].iloc[0]) / 100, d


def cox_fit_a(age, event, x, ties="breslow", start=(0.0, 0.0)):
    order = np.argsort(age)
    age = age[order]
    event = event[order]
    x = x[order]
    times = np.unique(age[event == 1])
    ix = np.searchsorted(age, times)
    dt = np.array([sum((age == t) & (event == 1)) for t in times])
    event_x = np.array([x[(age == t) & (event == 1)].sum(0) for t in times])

    def calc(b, detail=False):
        z = x @ b
        shift = max(z)
        r = np.exp(z - shift)
        s0 = np.cumsum(r[::-1])[::-1][ix]
        s1 = np.cumsum((r[:, None] * x)[::-1], axis=0)[::-1][ix]
        s2 = np.cumsum(
            (r[:, None, None] * x[:, :, None] * x[:, None, :])[::-1], axis=0
        )[::-1][ix]
        if ties == "breslow":
            mean = s1 / s0[:, None]
            ll = float((event_x @ b - dt * (np.log(s0) + shift)).sum())
            score = (event_x - dt[:, None] * mean).sum(0)
            info = (
                dt[:, None, None]
                * (s2 / s0[:, None, None] - mean[:, :, None] * mean[:, None, :])
            ).sum(0)
        else:
            ll = float(event_x.sum(0) @ b)
            score = event_x.sum(0).copy()
            info = np.zeros((2, 2))
            for j, t in enumerate(times):
                fail = (age == t) & (event == 1)
                rr = r[fail]
                xx = x[fail]
                e0 = rr.sum()
                e1 = rr @ xx
                e2 = (xx * rr[:, None]).T @ xx
                for l in range(dt[j]):
                    frac = l / dt[j]
                    ss = s0[j] - frac * e0
                    mm = (s1[j] - frac * e1) / ss
                    ll -= np.log(ss) + shift
                    score -= mm
                    info += (s2[j] - frac * e2) / ss - np.outer(mm, mm)
        return ll, score, info, dt * np.exp(-shift) / s0

    b = np.array(start, dtype=float)
    history = []
    for it in range(80):
        ll, g, H, inc = calc(b)
        history.append(float(max(abs(g))))
        if max(abs(g)) < 1e-9:
            break
        step = np.linalg.solve(H, g)
        f = 1.0
        while calc(b + f * step)[0] < ll - 1e-10:
            f *= 0.5
            if f < 2**-30:
                raise RuntimeError("Cox line search")
        b += f * step
    else:
        raise RuntimeError("Cox convergence")
    ll, g, H, inc = calc(b)
    return (
        b,
        times,
        inc,
        dict(
            loglik=ll,
            score=float(max(abs(g))),
            info_eigenvalues=np.linalg.eigvalsh(H),
            iterations=it,
            history=history,
        ),
    )


def cox_trajectory(
    fit, age, event, x, rate, stress=1.5, discount="effective", independent=False
):
    b, t, inc, diag = fit
    H = np.cumsum(inc)
    at = lambda z: np.r_[0, H][np.searchsorted(t, z, side="right")]
    profile = np.array([np.log(250000 / 100000), 48 / 120])
    rr = np.exp(profile @ b)
    H0 = at(730)
    surv = lambda z, m=1.0: np.exp(-m * rr * (at(z) - H0))
    knots = np.r_[730, t[(t > 730) & (t < 1460)], 1460]
    rm = float(np.sum(surv(knots[:-1]) * np.diff(knots)))
    grid = 730 + np.arange(25) * 365 / 12

    def price(c, m=1.0):
        i = c / 12
        n = 24
        payment = 100 / n if i == 0 else 100 * i / (-np.expm1(-n * np.log1p(i)))
        balances = []
        bal = 100.0
        for k in range(n):
            balances.append(bal)
            bal = bal * (1 + i) - payment
        if independent:
            balances = (
                payment * (-np.expm1(-(n - np.arange(n)) * np.log1p(i))) / i
                if i
                else 100 * (n - np.arange(n)) / n
            )
        ss = surv(grid, m)
        dp = ss[:-1] - ss[1:]
        u = np.arange(1, 25) / 12
        disc = np.exp(-(np.log1p(rate) if discount == "effective" else rate) * u)
        recovery = 0.4 * np.asarray(balances) * dp * disc
        cash = payment * ss[1:] * disc + recovery
        return (
            float(cash.sum()),
            float(cash @ u / cash.sum()),
            float(recovery.sum()),
            float(bal),
            payment,
        )

    base = price(0.08)
    fun = lambda c: price(c, stress)[0] - 100
    if independent:
        lo, hi = 0.0, 0.3
        for _ in range(60):
            mid = (lo + hi) / 2
            if fun(mid) > 0:
                hi = mid
            else:
                lo = mid
        coupon = (lo + hi) / 2
    else:
        coupon = brentq(fun, 0, 0.3, xtol=1e-13)
    fair = price(coupon, stress)
    out = {
        "cohort_count": len(age),
        "chargeoff_count": int(event.sum()),
        "log_approval_coefficient": float(b[0]),
        "term_coefficient": float(b[1]),
        "partial_log_likelihood": diag["loglik"],
        "baseline_cumulative_hazard_day730": float(H0),
        "conditional_net_default_pct": float((1 - surv(1460)) * 100),
        "restricted_mean_default_free_days": rm,
        "seasoned_loan_value_per100": base[0],
        "seasoned_payment_duration_years": base[1],
        "stressed_break_even_coupon_pct": coupon * 100,
        "stressed_recovery_pv_per100": fair[2],
        "stressed_payment_duration_years": fair[1],
    }
    return out, dict(
        **diag,
        minimum_risk_set=int(sum(age >= 1460)),
        profile_risk=rr,
        root_residual=abs(fair[0] - 100),
        remaining_balance=abs(fair[3]),
        probability_residual=abs((1 - surv(1460)) + surv(1460) - 1),
        hazard_min=float(min(inc)),
        coupon_endpoint_values=[fun(0), fun(0.3)],
        H=H,
        times=t,
        beta=b,
    )


def cox_a(
    cutoff="2024-12-31", origin="disbursement", ties="breslow", discount="effective"
):
    age, event, x, r, d = cox_inputs(cutoff, origin)
    return cox_trajectory(
        cox_fit_a(age, event, x, ties), age, event, x, r, discount=discount
    )


TURN_1_NAMES = [
    "cohort_count",
    "chargeoff_count",
    "log_approval_coefficient",
    "term_coefficient",
    "partial_log_likelihood",
]
TURN_2_NAMES = [
    "baseline_cumulative_hazard_day730",
    "conditional_net_default_pct",
    "restricted_mean_default_free_days",
]
TURN_3_NAMES = ["seasoned_loan_value_per100", "seasoned_payment_duration_years"]
TURN_4_NAMES = [
    "stressed_break_even_coupon_pct",
    "stressed_recovery_pv_per100",
    "stressed_payment_duration_years",
]

variables = [
    Variable(
        "cohort_count",
        None,
        "Store the eligible equally weighted public loan subject count as an integer.",
    ),
    Variable(
        "chargeoff_count",
        None,
        "Store the observed charge-off event count as an integer.",
    ),
    Variable(
        "log_approval_coefficient",
        None,
        "Store the log gross-approval Cox coefficient, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "term_coefficient",
        None,
        "Store the original-term Cox coefficient, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "partial_log_likelihood",
        None,
        "Store the unnormalized Breslow partial log likelihood without additive constants, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "baseline_cumulative_hazard_day730",
        None,
        "Store the Breslow cumulative baseline hazard at age730 days and zero covariates, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "conditional_net_default_pct",
        None,
        "Store the conditional net default probability from day730 through1460 in percent, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "restricted_mean_default_free_days",
        None,
        "Store the restricted mean remaining default-free days between730 and1460, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "seasoned_loan_value_per100",
        None,
        "Store the actuarial loan value per100 current outstanding, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "seasoned_payment_duration_years",
        None,
        "Store the Macaulay duration of expected discounted loan cash flows in years, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "stressed_break_even_coupon_pct",
        None,
        "Store the stressed break-even nominal annual coupon in percent, convertible monthly, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "stressed_recovery_pv_per100",
        None,
        "Store the present value of stressed recoveries per100 current outstanding, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
    Variable(
        "stressed_payment_duration_years",
        None,
        "Store the Macaulay duration of stressed expected discounted loan cash flows in years, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.",
    ),
]
DECIMALS = [0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def ground_truth():
    result = cox_a()[0]
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
