"""Quantitative stochastic-control case with a single strict PV channel."""

from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import logsumexp
from core.data import load_expansion_table
from core.dataset_layers import load_catalog, DATASETS_DIR
from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator


@lru_cache(None)
def household_inputs(state="TX", purpose="1", quantiles="inverted_cdf"):
    meta = load_catalog()["tables"]["hmda.loan_applications"]
    cols = [
        "activity_year",
        "state_code",
        "action_taken",
        "lien_status",
        "loan_type",
        "loan_purpose",
        "occupancy_type",
        "total_units",
        "income",
    ]
    out = []
    eligible = 0
    for d in pd.read_csv(
        DATASETS_DIR / meta["layers"]["runtime"]["path"],
        usecols=cols,
        dtype=str,
        chunksize=300000,
    ):
        take = (
            d.activity_year.eq("2024")
            & d.state_code.eq(state)
            & d.action_taken.eq("1")
            & d.lien_status.eq("1")
            & d.loan_type.eq("1")
            & d.loan_purpose.eq(purpose)
            & d.occupancy_type.eq("1")
            & d.total_units.eq("1")
        )
        x = pd.to_numeric(d.loc[take, "income"], errors="coerce")
        eligible += len(x)
        out.extend(x[x.between(10, 500)].tolist())
    x = np.array(out)
    income = np.quantile(x, [0.2, 0.5, 0.8], method=quantiles)
    rates = load_expansion_table("nyfed_reference_rates")
    sofr = rates[
        rates["Rate Type"].eq("SOFR")
        & rates["Effective Date"].between("2024-01-01", "2024-12-31")
    ]
    assert sofr["Effective Date"].is_unique
    rate = sofr["Rate (%)"].mean() / 100
    return (
        income,
        rate,
        len(x),
        {
            "eligible_before_income": eligible,
            "observations": len(x),
            "quantiles": income.tolist(),
            "income_mean": float(x.mean()),
            "sofr_count": len(sofr),
            "sofr_annual_effective": float(rate),
        },
    )


def gamma_fit(incomes, target=0.9):
    ys = np.asarray(incomes, float) / np.mean(incomes)

    def ce(g):
        a = 1 - g
        return (
            np.exp(np.mean(np.log(ys)))
            if abs(a) < 1e-8
            else np.exp((logsumexp(a * np.log(ys)) - np.log(len(ys))) / a)
        )

    gamma = brentq(lambda g: ce(g) - target, 0.001, 20, xtol=1e-13)
    return gamma, ce(gamma) - target


def make_tree(ys, R, x, beta=0.96):
    # Preorder by date: 1+3+9 decisions, 27 terminal consumptions.
    nodes = [(0, None, 0.0)]
    levels = [[0]]
    for t in range(1, 4):
        level = []
        for parent in levels[-1]:
            for y in ys:
                level.append(len(nodes))
                nodes.append((t, parent, y))
        levels.append(level)
    M = np.zeros((40, 13))
    b = np.zeros(40)
    weights = np.zeros(40)
    for i, (t, parent, y) in enumerate(nodes):
        b[i] = x if i == 0 else y
        weights[i] = (beta / 3) ** t
        if parent is not None:
            M[i, parent] = R
        if t < 3:
            M[i, i] = -1
    return M, b, weights, levels


def optimize(ys, R, gamma, x=1.2, limit=0.0, beta=0.96, start=None):
    M, b, w, levels = make_tree(ys, R, x, beta)

    def u(c):
        return (
            np.log(c)
            if abs(gamma - 1) < 1e-10
            else (c ** (1 - gamma) - 1) / (1 - gamma)
        )

    def fun(s):
        c = b + M @ s
        if np.min(c) <= 0:
            raise ValueError("nonpositive trial consumption")
        return -float(w @ u(c))

    def jac(s):
        return -M.T @ (w * (b + M @ s) ** -gamma)

    def hess(s):
        return M.T @ ((w * gamma * (b + M @ s) ** (-gamma - 1))[:, None] * M)

    s0 = np.zeros(13) if start is None else np.asarray(start)
    sol = minimize(
        fun,
        s0,
        jac=jac,
        method="SLSQP",
        bounds=[(-limit, None)] * 13,
        constraints=[
            {"type": "ineq", "fun": lambda s: b + M @ s - 1e-10, "jac": lambda s: M}
        ],
        options={"ftol": 1e-13, "maxiter": 500},
    )
    s = sol.x.copy()
    active = s + limit < 1e-7
    s[active] = -limit
    free = ~active
    for it in range(30):
        g = jac(s)
        H = hess(s)
        if np.max(abs(g[free])) < 2e-13:
            break
        step = np.zeros(13)
        step[free] = np.linalg.solve(H[np.ix_(free, free)], -g[free])
        alpha = 1.0
        while (
            np.min(b + M @ (s + alpha * step)) <= 0
            or np.min(s[free] + alpha * step[free]) <= -limit
        ):
            alpha *= 0.5
        s += alpha * step
    c = b + M @ s
    g = jac(s)
    V = -fun(s)
    average_utility = V / w.sum()
    ce = (
        np.exp(average_utility)
        if abs(gamma - 1) < 1e-10
        else (1 + (1 - gamma) * average_utility) ** (1 / (1 - gamma))
    )
    kkt = max(
        float(np.max(abs(g[free]))),
        float(max(0.0, -np.min(g[active]))) if active.any() else 0.0,
    )
    if kkt > 1e-10 or np.min(c) <= 0 or np.min(s) < -limit - 1e-12:
        raise RuntimeError("KKT/domain not certified")
    return {
        "s": s,
        "c": c,
        "V": V,
        "ce": ce,
        "kkt": kkt,
        "active": np.flatnonzero(active),
        "multipliers": g[active],
        "min_consumption": min(c),
        "min_free_bound_margin": min(s[free] + limit),
        "hessian_condition": np.linalg.cond(hess(s)),
        "iterations": sol.nit,
        "status": sol.message,
        "date1_borrow_prob": float(np.mean(s[1:4] < -1e-9)),
        "date2_debt": float(np.mean(np.maximum(-s[4:13], 0))),
    }


def household(incomes, rate, target=0.9, beta=0.96, limit=0.25):
    scale = float(np.mean(incomes))
    ys = np.asarray(incomes, float) / scale
    R = 1 + rate
    g, res = gamma_fit(incomes, target)
    base = optimize(ys, R, g, beta=beta)
    credit = optimize(ys, R, g, limit=limit, beta=beta)
    fee = brentq(
        lambda f: (
            optimize(ys, R, g, x=1.2 - f, limit=limit, beta=beta)["V"] - base["V"]
        ),
        0,
        1.1,
        xtol=5e-13,
        rtol=1e-14,
    )
    fair = optimize(ys, R, g, x=1.2 - fee, limit=limit, beta=beta)
    out = {
        "income_record_count": 0,
        "low_income_usd_thousand": float(incomes[0]),
        "middle_income_usd_thousand": float(incomes[1]),
        "high_income_usd_thousand": float(incomes[2]),
        "crra_risk_aversion": g,
        "cash_annual_effective_pct": rate * 100,
        "baseline_initial_consumption_usd_thousand": base["c"][0] * scale,
        "baseline_constant_consumption_equivalent_usd_thousand": base["ce"] * scale,
        "credit_initial_consumption_usd_thousand": credit["c"][0] * scale,
        "credit_constant_consumption_equivalent_usd_thousand": credit["ce"] * scale,
        "year_one_borrowing_probability": credit["date1_borrow_prob"],
        "maximum_upfront_fee_usd_thousand": fee * scale,
        "fair_fee_year_two_expected_debt_usd_thousand": fair["date2_debt"] * scale,
    }
    return out, {
        "gamma_fit_residual": res,
        "scale": scale,
        "base": base,
        "credit": credit,
        "fair_fee": fair,
        "utility_indifference": fair["V"] - base["V"],
    }


def calculate(
    *, income_scale=1.0, rate_multiplier=1.0, beta=0.96, target=0.9, limit=0.25
):
    incomes, rate, n, _ = household_inputs()
    out, diag = household(
        incomes * income_scale,
        rate * rate_multiplier,
        target=target,
        beta=beta,
        limit=limit,
    )
    out["income_record_count"] = n
    return out, diag


TURN_1_NAMES = [
    "income_record_count",
    "low_income_usd_thousand",
    "middle_income_usd_thousand",
    "high_income_usd_thousand",
    "crra_risk_aversion",
]
TURN_2_NAMES = [
    "cash_annual_effective_pct",
    "baseline_initial_consumption_usd_thousand",
    "baseline_constant_consumption_equivalent_usd_thousand",
]
TURN_3_NAMES = [
    "credit_initial_consumption_usd_thousand",
    "credit_constant_consumption_equivalent_usd_thousand",
    "year_one_borrowing_probability",
]
TURN_4_NAMES = [
    "maximum_upfront_fee_usd_thousand",
    "fair_fee_year_two_expected_debt_usd_thousand",
]

variables = [
    Variable(
        "income_record_count",
        None,
        "Store eligible physical record count as an integer.",
    ),
    Variable(
        "low_income_usd_thousand",
        None,
        "Store 20th-percentile income in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "middle_income_usd_thousand",
        None,
        "Store 50th-percentile income in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "high_income_usd_thousand",
        None,
        "Store 80th-percentile income in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "crra_risk_aversion",
        None,
        "Store positive CRRA risk-aversion coefficient rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "cash_annual_effective_pct",
        None,
        "Store annual effective cash rate in percent rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "baseline_initial_consumption_usd_thousand",
        None,
        "Store optimal date-zero consumption without borrowing in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "baseline_constant_consumption_equivalent_usd_thousand",
        None,
        "Store four-date constant-consumption equivalent without borrowing in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "credit_initial_consumption_usd_thousand",
        None,
        "Store optimal date-zero consumption with the credit line in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "credit_constant_consumption_equivalent_usd_thousand",
        None,
        "Store four-date constant-consumption equivalent with the credit line in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "year_one_borrowing_probability",
        None,
        "Store unconditional probability of negative end-of-date-one assets as a decimal rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "maximum_upfront_fee_usd_thousand",
        None,
        "Store maximum welfare-equivalent upfront credit fee in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "fair_fee_year_two_expected_debt_usd_thousand",
        None,
        "Store unconditional expected positive end-of-date-two debt under the policy at the maximum fee in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


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
