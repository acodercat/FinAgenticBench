"""One executable annuity-support policy across five SCF imputation worlds.

This is a hypothetical conversion of positive net worth to annual resources,
not observed retirement income or an insurance annuity quote. Each governed
SCF row weight is already one fifth of a final analysis weight; a per-world
absolute fiscal budget needs that fifth restored. At the USD 650bn budget the
mean-world policy breaches world 4's budget by USD 6.6613bn, despite meeting
its average limit. The common robust policy has a different allocation and
meets every world separately. Using one-fifth weights without restoring them
would understate world-4 cost by a factor of five; treating a Treasury par
quote as a spot discount rate changes all three annuity factors and the
optimized policy. Averaging five independently optimized policies is a
different decision than committing to one policy feasible in every world.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


CASE_ID = "scf_imputation_robust_annuity_support"
PAR_NODES = ("6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr")
PAR_TIMES = np.array([.5, 1, 2, 3, 5, 7, 10, 20], dtype=float)
BUDGET_BN = 650.0
TARGET_KUSD = 10.0


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("eligible_mean_population_mn", "Store the five-implicate mean eligible survey-weighted population in millions, rounded to 4 decimals."),
    _v("annuity_45_54_factor", "Store the 45–54 annual-payment present-value factor, rounded to 4 decimals."),
    _v("annuity_55_64_factor", "Store the 55–64 annual-payment present-value factor, rounded to 4 decimals."),
    _v("annuity_65_74_factor", "Store the 65–74 annual-payment present-value factor, rounded to 4 decimals."),
    _v("baseline_mean_shortfall_kusd", "Store the five-world mean eligible-household annual positive shortfall before any subsidy in USD thousands, rounded to 4 decimals."),
    _v("mean_policy_45_54_kusd", "Store the common mean-world 45–54 annual subsidy in USD thousands, rounded to 2 decimals."),
    _v("mean_policy_55_64_kusd", "Store the common mean-world 55–64 annual subsidy in USD thousands, rounded to 2 decimals."),
    _v("mean_policy_65_74_kusd", "Store the common mean-world 65–74 annual subsidy in USD thousands, rounded to 2 decimals."),
    _v("mean_policy_mean_squared_shortfall", "Store the five-world mean squared positive annual shortfall in squared USD thousands, rounded to 4 decimals, computed from the unrounded policy."),
    _v("mean_policy_worst_budget_world", "Store exactly one token from I1, I2, I3, I4, I5 for the greatest present-value policy cost."),
    _v("mean_policy_budget_overage_bn", "Store the largest world-specific cost above the budget in USD billions, rounded to 4 decimals, computed from the unrounded policy."),
    _v("mean_policy_worst_shortfall_world", "Store exactly one token from I1, I2, I3, I4, I5 for the greatest world-specific squared positive shortfall."),
    _v("robust_45_54_kusd", "Store the common robust 45–54 annual subsidy in USD thousands, rounded to 2 decimals."),
    _v("robust_55_64_kusd", "Store the common robust 55–64 annual subsidy in USD thousands, rounded to 2 decimals."),
    _v("robust_65_74_kusd", "Store the common robust 65–74 annual subsidy in USD thousands, rounded to 2 decimals."),
    _v("robust_worst_squared_shortfall", "Store the largest world-specific squared positive annual shortfall under the robust policy in squared USD thousands, rounded to 4 decimals, computed from the unrounded policy."),
    _v("robust_binding_budget_world", "Store exactly one token from I1, I2, I3, I4, I5 for the world with greatest present-value robust-policy cost."),
    _v("revised_45_54_kusd", "Store the revised robust 45–54 annual subsidy in USD thousands, rounded to 2 decimals."),
    _v("revised_55_64_kusd", "Store the revised robust 55–64 annual subsidy in USD thousands, rounded to 2 decimals."),
    _v("revised_65_74_kusd", "Store the revised robust 65–74 annual subsidy in USD thousands, rounded to 2 decimals."),
    _v("revised_worst_squared_shortfall", "Store the revised robust policy's largest world-specific squared positive annual shortfall in squared USD thousands, rounded to 4 decimals, computed from the unrounded policy."),
]
TURN_NAMES = [
    ["eligible_mean_population_mn", "annuity_45_54_factor", "annuity_55_64_factor", "annuity_65_74_factor", "baseline_mean_shortfall_kusd"],
    ["mean_policy_45_54_kusd", "mean_policy_55_64_kusd", "mean_policy_65_74_kusd", "mean_policy_mean_squared_shortfall"],
    ["mean_policy_worst_budget_world", "mean_policy_budget_overage_bn", "mean_policy_worst_shortfall_world"],
    ["robust_45_54_kusd", "robust_55_64_kusd", "robust_65_74_kusd", "robust_worst_squared_shortfall", "robust_binding_budget_world"],
    ["revised_45_54_kusd", "revised_55_64_kusd", "revised_65_74_kusd", "revised_worst_squared_shortfall"],
]
DECIMALS = [4, 4, 4, 4, 4, 2, 2, 2, 4, None, 4, None, 2, 2, 2, 4, None, 2, 2, 2, 4]


@lru_cache(maxsize=1)
def _discount_factors():
    curve = load_expansion_table("treasury_yield_curve")
    row = curve.loc[curve.Date.eq("2025-12-31")]
    if len(row) != 1:
        raise ValueError("Treasury valuation date is not unique")
    par = row.iloc[0][list(PAR_NODES)].to_numpy(float) / 100
    half_year_par = np.interp(np.arange(1, 41) / 2, PAR_TIMES, par)
    discount = np.empty(40)
    accumulated = 0.0
    for i, coupon in enumerate(half_year_par):
        discount[i] = (1 - coupon * accumulated / 2) / (1 + coupon / 2)
        accumulated += discount[i]
    if not np.all(np.isfinite(discount)) or np.any(discount <= 0) or np.any(np.diff(discount) >= 0):
        raise ValueError("Hypothetical bootstrapped discount curve invalid")
    return {years: float(discount[1:2*years:2].sum()) for years in (10, 15, 20)}


@lru_cache(maxsize=2)
def _worlds(periods):
    scf = load_expansion_table("fed_scf")
    factors = _discount_factors()
    group_factors = np.array([factors[x] for x in periods])
    worlds = []
    cost = np.zeros((5, 3))
    for implicate in range(1, 6):
        subset = scf.loc[
            scf.implicate.eq(implicate)
            & scf.reference_person_age.between(45, 74)
            & scf.net_worth_usd.lt(50_000)
        ].copy()
        group = np.searchsorted([55, 65], subset.reference_person_age.to_numpy(int), side="right")
        weight_mn = subset.survey_weight.to_numpy(float) * 5 / 1e6
        gap_kusd = TARGET_KUSD - np.maximum(subset.net_worth_usd.to_numpy(float), 0) / 1000 / group_factors[group]
        if len(subset) < 500 or not np.all(gap_kusd > 0) or not np.all(weight_mn > 0):
            raise ValueError("SCF eligible population or gaps changed")
        for band in range(3):
            cost[implicate-1, band] = weight_mn[group == band].sum() * group_factors[band]
        worlds.append((group, weight_mn, gap_kusd))
    if scf.household_id.nunique() != 4595 or len(scf) != 22975:
        raise ValueError("SCF frozen five-implicate structure changed")
    return group_factors, worlds, cost


def _scores(worlds, subsidy):
    scores = []
    derivatives = []
    for group, weight, gap in worlds:
        remaining = np.maximum(gap - subsidy[group], 0)
        total_weight = weight.sum()
        scores.append(float(np.dot(weight, remaining*remaining) / total_weight))
        derivatives.append(np.bincount(group, weights=-2*weight*remaining, minlength=3) / total_weight)
    return np.array(scores), np.array(derivatives)


def _mean_solve(worlds, cost):
    mean_coeff = cost.mean(axis=0)
    result = minimize(
        lambda u: float(_scores(worlds, u)[0].mean()),
        np.array([3.3, 4.3, 5.2]),
        jac=lambda u: _scores(worlds, u)[1].mean(axis=0),
        bounds=[(0, 10)]*3,
        constraints=[{"type": "ineq", "fun": lambda u: BUDGET_BN - mean_coeff@u, "jac": lambda u: -mean_coeff}],
        method="SLSQP", options={"ftol": 1e-11, "maxiter": 2000},
    )
    if not result.success or not np.all(np.isfinite(result.x)) or mean_coeff@result.x > BUDGET_BN + 1e-6:
        raise ValueError(f"Mean-world policy did not converge: {result.message}")
    return result.x


def _robust_solve(worlds, cost, start):
    initial = np.r_[start, max(_scores(worlds, start)[0])]
    result = minimize(
        lambda z: float(z[3]), initial,
        jac=lambda z: np.array([0., 0., 0., 1.]),
        bounds=[(0, 10)]*3 + [(0, None)],
        constraints=[
            {"type": "ineq", "fun": lambda z: BUDGET_BN - cost@z[:3],
             "jac": lambda z: np.c_[-cost, np.zeros(5)]},
            {"type": "ineq", "fun": lambda z: z[3] - _scores(worlds, z[:3])[0],
             "jac": lambda z: np.c_[-_scores(worlds, z[:3])[1], np.ones(5)]},
        ],
        method="SLSQP", options={"ftol": 1e-10, "maxiter": 2000},
    )
    if (not result.success or not np.all(np.isfinite(result.x))
            or np.max(cost@result.x[:3]) > BUDGET_BN + 1e-6
            or np.max(_scores(worlds, result.x[:3])[0]) > result.x[3] + 1e-7):
        raise ValueError(f"Five-world robust policy did not converge: {result.message}")
    return result.x[:3]


@lru_cache(maxsize=1)
def analysis():
    initial_factors, initial_worlds, initial_cost = _worlds((20, 15, 10))
    mean_policy = _mean_solve(initial_worlds, initial_cost)
    mean_scores, _ = _scores(initial_worlds, mean_policy)
    mean_costs = initial_cost @ mean_policy
    robust_policy = _robust_solve(initial_worlds, initial_cost, mean_policy)
    robust_scores, _ = _scores(initial_worlds, robust_policy)
    robust_costs = initial_cost @ robust_policy
    _, revised_worlds, revised_cost = _worlds((15, 15, 10))
    revised_policy = _robust_solve(revised_worlds, revised_cost, robust_policy)
    revised_scores, _ = _scores(revised_worlds, revised_policy)
    revised_costs = revised_cost @ revised_policy
    if not (np.argmax(mean_costs) == 3 and np.argmax(robust_costs) == 3
            and np.argmax(mean_scores) == 0 and np.argmax(robust_scores) == 0
            and np.argmax(revised_scores) == 0 and np.argmax(revised_costs) == 3):
        raise ValueError("Frozen five-world ordering changed")
    baseline = np.mean([
        np.dot(weight, gap) / weight.sum() for _, weight, gap in initial_worlds
    ])
    return {
        "eligible_mean_population_mn": float(np.mean([weight.sum() for _, weight, _ in initial_worlds])),
        "annuity_45_54_factor": float(initial_factors[0]),
        "annuity_55_64_factor": float(initial_factors[1]),
        "annuity_65_74_factor": float(initial_factors[2]),
        "baseline_mean_shortfall_kusd": float(baseline),
        "mean_policy_45_54_kusd": float(mean_policy[0]),
        "mean_policy_55_64_kusd": float(mean_policy[1]),
        "mean_policy_65_74_kusd": float(mean_policy[2]),
        "mean_policy_mean_squared_shortfall": float(mean_scores.mean()),
        "mean_policy_worst_budget_world": "I" + str(np.argmax(mean_costs) + 1),
        "mean_policy_budget_overage_bn": float(max(0.0, max(mean_costs) - BUDGET_BN)),
        "mean_policy_worst_shortfall_world": "I" + str(np.argmax(mean_scores) + 1),
        "robust_45_54_kusd": float(robust_policy[0]),
        "robust_55_64_kusd": float(robust_policy[1]),
        "robust_65_74_kusd": float(robust_policy[2]),
        "robust_worst_squared_shortfall": float(max(robust_scores)),
        "robust_binding_budget_world": "I" + str(np.argmax(robust_costs) + 1),
        "revised_45_54_kusd": float(revised_policy[0]),
        "revised_55_64_kusd": float(revised_policy[1]),
        "revised_65_74_kusd": float(revised_policy[2]),
        "revised_worst_squared_shortfall": float(max(revised_scores)),
    }


@lru_cache(maxsize=1)
def ground_truth():
    state = analysis()
    return tuple(state[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = analysis()
    precision = dict(zip((v.name for v in variables), DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[n] for n in names], [truth[n] for n in names],
        [precision[n] for n in names],
    )


def validate(outputs):
    return _validate_subset(outputs, [v.name for v in variables])


validators = {
    f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _validate_subset(outputs, names))
    for i, names in enumerate(TURN_NAMES, start=1)
}
