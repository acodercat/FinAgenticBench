"""Author A: Marshallian state-price equilibrium, implicit Borch derivatives."""

import numpy as np
from scipy.optimize import root
from core.data import load_expansion_table
from functools import lru_cache
from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator


@lru_cache(maxsize=None)
def inputs(year=2024, net=False, pooled_implicates=False):
    s = load_expansion_table("fed_scf")
    s = s[s.reference_person_age.between(35, 64) & s.net_worth_usd.gt(0)]
    qs = []
    groups = [(0, s)] if pooled_implicates else s.groupby("implicate")
    for imp, g in groups:
        g = g.sort_values("net_worth_usd")
        qs.append(
            g.net_worth_usd.to_numpy(dtype=float)[
                np.searchsorted(
                    g.survey_weight.cumsum().to_numpy(dtype=float),
                    g.survey_weight.sum() * np.array([0.25, 0.5, 0.75]),
                )
            ]
            / 1000
        )
    wealth = np.mean(qs, axis=0)
    f = load_expansion_table("fema_nfip")
    f = f[
        f.state_usps.eq("FL") & f.date_of_loss.between(f"{year}-01-01", f"{year}-12-31")
    ]
    prefix = "net" if net else "gross"
    cols = [f"{prefix}_{x}_payment_usd" for x in ["building", "contents", "icc"]]
    loss = f[cols].sum(axis=1, min_count=3)
    loss = loss[loss.gt(0)]
    sev = np.quantile(loss, [0.25, 0.5, 0.75, 0.95], method="inverted_cdf") / 1000
    return (
        wealth,
        sev,
        {
            "SCF_rows": len(s),
            "SCF_households": s.household_id.nunique(),
            "implicate_quantiles": np.array(qs).tolist(),
            "NFIP_complete_positive_count": len(loss),
        },
    )


def compute(wealth, sev, loss_fraction=0.25, gamma=(0.5, 0.75, 1), initial=None):
    gam = np.array(gamma)
    pi = np.ones(4) / 4
    end = wealth[:, None] - np.array([0.5, 0.3, 0.2])[:, None] * (loss_fraction * sev)
    assert end.min() > 0

    def demand(q):
        d = (pi[None, :] / q[None, :]) ** (1 / gam[:, None])
        return (end @ q)[:, None] * d / (d @ q)[:, None]

    def fun(v):
        q = np.r_[np.exp(v), 1]
        q /= q.sum()
        return ((demand(q).sum(axis=0) - end.sum(axis=0)) / end.sum(axis=0))[:3]

    r = root(fun, np.zeros(3) if initial is None else initial, tol=1e-11)
    assert np.linalg.norm(fun(r.x), ord=np.inf) < 1e-11
    q = np.r_[np.exp(r.x), 1]
    q /= q.sum()
    c = demand(q)
    shares = (c / gam[:, None]) / (c / gam[:, None]).sum(axis=0)

    def ce(x, g):
        return float(
            np.exp(np.mean(np.log(x)))
            if g == 1
            else np.mean(x ** (1 - g)) ** (1 / (1 - g))
        )

    total = c.sum(axis=0)
    scale = total.mean()
    z = (total - scale) / total.std()
    weights = (c[:, 0] / scale) ** gam
    C = np.zeros((3, 4, 4))
    C[0, :, 0] = 1
    C[0, :, 2] = z
    C[1, :, 1] = 1
    C[1, :, 3] = z
    C[2] = -C[0] - C[1]
    base = np.zeros((3, 4))
    base[2] = total / scale
    start = np.array(
        [
            c[0].mean() / scale,
            c[1].mean() / scale,
            np.cov(c[0], z, ddof=0)[0, 1] / scale,
            np.cov(c[1], z, ddof=0)[0, 1] / scale,
        ]
    )

    def grad(a):
        x = base + np.einsum("isk,k->is", C, a)
        return np.einsum("is,isk->k", weights[:, None] * x ** (-gam[:, None]), C) / 4

    rr = root(grad, start, tol=1e-11)
    assert np.max(abs(grad(rr.x))) < 1e-11
    x = base + np.einsum("isk,k->is", C, rr.x)
    quota = np.r_[
        rr.x[2:] * scale / total.std(), 1 - rr.x[2:].sum() * scale / total.std()
    ]
    assert x.min() > 0 and quota.min() > 0 and quota.max() < 1
    hess = (
        np.einsum(
            "is,isk,isl->kl",
            weights[:, None] * gam[:, None] * x ** (-gam[:, None] - 1),
            C,
            C,
        )
        / 4
    )

    def util(a):
        return sum(
            weights[i]
            * np.mean(
                np.log(a[i]) if gam[i] == 1 else a[i] ** (1 - gam[i]) / (1 - gam[i])
            )
            for i in range(3)
        )

    loss = util(c / scale) - util(x)
    assert loss > -1e-12
    v = [
        *wealth,
        sev[-1],
        q[-1],
        c[0, -1],
        c[2, 0],
        shares[0, 0] * 100,
        shares[0, -1] * 100,
        ce(c[0], gam[0]),
        quota[0] * 100,
        quota[1] * 100,
        x[0, -1] * scale,
    ]
    mu = pi[None, :] * c ** (-gam[:, None]) / q[None, :]
    return v, {
        "q": q.tolist(),
        "endowments": end.tolist(),
        "allocation": c.tolist(),
        "shares": shares.tolist(),
        "budget_residual": float(np.max(abs((c - end) @ q))),
        "market_residual": float(np.max(abs((c - end).sum(axis=0)))),
        "FOC_relative": float(np.max(np.ptp(mu, axis=1) / mu.mean(axis=1))),
        "quota": quota.tolist(),
        "restricted_allocation": (x * scale).tolist(),
        "fixed_Pareto_weights_normalized_units": weights.tolist(),
        "restricted_stationarity": float(np.max(abs(grad(rr.x)))),
        "restricted_hessian_condition": float(np.linalg.cond(hess)),
        "restricted_hessian_eigenvalues": np.linalg.eigvalsh(hess).tolist(),
        "restricted_welfare_loss": float(loss),
        "minimum_endowment": float(end.min()),
        "minimum_consumption": float(c.min()),
        "autarky_CE": [ce(end[i], gam[i]) for i in range(3)],
        "equilibrium_CE": [ce(c[i], gam[i]) for i in range(3)],
        "solver_success": bool(r.success),
        "price_coordinate_root": r.x.tolist(),
    }


TURN_1_NAMES = [
    "member_one_resources_thousand",
    "member_two_resources_thousand",
    "member_three_resources_thousand",
    "gross_claim_p95_thousand",
]

TURN_2_NAMES = [
    "largest_loss_state_price",
    "member_one_equilibrium_worst_thousand",
    "member_three_equilibrium_best_thousand",
]

TURN_3_NAMES = [
    "member_one_best_marginal_share_pct",
    "member_one_worst_marginal_share_pct",
    "member_one_equilibrium_ce_thousand",
]

TURN_4_NAMES = [
    "restricted_member_one_quota_pct",
    "restricted_member_two_quota_pct",
    "restricted_member_one_worst_thousand",
]

variables = [
    Variable(
        "member_one_resources_thousand",
        None,
        "Store member one initial resources in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "member_two_resources_thousand",
        None,
        "Store member two initial resources in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "member_three_resources_thousand",
        None,
        "Store member three initial resources in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "gross_claim_p95_thousand",
        None,
        "Store 95th-percentile gross claim payment in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "largest_loss_state_price",
        None,
        "Store price of the largest-loss one-unit state claim in bond-numeraire units rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "member_one_equilibrium_worst_thousand",
        None,
        "Store member one equilibrium terminal wealth in the largest-loss state in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "member_three_equilibrium_best_thousand",
        None,
        "Store member three equilibrium terminal wealth in the smallest-loss state in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "member_one_best_marginal_share_pct",
        None,
        "Store member one marginal resource share in the smallest-loss state in percent rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "member_one_worst_marginal_share_pct",
        None,
        "Store member one marginal resource share in the largest-loss state in percent rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "member_one_equilibrium_ce_thousand",
        None,
        "Store member one equilibrium certainty-equivalent terminal wealth in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "restricted_member_one_quota_pct",
        None,
        "Store optimal restricted member one quota share in percent rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "restricted_member_two_quota_pct",
        None,
        "Store optimal restricted member two quota share in percent rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "restricted_member_one_worst_thousand",
        None,
        "Store member one restricted-treaty terminal wealth in the largest-loss state in USD thousands rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


def calculate(year=2024, net=False, pooled_implicates=False, loss_fraction=0.25):
    w, s, _ = inputs(year, net, pooled_implicates)
    v, e = compute(w, s, loss_fraction=loss_fraction)
    return dict(zip((x.name for x in variables), v)), e


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
