"""Author A: cluster sufficient-moment polynomial updates and normal-equation GMM."""

import numpy as np, pandas as pd
from scipy.stats import norm
from core.data import load_expansion_table
from functools import lru_cache
from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator


@lru_cache(maxsize=None)
def inputs(rate_type="SOFR", end="2024-12-31", positive=True):
    f = load_expansion_table("fdic_bankfind")
    f = f[f.REPDTE.between("2021-01-01", end)]
    assert not f.duplicated(["CERT", "REPDTE"]).any()
    panel = f.pivot(index="CERT", columns="REPDTE", values="NIMY").dropna()
    panel = panel.loc[(panel > 0).all(axis=1)] if positive else panel
    Y = panel.to_numpy(dtype=float)
    r = load_expansion_table("nyfed_reference_rates")
    r = r[
        r["Rate Type"].eq(rate_type) & r["Effective Date"].between("2021-01-01", end)
    ].copy()
    assert not r["Effective Date"].duplicated().any()
    r["q"] = pd.to_datetime(r["Effective Date"]).dt.to_period("Q")
    R = r.groupby("q")["Rate (%)"].mean().to_numpy(dtype=float)
    assert len(R) == Y.shape[1] and np.isfinite(Y).all() and np.isfinite(R).all()
    return Y, R, panel.index.astype(str).tolist()


def compute(
    Y,
    R,
    lags=(2, 3, 4),
    one_identity=False,
    center=False,
    freeze_jackknife_weight=False,
):
    N, T = Y.shape
    D = Y[:, 2:] - Y[:, 1:-1]
    X = np.stack(
        [Y[:, 1:-1] - Y[:, :-2], np.broadcast_to(np.diff(R)[1:], D.shape)], axis=2
    )
    Z = np.zeros((N, T - 2, len(lags) + 1))
    for k, lag in enumerate(lags):
        for t in range(2, T):
            if t - lag >= 0:
                Z[:, t - 2, k] = Y[:, t - lag]
    Z[:, :, -1] = np.diff(R)[1:]
    H = (
        np.eye(T - 2)
        if one_identity
        else 2 * np.eye(T - 2) - np.eye(T - 2, k=1) - np.eye(T - 2, k=-1)
    )
    A = np.einsum("ntj,ntk->njk", Z, X)
    b = np.einsum("ntj,nt->nj", Z, D)
    h = np.einsum("ntj,tu,nuk->njk", Z, H, Z)
    SA = A.sum(axis=0)
    Sb = b.sum(axis=0)
    SH = h.sum(axis=0)
    # Polynomial sum of cluster moment outer products, evaluated at each one-step coefficient.
    bb = np.einsum("ni,nj->ij", b, b)
    ba = np.einsum("ni,njk->ijk", b, A)
    aa = np.einsum("nik,njl->ijkl", A, A)

    def fit(G, v, h, drop=None):
        np.linalg.cholesky(h)
        W = np.linalg.inv(h)
        b1 = np.linalg.solve(G.T @ W @ G, G.T @ W @ v)
        S = (
            bb
            - np.einsum("ijk,k->ij", ba, b1)
            - np.einsum("jik,k->ij", ba, b1)
            + np.einsum("ijkl,k,l->ij", aa, b1, b1)
        )
        if drop is not None:
            gi = b[drop] - A[drop] @ b1
            S -= np.outer(gi, gi)
        if center:
            g = v - G @ b1
            S -= np.outer(g, g) / (N - (drop is not None))
        np.linalg.cholesky(S)
        W2 = np.linalg.inv(S)
        b2 = np.linalg.solve(G.T @ W2 @ G, G.T @ W2 @ v)
        return b1, b2, S, W2

    b1, b2, S, W = fit(SA, Sb, SH)
    jk = []
    mins = []
    for i in range(N):
        if freeze_jackknife_weight:
            g = SA - A[i]
            v = Sb - b[i]
            jk.append(np.linalg.solve(g.T @ W @ g, g.T @ W @ v))
        else:
            jk.append(fit(SA - A[i], Sb - b[i], SH - h[i], i)[1])
    jk = np.array(jk)
    C = (N - 1) / N * (jk - jk.mean(axis=0)).T @ (jk - jk.mean(axis=0))
    se = np.sqrt(np.diag(C))
    z = norm.ppf(0.975)
    d = 1 - b2[0]
    v = b2[1]
    poly = np.array(
        [
            d * d - z * z * C[0, 0],
            -2 * (d * v + z * z * C[0, 1]),
            v * v - z * z * C[1, 1],
        ]
    )
    disc = poly[1] ** 2 - 4 * poly[0] * poly[2]
    assert np.isfinite(jk).all() and disc > 0
    bounds = np.sort(np.roots(poly))
    kind = "BOUNDED_INTERVAL" if poly[0] > 0 else "DISJOINT_RAYS"
    g = Sb - SA @ b2
    values = [
        N,
        *b1,
        *b2,
        float(g @ W @ g),
        *se,
        float(C[0, 1] / np.prod(se)),
        kind,
        *bounds,
    ]
    return values, {
        "one": b1.tolist(),
        "two": b2.tolist(),
        "cov": C.tolist(),
        "jackknife": jk.tolist(),
        "condition_one": float(np.linalg.cond(SA.T @ np.linalg.solve(SH, SA))),
        "condition_two": float(np.linalg.cond(SA.T @ W @ SA)),
        "weight_eigenvalues": np.linalg.eigvalsh(S).tolist(),
        "normal_equation_relative": float(
            np.linalg.norm(SA.T @ W @ g) / (1 + np.linalg.norm(SA.T @ W @ Sb))
        ),
        "fieller_polynomial": poly.tolist(),
        "fieller_discriminant": float(disc),
        "fieller_endpoint_residual": float(max(abs(np.polyval(poly, bounds)))),
        "minimum_one_minus_phi_abs": float(np.min(abs(1 - jk[:, 0]))),
    }


TURN_1_NAMES = ["bank_count", "one_step_persistence", "one_step_sofr_coefficient"]

TURN_2_NAMES = [
    "two_step_persistence",
    "two_step_sofr_coefficient",
    "hansen_j_statistic",
]

TURN_3_NAMES = [
    "jackknife_persistence_se",
    "jackknife_sofr_se",
    "jackknife_coefficient_correlation",
]

TURN_4_NAMES = [
    "fieller_set_topology",
    "fieller_lower_finite_boundary",
    "fieller_upper_finite_boundary",
]

variables = [
    Variable("bank_count", None, "Store bank count as an integer."),
    Variable(
        "one_step_persistence",
        None,
        "Store one-step lagged-margin coefficient, dimensionless rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "one_step_sofr_coefficient",
        None,
        "Store one-step SOFR coefficient in margin percentage points per SOFR percentage point rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "two_step_persistence",
        None,
        "Store two-step lagged-margin coefficient, dimensionless rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "two_step_sofr_coefficient",
        None,
        "Store two-step SOFR coefficient in margin percentage points per SOFR percentage point rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "hansen_j_statistic",
        None,
        "Store Hansen overidentification statistic, dimensionless rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "jackknife_persistence_se",
        None,
        "Store persistence coefficient standard error, dimensionless rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "jackknife_sofr_se",
        None,
        "Store SOFR coefficient standard error in margin percentage points per SOFR percentage point rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "jackknife_coefficient_correlation",
        None,
        "Store sampling correlation of persistence and SOFR coefficients, dimensionless rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "fieller_set_topology",
        None,
        "Store Fieller confidence-set topology as an exact canonical string token from {BOUNDED_INTERVAL, DISJOINT_RAYS, ALL_REAL, HALF_LINE, EMPTY_SET}.",
    ),
    Variable(
        "fieller_lower_finite_boundary",
        None,
        "Store smaller finite Fieller boundary in margin percentage points per SOFR percentage point rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "fieller_upper_finite_boundary",
        None,
        "Store larger finite Fieller boundary in margin percentage points per SOFR percentage point rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 4, 4, None, 4, 4]


def calculate(
    rate_type="SOFR", lags=(2, 3, 4), one_identity=False, freeze_jackknife_weight=False
):
    y, r, _ = inputs(rate_type=rate_type)
    v, e = compute(
        y,
        r,
        lags=lags,
        one_identity=one_identity,
        freeze_jackknife_weight=freeze_jackknife_weight,
    )
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
