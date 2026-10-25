"""Author A: spectral Riccati root, scalar disclosure search, sequential conditioning."""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from functools import lru_cache
from core.data import load_expansion_table


@lru_cache(None)
def inputs(scope="futures_only", ddof=1, start="2024-01-01", duration10=8.0):
    t = load_expansion_table("treasury_yield_curve").set_index("Date")[
        ["2 Yr", "5 Yr", "10 Yr"]
    ]
    c = load_expansion_table("cftc_cot")
    codes = ["042601", "044601", "043602"]
    c = c[
        c.report_scope.eq(scope)
        & c.CFTC_Contract_Market_Code.isin(codes)
        & c["Report_Date_as_YYYY-MM-DD"].between(start, "2024-12-31")
    ].copy()
    assert not c.duplicated(
        ["CFTC_Contract_Market_Code", "Report_Date_as_YYYY-MM-DD"]
    ).any()
    c["net"] = c.Lev_Money_Positions_Long_All - c.Lev_Money_Positions_Short_All
    n = c.pivot(
        index="Report_Date_as_YYYY-MM-DD",
        columns="CFTC_Contract_Market_Code",
        values="net",
    ).reindex(columns=codes)
    d = n.join(t, how="inner").dropna().sort_index()
    r = d[["2 Yr", "5 Yr", "10 Yr"]].to_numpy(float) / 100
    price = -np.diff(r, axis=0) * np.array([2.0, 4.5, duration10])
    flow = np.diff(d[codes].to_numpy(float), axis=0) * 0.1
    S = np.cov(price, rowvar=False, ddof=ddof)
    O = np.cov(flow, rowvar=False, ddof=ddof)
    return (
        S,
        O,
        {
            "dates": d.index.tolist(),
            "price_innovations": price.tolist(),
            "flow_innovations_million": flow.tolist(),
            "n": len(price),
        },
    )


def impact(S, O):
    e, Q = np.linalg.eigh(O)
    assert e.min() > 0 and np.linalg.eigvalsh(S).min() > 0
    W = (Q * np.sqrt(e)) @ Q.T
    Wi = (Q / np.sqrt(e)) @ Q.T
    a, U = np.linalg.eigh(W @ S @ W)
    L = 0.5 * Wi @ (U * np.sqrt(a)) @ U.T @ Wi
    return (L + L.T) / 2


def compute(S, O, n, target=0.6):
    L = impact(S, O)
    profit = float(np.trace(L @ O))
    D = np.diag(np.diag(S))

    def post(t):
        return np.linalg.inv(np.linalg.inv(S) + np.linalg.inv(t * D))

    def f(t):
        return float(np.trace(impact(post(t), O) @ O)) - target * profit

    lo = 1e-10
    hi = 1.0
    while f(hi) < 0:
        hi *= 2
    tt = brentq(f, lo, hi, xtol=1e-12)
    C = post(tt)
    Lc = impact(C, O)
    z = np.array([10.0, -5.0, 8.0]) / 10000
    flow = np.array([1.0, -0.5, 0.75]) * 1000
    mean = S @ np.linalg.solve(S + tt * D, z)
    final = mean + Lc @ flow
    # Independent direct conditional covariance using informed strategy.
    B = np.linalg.inv(2 * Lc)
    cross = C @ B.T
    V = C - cross @ np.linalg.solve(B @ C @ B.T + O, cross.T)
    vals = [
        n,
        np.sqrt(S[0, 0]) * 10000,
        np.sqrt(S[2, 2]) * 10000,
        np.sqrt(O[2, 2]) / 1000,
        L[0, 1] * 1e7,
        L[2, 2] * 1e7,
        profit,
        tt,
        np.sqrt(C[2, 2]) * 10000,
        Lc[0, 1] * 1e7,
        (1 + final[0]) * 100,
        (1 + final[2]) * 100,
        norm.cdf(-final[2] / np.sqrt(V[2, 2])),
    ]
    return vals, {
        "Sigma": S.tolist(),
        "Omega": O.tolist(),
        "impact": L.tolist(),
        "profit": profit,
        "disclosure_noise_multiplier": tt,
        "posterior_covariance": C.tolist(),
        "post_impact": Lc.tolist(),
        "public_mean_deviation": mean.tolist(),
        "final_mean_deviation": final.tolist(),
        "final_covariance": V.tolist(),
        "informed_strategy": B.tolist(),
        "riccati_relative": float(
            np.linalg.norm(4 * L @ O @ L - S) / np.linalg.norm(S)
        ),
        "post_riccati_relative": float(
            np.linalg.norm(4 * Lc @ O @ Lc - C) / np.linalg.norm(C)
        ),
        "camouflage_relative": float(
            np.linalg.norm(B @ C @ B.T - O) / np.linalg.norm(O)
        ),
        "posterior_partition_relative": float(
            np.linalg.norm(V - C / 2) / np.linalg.norm(C)
        ),
        "root_residual": f(tt),
        "conditions": [float(np.linalg.cond(v)) for v in [S, O, L, C, Lc]],
        "disclosure_bracket": [lo, hi],
        "domain_eigenvalues": [
            np.linalg.eigvalsh(v).tolist() for v in [S, O, L, C, Lc, V]
        ],
    }


from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = [
    "increment_count",
    "two_year_fundamental_sd_bp",
    "ten_year_fundamental_sd_bp",
    "ten_year_noise_sd_billion",
]

TURN_2_NAMES = [
    "two_year_from_five_year_impact",
    "ten_year_own_impact",
    "expected_insider_profit_million",
]

TURN_3_NAMES = [
    "disclosure_error_multiplier",
    "disclosed_ten_year_sd_bp",
    "disclosed_cross_impact",
]

TURN_4_NAMES = [
    "posterior_two_year_price_per100",
    "posterior_ten_year_price_per100",
    "ten_year_below_par_probability",
]

variables = [
    Variable(
        "increment_count",
        None,
        "Store number of retained consecutive increments as an integer.",
    ),
    Variable(
        "two_year_fundamental_sd_bp",
        None,
        "Store two-year fundamental standard deviation in basis points of par rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "ten_year_fundamental_sd_bp",
        None,
        "Store ten-year fundamental standard deviation in basis points of par rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "ten_year_noise_sd_billion",
        None,
        "Store ten-year noise-order standard deviation in USD billions of face rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "two_year_from_five_year_impact",
        None,
        "Store two-year price response to five-year flow in basis points of par per USD billion of face rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "ten_year_own_impact",
        None,
        "Store ten-year own-price response in basis points of par per USD billion of face rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "expected_insider_profit_million",
        None,
        "Store ex-ante expected informed-trader profit in USD millions rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "disclosure_error_multiplier",
        None,
        "Store public measurement-error covariance multiplier, dimensionless rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "disclosed_ten_year_sd_bp",
        None,
        "Store conditional ten-year fundamental standard deviation in basis points of par after public disclosure rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "disclosed_cross_impact",
        None,
        "Store post-disclosure two-year price response to five-year flow in basis points of par per USD billion of face rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "posterior_two_year_price_per100",
        None,
        "Store two-year posterior price per 100 of par rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "posterior_ten_year_price_per100",
        None,
        "Store ten-year posterior price per 100 of par rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "ten_year_below_par_probability",
        None,
        "Store conditional ten-year below-par probability as a decimal rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


def calculate(scope="futures_only", ddof=1, start="2024-01-01", duration10=8.0):
    S, O, e = inputs(scope, ddof, start, duration10)
    v, d = compute(S, O, e["n"])
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
