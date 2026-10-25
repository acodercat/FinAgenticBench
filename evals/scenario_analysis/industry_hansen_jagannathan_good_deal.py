"""Formula-free hard case; one strict PV channel, private invariants. Raw certification: tests/test_raw_certification_quant_cases.py rebuilds this case's inputs from the source-native archive and asserts they match the governed runtime tables value for value."""

from functools import lru_cache
import numpy as np
from scipy.optimize import minimize
from core.data import load_ken_french_table, load_expansion_table
from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator


@lru_cache(None)
def hj_inputs(end="2024-03-31", cash="effective"):
    f = load_ken_french_table("daily_industry_returns")
    cols = [
        k + "_pct"
        for k in [
            "nodur",
            "durbl",
            "manuf",
            "enrgy",
            "hitec",
            "telcm",
            "shops",
            "hlth",
            "utils",
            "other",
        ]
    ]
    d = f[f.date.between("2024-01-01", end)].dropna(subset=cols).sort_values("date")
    x = d[cols].to_numpy(float) / 100
    rr = load_expansion_table("nyfed_reference_rates")
    q = rr[
        rr["Rate Type"].eq("SOFR") & rr["Effective Date"].between("2024-01-01", end)
    ].sort_values("Effective Date")
    rate = q["Rate (%)"].astype(float).mean() / 100
    rf = (1 + rate) ** (1 / 252) if cash == "effective" else 1 + rate / 252
    assert not d.date.duplicated().any() and not q["Effective Date"].duplicated().any()
    assert np.isfinite(x).all() and q["Rate (%)"].notna().all()
    return x, rf, d, q


def hj_a(end="2024-03-31", cash="effective", positivity=True, dispersion=1.5):
    x, rf, d, q = hj_inputs(end, cash)
    n = len(x)
    z = np.c_[np.ones(n), (1 + x - rf) * 100]
    b = np.r_[1 / rf, np.zeros(10)]
    coef = np.linalg.solve(z.T @ z / n, b)
    m = z @ coef

    # Semismooth convex dual: exact active-set Newton with safeguarded line search.
    def objective(a):
        v = np.maximum(z @ a, 0)
        return 0.5 * v @ v / n - b @ a

    a = coef.copy()
    for iteration in range(100):
        v = np.maximum(z @ a, 0)
        grad = z.T @ v / n - b
        if np.max(abs(grad)) < 2e-13:
            break
        active = z @ a > 0
        direction = np.linalg.solve(z[active].T @ z[active] / n, grad)
        step = 1.0
        while (
            objective(a - step * direction)
            > objective(a) - 1e-4 * step * grad @ direction
        ):
            step *= 0.5
        a -= step * direction
    else:
        raise RuntimeError("HJ dual failed")
    mp = np.maximum(z @ a, 0) if positivity else m.copy()
    claim = np.maximum(-x.mean(1), 0) * 100
    digital = claim > 0
    cap = 1 / rf**2 + (dispersion * mp.std(ddof=0)) ** 2
    bounds = []
    cert = []
    for sign in [1, -1]:
        opt = minimize(
            lambda v: sign * v @ claim / n,
            mp,
            jac=lambda v: sign * claim / n,
            bounds=[(0, None)] * n if positivity else None,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda v: z.T @ v / n - b,
                    "jac": lambda v: z.T / n,
                },
                {
                    "type": "ineq",
                    "fun": lambda v: cap - v @ v / n,
                    "jac": lambda v: -2 * v / n,
                },
            ],
            method="SLSQP",
            options={"ftol": 1e-13, "maxiter": 1500},
        )
        if not opt.success:
            raise RuntimeError(opt.message)
        v = opt.x
        active = v > 1e-8 if positivity else np.ones(n, bool)
        dual = np.linalg.lstsq(
            np.c_[z[active], v[active]], -sign * claim[active], rcond=None
        )[0]
        station = sign * claim + z @ dual[:-1] + dual[-1] * v
        cert.append(
            dict(
                pricing=float(max(abs(z.T @ v / n - b))),
                norm=float(abs(v @ v / n - cap)),
                stationarity=float(max(abs(station[active]))),
                inactive_margin=float(min(station[~active]))
                if (~active).any()
                else None,
                eta=float(dual[-1]),
                iterations=opt.nit,
                state=v,
            )
        )
        bounds.append(float(v @ claim / n))
    out = dict(
        scenario_count=n,
        kernel_mean=1 / rf,
        unrestricted_kernel_sd=float(m.std()),
        unrestricted_minimum_kernel=float(min(m)),
        nonnegative_kernel_sd=float(mp.std()),
        maximum_nonnegative_kernel=float(max(mp)),
        zero_kernel_states=int(sum(mp < 1e-10)),
        downside_claim_price_usd=float(mp @ claim / n),
        downside_digital_price_usd=float(mp @ digital / n),
        good_deal_lower_usd=bounds[0],
        good_deal_upper_usd=bounds[1],
    )
    return out, dict(
        z=z,
        b=b,
        m=m,
        mp=mp,
        dual=a,
        claim=claim,
        cap=cap,
        certificates=cert,
        pricing=float(max(abs(z.T @ mp / n - b))),
        condition=float(np.linalg.cond(z)),
        dual_iterations=iteration,
        minimum_active_kernel=float(min(mp[mp > 1e-10])),
        inactive_dual_margin=float(-max((z @ a)[mp < 1e-10]))
        if (mp < 1e-10).any()
        else None,
    )


TURN_1_NAMES = [
    "scenario_count",
    "kernel_mean",
    "unrestricted_kernel_sd",
    "unrestricted_minimum_kernel",
]
TURN_2_NAMES = [
    "nonnegative_kernel_sd",
    "maximum_nonnegative_kernel",
    "zero_kernel_states",
]
TURN_3_NAMES = ["downside_claim_price_usd", "downside_digital_price_usd"]
TURN_4_NAMES = ["good_deal_lower_usd", "good_deal_upper_usd"]

variables = [
    Variable("scenario_count", None, "Store scenario count as an integer."),
    Variable(
        "kernel_mean",
        None,
        "Store dimensionless kernel mean rounded to 6 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "unrestricted_kernel_sd",
        None,
        "Store dimensionless population standard deviation of the unrestricted kernel rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "unrestricted_minimum_kernel",
        None,
        "Store minimum signed dimensionless unrestricted kernel value rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "nonnegative_kernel_sd",
        None,
        "Store dimensionless population standard deviation of the nonnegative kernel rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "maximum_nonnegative_kernel",
        None,
        "Store largest dimensionless nonnegative kernel value rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "zero_kernel_states", None, "Store number of zero-kernel states as an integer."
    ),
    Variable(
        "downside_claim_price_usd",
        None,
        "Store downside claim price in USD for the USD100 portfolio rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "downside_digital_price_usd",
        None,
        "Store USD1 downside digital price in USD rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "good_deal_lower_usd",
        None,
        "Store sharp lower good-deal price in USD rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "good_deal_upper_usd",
        None,
        "Store sharp upper good-deal price in USD rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [0, 6, 4, 4, 4, 4, 0, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def ground_truth():
    out = hj_a()[0]
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
