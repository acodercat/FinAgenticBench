"""Four pension books, an actuarial-basis comparison, and a stress applied to the committed program.

Four plans are drawn from matched Form 5500 and Schedule SB filings, funded at current
asset values, and given a max-min cash injection. A comparison reruns the allocation on
actuarial asset values and is projected five years; the stress, the plan-by-plan
shortfall and the restoration reserve then return to the original program, with its own
current-value assets and its own grants.

Rejected alternative conventions, each measured against the pinned reading:

- Stressing the program from actuarial instead of current asset values raises the
  fifth-year minimum funded ratio from 59.8588 to 69.3103 percent and lowers the
  restoration reserve from 38,840.5620 to 28,575.6195 million.
- Stressing the original assets with the comparison's grants moves the fifth-year
  minimum funded ratio from 59.8588 to 59.9045 percent.
- Withdrawing expenses at the beginning rather than the end of each year lowers the
  comparison's fifth-year minimum funded ratio from 76.4639 to 74.5788 percent.
- Projecting four years instead of five raises the comparison's minimum funded ratio to
  82.9095 percent and changes 7 of the 15 outputs.
"""

from functools import lru_cache
import numpy as np
from core.data import load_expansion_table


@lru_cache(None)
def inputs():
    s = load_expansion_table("dol_form5500_schedule_sb")
    f = load_expansion_table("dol_form5500_financials")
    r = load_expansion_table("treasury_yield_curve")
    s = s[
        s.plan_year_end.eq("2024-12-31")
        & s.valuation_date.eq("2024-01-01")
        & s.filing_status.eq("FILING_RECEIVED")
        & s.date_received.le("2025-12-31")
    ]
    q = s.merge(
        f[f.filing_status.eq("FILING_RECEIVED")][["filing_id", "total_expenses_usd"]],
        on="filing_id",
        validate="one_to_one",
    )
    q = q.sort_values(["date_received", "filing_id"]).drop_duplicates(
        ["sponsor_ein", "plan_number", "plan_year_end"], keep="last"
    )
    cols = [
        "current_value_assets_usd",
        "actuarial_value_assets_usd",
        "total_funding_target_usd",
        "total_expenses_usd",
    ]
    q = q[q[cols].notna().all(axis=1) & q[cols].gt(0).all(axis=1)]
    z = q.sort_values(["total_funding_target_usd", "filing_id"], ascending=[False, True]).head(4)
    assert len(z) == 4
    rr = r[r.Date.eq("2024-12-31")]
    assert len(rr) == 1
    return z, len(q), float(rr["5 Yr"].iloc[0]) / 100


def allocate(a, l, budget):
    order = np.argsort(a / l)
    level = None
    for j in range(1, len(a) + 1):
        ix = order[:j]
        candidate = (budget + a[ix].sum()) / l[ix].sum()
        if j == len(a) or candidate <= a[order[j]] / l[order[j]]:
            level = candidate
            break
    g = np.maximum(level * l - a, 0)
    assert abs(g.sum() - budget) < 1e-7
    return g, float(level)


def project(a, l, e, g, r, years=5, timing="end", topup=False):
    assets = a + g
    target = l.copy()
    calls = []
    states = []
    for j in range(1, years + 1):
        expense = e * 1.03 ** (j - 1)
        assets = assets * (1 + r) - expense if timing == "end" else (assets - expense) * (1 + r)
        target = target * 1.02
        c = np.maximum(target - assets, 0) if topup else np.zeros(len(a))
        assets += c
        calls.append(float(c.sum()))
        states.append(assets.copy())
    assert np.all(assets > 0)
    return assets, target, np.array(calls), states


def calculate(asset_state="current", grant_state="original", expense_timing="end", horizon=5):
    z, count, r = inputs()
    a = z.current_value_assets_usd.to_numpy(float) / 1e6
    v = z.actuarial_value_assets_usd.to_numpy(float) / 1e6
    l = z.total_funding_target_usd.to_numpy(float) / 1e6
    e = z.total_expenses_usd.to_numpy(float) / 1e6
    B = 0.04 * l.sum()
    g, level = allocate(a, l, B)
    gv, lv = allocate(v, l, B)
    winner = sorted(range(4), key=lambda i: (-g[i], str(z.filing_id.iloc[i])))[0]
    alt, t, _, _ = project(v, l, e, gv, r, horizon, expense_timing)
    act = a if asset_state == "current" else v
    grant = g if grant_state == "original" else gv
    stressed, ts, _, _ = project(act, l, e, grant, r - 0.015, horizon, expense_timing)
    restored, tr, calls, states = project(
        act, l, e, grant, r - 0.015, horizon, expense_timing, True
    )
    out = dict(
        eligible_plan_count=count,
        largest_target_filing=str(z.filing_id.iloc[0]),
        original_aggregate_funded_pct=a.sum() / l.sum() * 100,
        original_maxmin_funded_pct=level * 100,
        largest_grant_filing=str(z.filing_id.iloc[winner]),
        largest_grant_million=float(g[winner]),
        actuarial_maxmin_funded_pct=lv * 100,
        actuarial_grant_to_original_leader_million=float(gv[winner]),
        actuarial_year5_min_funded_pct=float(min(alt / t)) * 100,
        actuarial_year5_shortfall_million=float(np.maximum(t - alt, 0).sum()),
        original_stress_year5_min_funded_pct=float(min(stressed / ts)) * 100,
        original_stress_year5_shortfall_million=float(np.maximum(ts - stressed, 0).sum()),
        restoration_reserve_pv_million=float(calls @ (1 + r) ** (-np.arange(1, horizon + 1))),
        largest_restoration_call_million=float(max(calls)),
        largest_restoration_call_year=int(np.argmax(calls) + 1),
    )
    return out, dict(
        ids=z.filing_id.tolist(),
        current=a.tolist(),
        actuarial=v.tolist(),
        target=l.tolist(),
        expenses=e.tolist(),
        grants=g.tolist(),
        alternative_grants=gv.tolist(),
        budget=B,
        benchmark_rate=r,
        calls=calls.tolist(),
        restored_assets=restored.tolist(),
        terminal_target=tr.tolist(),
        budget_residual=float(g.sum() - B),
        floor_slacks=((a + g) / l - level).tolist(),
        active_set=np.where(g > 0)[0].tolist(),
    )


from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = ["eligible_plan_count", "largest_target_filing", "original_aggregate_funded_pct"]
TURN_2_NAMES = ["original_maxmin_funded_pct", "largest_grant_filing", "largest_grant_million"]
TURN_3_NAMES = ["actuarial_maxmin_funded_pct", "actuarial_grant_to_original_leader_million"]
TURN_4_NAMES = ["actuarial_year5_min_funded_pct", "actuarial_year5_shortfall_million"]
TURN_5_NAMES = ["original_stress_year5_min_funded_pct", "original_stress_year5_shortfall_million"]
TURN_6_NAMES = [
    "restoration_reserve_pv_million",
    "largest_restoration_call_million",
    "largest_restoration_call_year",
]

variables = [
    Variable(
        "eligible_plan_count",
        None,
        "Store eligible plan count before selecting four as an integer.",
    ),
    Variable(
        "largest_target_filing",
        None,
        "Store largest funding target’s filing identifier exactly as a source string.",
    ),
    Variable(
        "original_aggregate_funded_pct",
        None,
        "Store original aggregate funded ratio in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_maxmin_funded_pct",
        None,
        "Store optimal original minimum funded ratio in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "largest_grant_filing",
        None,
        "Store largest original grant recipient’s filing identifier exactly as a source string.",
    ),
    Variable(
        "largest_grant_million",
        None,
        "Store largest original initial cash grant in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "actuarial_maxmin_funded_pct",
        None,
        "Store comparison optimal minimum funded ratio in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "actuarial_grant_to_original_leader_million",
        None,
        "Store comparison grant to the original largest-grant recipient in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "actuarial_year5_min_funded_pct",
        None,
        "Store comparison fifth-year minimum funded ratio in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "actuarial_year5_shortfall_million",
        None,
        "Store comparison fifth-year sum of positive plan shortfalls in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_stress_year5_min_funded_pct",
        None,
        "Store original stressed fifth-year minimum funded ratio in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_stress_year5_shortfall_million",
        None,
        "Store original stressed fifth-year sum of positive plan shortfalls in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "restoration_reserve_pv_million",
        None,
        "Store date-zero reserve present value in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "largest_restoration_call_million",
        None,
        "Store largest annual total additional contribution in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "largest_restoration_call_year",
        None,
        "Store year of largest additional contribution as an integer from 1 through 5.",
    ),
]
DECIMALS = [0, None, 4, 4, None, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]


@lru_cache(maxsize=1)
def ground_truth():
    out = calculate()[0]
    return tuple(out[v.name] for v in variables)


def _validate_subset(outputs, names):
    by = {v.name: v for v in variables}
    truth = dict(zip(by, ground_truth()))
    places = dict(zip(by, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by[n] for n in names], [truth[n] for n in names], [places[n] for n in names]
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


def validate_turn_5(outputs):
    return _validate_subset(outputs, TURN_5_NAMES)


def validate_turn_6(outputs):
    return _validate_subset(outputs, TURN_6_NAMES)


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1, 7)
}
