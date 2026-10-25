"""A money-fund liquidity vehicle, a Prime alternative, and a facility budgeted for the original.

A hypothetical USD10 million vehicle takes the balance-sheet mix of the three largest
Government money funds, with reported daily and weekly liquid assets released on a
stipulated ladder; withdrawals are met from that ladder and a bridge facility. A Prime
alternative is screened, funded and then constrained by a line sized to the Government
vehicle's peak draw, before the facility budget returns to the Government vehicle. The
Prime analysis must not overwrite the vehicle the budget is for.

Rejected alternative conventions, each measured against the pinned reading:

- Budgeting the facility on the Prime ledger instead of the Government vehicle's raises
  total financing cost from 7,860.1006 to 9,731.1157 USD and cuts the maximum
  commitment-fee rate from 238.3975 to 67.9312 basis points.
- Treating the whole weekly liquid-asset amount as available on day 1, instead of adding
  only its increment on day 5, lifts the day-1 fraction from 62.9738 to 78.3327 percent
  and lowers the Government peak draw from 2.7026 to 1.6667 million.
- Weighting the three selected series equally instead of by net assets changes 10 of the
  12 outputs, moving the blended maturity from 39.3691 to 39.6667 days.
- Accruing interest on Actual/365 instead of Actual/360 lowers Government interest from
  7,609.4065 to 7,505.1681 USD.
"""

from functools import lru_cache
import numpy as np
from core.data import load_expansion_table


@lru_cache(None)
def inputs():
    s = load_expansion_table("sec_nmfp")
    l = load_expansion_table("sec_nmfp_liquidity")
    s = s[s.report_date.eq("2025-10-31") & s.form.eq("N-MFP3")]
    assert s.series_id.is_unique and s.accession.is_unique
    q = s.merge(l[l.liquidity_date.eq("2025-10-31")], on="accession", validate="one_to_one")
    q = q[
        (q.net_assets_usd > 0)
        & (q.daily_liquid_assets_usd > 0)
        & (q.weekly_liquid_assets_usd >= q.daily_liquid_assets_usd)
        & (q.weekly_liquid_assets_usd <= q.net_assets_usd)
        & (q.weighted_average_maturity_days > 0)
    ]
    rates = load_expansion_table("nyfed_reference_rates")
    r = rates[rates["Rate Type"].eq("SOFR") & rates["Effective Date"].eq("2025-10-31")]
    assert len(r) == 1
    return q, float(r["Rate (%)"].iloc[0]) / 100 + 0.01


def select(q, category, ceiling=np.inf, weighting="nav"):
    eligible = q[q.fund_category.eq(category) & (q.weighted_average_maturity_days <= ceiling)]
    z = eligible.sort_values(["net_assets_usd", "series_id"], ascending=[False, True]).head(3)
    assert len(z) == 3
    w = z.net_assets_usd.to_numpy(float)
    w = w / w.sum() if weighting == "nav" else np.full(3, 1 / 3)
    da = float(w @ (z.daily_liquid_assets_usd / z.net_assets_usd))
    wa = float(w @ (z.weekly_liquid_assets_usd / z.net_assets_usd))
    return dict(
        count=len(eligible),
        ids=z.series_id.tolist(),
        accessions=z.accession.tolist(),
        wam=float(w @ z.weighted_average_maturity_days),
        buckets=1e7 * np.array([da, wa - da, 1 - wa]),
        weights=w,
    )


def funding(s, r, cap=np.inf, haircut=0.02):
    b = s["buckets"].copy()
    cash = 0.0
    debt = 0.0
    area = 0.0
    sold = 0.0
    peak = 0.0
    previous = 0.0
    trace = []
    for day, inflow, withdrawal in [(1, b[0], 9e6), (5, b[1], 0.5e6), (30, b[2], 0.0)]:
        if day == 30:
            inflow -= sold
        area += debt * (day - previous)
        cash += inflow - withdrawal
        if cash < 0:
            draw = min(-cash, cap - debt)
            debt += draw
            cash += draw
            if cash < 0:
                sale = -cash / (1 - haircut)
                sold += sale
                cash += sale * (1 - haircut)
        repay = min(cash, debt)
        cash -= repay
        debt -= repay
        peak = max(peak, debt)
        assert cash >= -1e-6 and debt >= -1e-6 and sold <= b[2] + 1e-6
        trace.append([day, cash, debt, sold])
        previous = day
    assert abs(debt) < 1e-6
    interest = area * r / 360
    return dict(
        peak=peak, area=area, interest=interest, sold=sold, end_cash=cash - interest, trace=trace
    )


def calculate(
    contamination="none", bucket_scope="incremental", weighting="nav", rate_basis="act360"
):
    q, r = inputs()
    g = select(q, "Government", weighting=weighting)
    p = select(q, "Prime", g["wam"], weighting)
    if bucket_scope == "weekly_as_daily":
        for state in [g, p]:
            state["buckets"][0] += state["buckets"][1]
            state["buckets"][1] = 0.0
    r = r if rate_basis == "act360" else r * 360 / 365
    a = funding(g, r)
    b = funding(p, r)
    cap = a["peak"] + 250000
    c = funding(p, r, cap)
    active = c if contamination == "prime" else a
    unused = cap * 30 - active["area"]
    assert unused > 0
    cost = active["interest"] + unused * 0.0025 / 360
    maxfee = (10000 - active["interest"]) * 360 / unused * 10000
    out = dict(
        government_lead_series=g["ids"][0],
        government_wam_days=g["wam"],
        government_daily_liquidity_pct=g["buckets"][0] / 1e5,
        government_peak_bridge_million=a["peak"] / 1e6,
        government_interest_usd=a["interest"],
        prime_eligible_count=p["count"],
        prime_peak_bridge_million=b["peak"] / 1e6,
        prime_interest_usd=b["interest"],
        prime_forced_sale_million=c["sold"] / 1e6,
        prime_day30_cash_usd=c["end_cash"],
        government_all_in_cost_usd=cost,
        government_max_commitment_fee_bp=maxfee,
    )
    return out, dict(
        gov=g,
        prime=p,
        gov_funding=a,
        prime_funding=b,
        prime_capped=c,
        facility=cap,
        unused_dollar_days=unused,
        rate=r,
    )


from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = ["government_lead_series", "government_wam_days", "government_daily_liquidity_pct"]
TURN_2_NAMES = ["government_peak_bridge_million", "government_interest_usd"]
TURN_3_NAMES = ["prime_eligible_count", "prime_peak_bridge_million", "prime_interest_usd"]
TURN_4_NAMES = ["prime_forced_sale_million", "prime_day30_cash_usd"]
TURN_5_NAMES = ["government_all_in_cost_usd", "government_max_commitment_fee_bp"]

variables = [
    Variable(
        "government_lead_series",
        None,
        "Store the largest selected Government fund SEC series ID exactly as a string.",
    ),
    Variable(
        "government_wam_days",
        None,
        "Store Government vehicle blended weighted-average maturity in days, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "government_daily_liquidity_pct",
        None,
        "Store Government day1 available asset share in percent, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "government_peak_bridge_million",
        None,
        "Store Government peak drawn principal in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "government_interest_usd",
        None,
        "Store Government bridge interest in USD, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "prime_eligible_count",
        None,
        "Store eligible Prime series count before selecting three as an integer.",
    ),
    Variable(
        "prime_peak_bridge_million",
        None,
        "Store unlimited Prime peak drawn principal in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "prime_interest_usd",
        None,
        "Store unlimited Prime bridge interest in USD, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "prime_forced_sale_million",
        None,
        "Store Prime assets sold early at par amount in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "prime_day30_cash_usd",
        None,
        "Store Prime cash at day30 after debt and interest settlement in USD, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "government_all_in_cost_usd",
        None,
        "Store Government interest plus commitment fee in USD, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "government_max_commitment_fee_bp",
        None,
        "Store maximum annual commitment fee in basis points, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [None, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4]


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


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1, 6)
}
