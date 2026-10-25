"""A forecast-sized WTI purchase program, a later proposal, and financing that stays with the original.

Monthly barrel quantities are fixed from the EIA Short-Term Energy Outlook published on
2024-01-04 and hedged with the smallest common swap fraction that keeps every month under
a stressed cash ceiling. An independent proposal repeats the exercise on the 2024-04-04
forecast and is settled at observed prices; financing, warehousing and the terminal
borrowing base then return to the original program's committed quantities and swaps.

Rejected alternative conventions, each measured against the pinned reading:

- Financing the proposal's quantities instead of the original program's lowers terminal
  debt from 78.2589 to 70.5356 million.
- Keeping the original quantities but the proposal's hedge fraction gives terminal debt
  of 78.4180 million and a peak monthly outlay of 10.1476 instead of 10.0164 million.
- Dropping May from the purchase window cuts original inventory from 1,032.4283 to
  906.6422 thousand barrels.
- Reversing the swap direction, paying the spot average and receiving the fixed price,
  raises proposal gross swap payments from 0.6216 to 1.0489 million and terminal debt to
  79.5648 million.
"""

from functools import lru_cache
import numpy as np, pandas as pd
from core.data import load_expansion_table


@lru_cache(None)
def inputs():
    v = load_expansion_table("eia_steo_vintages")
    b = load_expansion_table("eia_bulk_wti")
    r = load_expansion_table("nyfed_reference_rates")
    months = [f"2024-{m:02d}" for m in range(5, 13)]
    vs = []
    for origin in ["2024-01-04", "2024-04-04"]:
        z = v[
            v.publication_date.eq(origin)
            & v.value_status_in_vintage.eq("forecast")
            & v.target_month.isin(months)
        ].set_index("target_month")
        assert z.index.is_unique
        vs.append(z.loc[months, "wti_spot_average_usd_per_barrel"].to_numpy(float))
    z = b[b.series_id.eq("PET.RWTC.M")].set_index("observation_month")
    assert z.index.is_unique
    k = float(z.loc["2023-12", "wti_spot_price_usd_per_barrel"]) + 2
    actual = z.loc[months, "wti_spot_price_usd_per_barrel"].to_numpy(float)
    rr = r[r["Rate Type"].eq("SOFR") & r["Effective Date"].eq("2024-01-04")]
    assert len(rr) == 1
    return months, vs[0], vs[1], k, actual, float(rr["Rate (%)"].iloc[0]) / 100 + 0.01


def hedge(f, k):
    q = 1e7 / f
    need = (1.25 * q * f - 11e6) / (q * (1.25 * f - k))
    h = max(0.0, float(max(need)))
    assert 0 < h < 0.8
    return q, h, int(np.argmax(need))


def calculate(
    quantity_state="original", hedge_state="original", window="full", settlement_sign="receive"
):
    months, a, b, k, s, r = inputs()
    if window == "omit_may":
        months = months[1:]
        a = a[1:]
        b = b[1:]
        s = s[1:]
    q, h, bind = hedge(a, k)
    qb, hb, bb = hedge(b, k)
    sign = 1 if settlement_sign == "receive" else -1
    swaps_b = sign * hb * qb * (s - k)
    cost_b = qb * s - swaps_b
    active_q = q if quantity_state == "original" else qb
    active_h = h if hedge_state == "original" else hb
    swap = sign * active_h * active_q * (s - k)
    cash = active_q * s - swap
    dates = pd.to_datetime(months) + pd.offsets.MonthEnd(0)
    days = np.array([(pd.Timestamp("2024-12-31") - d).days for d in dates])
    debt = float(cash @ (1 + r) ** (days / 365))
    storage = float(0.02 * (active_q @ days))
    terminal = float(active_q.sum() * s[-1])
    required = max(debt + storage - 0.9 * terminal, 0)
    out = dict(
        original_inventory_thousand_barrels=q.sum() / 1000,
        may_june_volume_pct=q[: (2 if window == "full" else 1)].sum() / q.sum() * 100,
        original_hedge_pct=h * 100,
        original_binding_month=months[bind],
        proposal_inventory_thousand_barrels=qb.sum() / 1000,
        proposal_hedge_pct=hb * 100,
        proposal_net_purchase_million=cost_b.sum() / 1e6,
        proposal_gross_swap_payments_million=float(np.maximum(-swaps_b, 0).sum()) / 1e6,
        original_terminal_debt_million=debt / 1e6,
        original_peak_month_cash_million=float(cash.max()) / 1e6,
        original_break_even_sale_usd_barrel=(debt + storage) / active_q.sum(),
        original_collateral_cash_million=required / 1e6,
    )
    return out, dict(
        strike=k,
        rate=r,
        quantities=q.tolist(),
        hedge=h,
        proposal_quantities=qb.tolist(),
        proposal_hedge=hb,
        binding_constraints=np.where(
            np.isclose(1.25 * q * a - h * q * (1.25 * a - k), 11e6, rtol=0, atol=1e-7)
        )[0].tolist(),
        stress_cash=(1.25 * q * a - h * q * (1.25 * a - k)).tolist(),
        actual=s.tolist(),
        cash=cash.tolist(),
        storage=storage,
        days=days.tolist(),
        debt=debt,
        quantity_conservation=float(np.max(abs(q * a - 1e7))),
    )


from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = ["original_inventory_thousand_barrels", "may_june_volume_pct"]
TURN_2_NAMES = ["original_hedge_pct", "original_binding_month"]
TURN_3_NAMES = ["proposal_inventory_thousand_barrels", "proposal_hedge_pct"]
TURN_4_NAMES = ["proposal_net_purchase_million", "proposal_gross_swap_payments_million"]
TURN_5_NAMES = ["original_terminal_debt_million", "original_peak_month_cash_million"]
TURN_6_NAMES = ["original_break_even_sale_usd_barrel", "original_collateral_cash_million"]

variables = [
    Variable(
        "original_inventory_thousand_barrels",
        None,
        "Store original total inventory in thousand barrels, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "may_june_volume_pct",
        None,
        "Store May-and-June share of original planned barrels in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_hedge_pct",
        None,
        "Store minimum original common swap participation in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_binding_month",
        None,
        "Store earliest original binding target month as a YYYY-MM string.",
    ),
    Variable(
        "proposal_inventory_thousand_barrels",
        None,
        "Store proposal total inventory in thousand barrels, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "proposal_hedge_pct",
        None,
        "Store minimum proposal common swap participation in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "proposal_net_purchase_million",
        None,
        "Store proposal total net purchase spending in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "proposal_gross_swap_payments_million",
        None,
        "Store proposal gross cash swap settlement payments in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_terminal_debt_million",
        None,
        "Store original program terminal borrowing in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_peak_month_cash_million",
        None,
        "Store original largest monthly net purchase cash outlay in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_break_even_sale_usd_barrel",
        None,
        "Store original minimum terminal liquidation price in USD per barrel, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_collateral_cash_million",
        None,
        "Store original terminal external cash requirement in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [4, 4, 4, None, 4, 4, 4, 4, 4, 4, 4, 4]


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
