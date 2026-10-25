"""Synthetic borrower refinance choice with an executed-fee and horizon replan.

HMDA reports originations, not outstanding 24-payment loans or live offers.
The source amounts/coupons initialize an explicitly hypothetical book. A
payment-only comparison favors B's USD 116.7134m monthly payment over A's USD
119.8397m, whereas full 60-month savings are USD 734.1264m and USD 741.2650m,
respectively: financed fee and exit payoff reverse that ranking. Replacing the
chosen high-coupon A with B raises the constrained initial cost from USD
21,818.9109m to USD 21,821.2126m. Refunding the already paid USD 132.4781m A
fee when comparing month-six continuation is a false cash flow, changing the
KEEP forward cost from USD 6,639.5708m to USD 6,507.0927m. Alternative term,
fee and payoff conventions are distinct objects, not accepted tolerance paths.
"""

from functools import lru_cache
from itertools import product

import numpy as np

from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


CASE_ID = "mortgage_refinance_horizon_sunk_cost_replanning"
BANDS = ((3, 5), (5, 6), (6, 7), (7, 10))
FIRST_FIXING = "2024-12-31"
SECOND_FIXING = "2025-06-30"
CASH_BUDGET_SHARE = .0075


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("eligible_loan_count", "Store the eligible source-row count as an integer."),
    _v("month24_balance_mn", "Store aggregate balance just after payment 24 in USD millions, rounded to 4 decimals."),
    _v("old_monthly_payment_mn", "Store aggregate original monthly principal-and-interest payment in USD millions, rounded to 4 decimals."),
    _v("offer_a_monthly_payment_mn", "Store aggregate offer A monthly payment in USD millions, rounded to 4 decimals."),
    _v("offer_b_monthly_payment_mn", "Store aggregate offer B monthly payment in USD millions, rounded to 4 decimals."),
    _v("all_a_saving_mn", "Store present-cost saving from all-A rather than all-HOLD in USD millions, rounded to 4 decimals, computed from unrounded contracts."),
    _v("all_b_saving_mn", "Store present-cost saving from all-B rather than all-HOLD in USD millions, rounded to 4 decimals, computed from unrounded contracts."),
    _v("band_3_to_5_action", "Store exactly one token from HOLD, A, B for the [3%,5%) group."),
    _v("band_5_to_6_action", "Store exactly one token from HOLD, A, B for the [5%,6%) group."),
    _v("band_6_to_7_action", "Store exactly one token from HOLD, A, B for the [6%,7%) group."),
    _v("band_7_to_10_action", "Store exactly one token from HOLD, A, B for the [7%,10%) group."),
    _v("initial_upfront_cash_mn", "Store executed initial cash refinancing fees in USD millions, rounded to 4 decimals."),
    _v("chosen_60month_cost_mn", "Store the selected initial 60-month present cost including exit payoff in USD millions, rounded to 3 decimals; use unrounded source contracts."),
    _v("early_exit_action", "Store exactly one token from KEEP, B for the month-six early-exit decision."),
    _v("early_exit_payoff_mn", "Store the selected high-coupon group's balance paid at the early-exit month in USD millions, rounded to 4 decimals."),
    _v("early_exit_forward_cost_mn", "Store the selected high-coupon group's month-six forward present cost in USD millions, rounded to 3 decimals; exclude fees already paid."),
    _v("restored_horizon_action", "Store exactly one token from KEEP, B, PAYDOWN for the restored-horizon decision."),
    _v("restored_upfront_cash_mn", "Store new month-six cash paid for the selected restored-horizon action in USD millions, rounded to 4 decimals."),
    _v("restored_forward_cost_mn", "Store the selected high-coupon group's month-six forward present cost in USD millions, rounded to 3 decimals; exclude fees already paid."),
]

TURN_NAMES = [
    ["eligible_loan_count", "month24_balance_mn", "old_monthly_payment_mn"],
    ["offer_a_monthly_payment_mn", "offer_b_monthly_payment_mn", "all_a_saving_mn", "all_b_saving_mn"],
    ["band_3_to_5_action", "band_5_to_6_action", "band_6_to_7_action", "band_7_to_10_action", "initial_upfront_cash_mn", "chosen_60month_cost_mn"],
    ["early_exit_action", "early_exit_payoff_mn", "early_exit_forward_cost_mn"],
    ["restored_horizon_action", "restored_upfront_cash_mn", "restored_forward_cost_mn"],
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, None, None, None, None, 4, 3, None, 4, 3, None, 4, 3]


def _sofr(day):
    rates = load_expansion_table("nyfed_reference_rates")
    row = rates.loc[rates["Rate Type"].eq("SOFR") & rates["Effective Date"].eq(day)]
    if len(row) != 1:
        raise ValueError(f"SOFR fixing is not unique: {day}")
    return float(row.iloc[0]["Rate (%)"]) / 100


@lru_cache(maxsize=1)
def _cohort():
    frame = load_expansion_table("hmda")
    mask = (
        frame.activity_year.eq("2024") & frame.state_code.eq("IL")
        & frame.action_taken.eq("1") & frame.loan_type.eq("1")
        & frame.loan_purpose.eq("1") & frame.lien_status.eq("1")
        & frame.occupancy_type.eq("1") & frame.total_units.eq("1")
        & frame.construction_method.eq("1")
        & frame.loan_amount.between(100_000, 750_000)
        & frame.interest_rate.ge(3) & frame.interest_rate.lt(10)
    )
    rows = frame.loc[mask, ["loan_amount", "interest_rate"]].copy()
    if len(rows) != 69_993 or rows.loan_amount.sum() != 21_006_915_000:
        raise ValueError("Frozen Illinois HMDA cohort changed")
    amount = rows.loan_amount.to_numpy(dtype=float)
    annual_pct = rows.interest_rate.to_numpy(dtype=float)
    groups = np.searchsorted(np.array([5, 6, 7]), annual_pct, side="right")
    if not np.array_equal(np.bincount(groups, minlength=4), [453, 6009, 41671, 21860]):
        raise ValueError("Frozen coupon bands changed")
    return amount, annual_pct, groups


def _payment(principal, monthly_rate, periods):
    return principal * monthly_rate / (-np.expm1(-periods * np.log1p(monthly_rate)))


def _balance(principal, monthly_rate, periods, paid):
    pay = _payment(principal, monthly_rate, periods)
    return principal * np.power(1 + monthly_rate, paid) - pay * np.expm1(paid * np.log1p(monthly_rate)) / monthly_rate


def _pv(payment, payoff, horizon, monthly_discount):
    annuity = -np.expm1(-horizon * np.log1p(monthly_discount)) / monthly_discount
    return payment * annuity + payoff * np.power(1 + monthly_discount, -horizon)


def analysis():
    amount, annual_pct, groups = _cohort()
    first = _sofr(FIRST_FIXING)
    second = _sofr(SECOND_FIXING)
    r0 = annual_pct / 1200
    old_pay = _payment(amount, r0, 360)
    b24 = _balance(amount, r0, 360, 24)
    ra = (first + .01) / 12
    rb = (first + .0075) / 12
    pay_a = _payment(b24, ra, 336)
    pay_b = _payment(1.03 * b24, rb, 360)
    d1 = first / 12
    hold = _pv(old_pay, _balance(amount, r0, 360, 84), 60, d1)
    a = .02 * b24 + _pv(pay_a, _balance(b24, ra, 336, 60), 60, d1)
    b = _pv(pay_b, _balance(1.03 * b24, rb, 360, 60), 60, d1)
    group_costs = np.array([[z[groups == j].sum() for z in (hold, a, b)] for j in range(4)])
    group_fee = np.array([.02 * b24[groups == j].sum() for j in range(4)])
    budget = CASH_BUDGET_SHARE * b24.sum()
    candidates = []
    for choice in product(range(3), repeat=4):
        cash = sum(group_fee[j] for j, action in enumerate(choice) if action == 1)
        if cash <= budget + 1e-8:
            candidates.append((sum(group_costs[j, action] for j, action in enumerate(choice)), cash, choice))
    if not candidates:
        raise ValueError("No feasible financing allocation")
    best = min(candidates)
    if best[2] != (0, 0, 2, 1) or candidates and len(candidates) < 2:
        raise ValueError("Frozen group-decision frontier changed")
    actions = ["HOLD", "A", "B"]

    # Month six of the actually executed A contract in the high-coupon group.
    selected = groups == 3
    current = _balance(b24[selected], ra, 336, 6)
    old_a_pay = pay_a[selected]
    d2 = second / 12
    rb2 = (second + .0075) / 12

    def keep_cost(months):
        payoff = current * np.power(1 + ra, months) - old_a_pay * np.expm1(months * np.log1p(ra)) / ra
        return float(_pv(old_a_pay, payoff, months, d2).sum()), float(payoff.sum())

    def new_b_cost(months):
        new_principal = 1.03 * current
        new_payment = _payment(new_principal, rb2, 360)
        payoff = _balance(new_principal, rb2, 360, months)
        return float(_pv(new_payment, payoff, months, d2).sum()), float(payoff.sum())

    short = [(cost, token, payoff) for token, (cost, payoff) in (
        ("KEEP", keep_cost(12)), ("B", new_b_cost(12))
    )]
    early_cost, early_action, early_payoff = min(short)

    # Restoration is another decision from that same month-six state.
    paydown = .05 * current
    recast_fee = .001 * current
    recast_pay = _payment(current - paydown, ra, 330)
    recast_payoff = _balance(current - paydown, ra, 330, 60)
    paydown_cost = float((paydown + recast_fee + _pv(recast_pay, recast_payoff, 60, d2)).sum())
    long = [(cost, token, cash) for token, (cost, cash) in (
        ("KEEP", (keep_cost(60)[0], 0.0)),
        ("B", (new_b_cost(60)[0], 0.0)),
        ("PAYDOWN", (paydown_cost, float((paydown + recast_fee).sum()))),
    )]
    restored_cost, restored_action, restored_cash = min(long)
    if early_action != "KEEP" or restored_action != "PAYDOWN":
        raise ValueError("Frozen replan decisions changed")
    mn = lambda x: x / 1_000_000
    return {
        "eligible_loan_count": len(amount),
        "month24_balance_mn": mn(b24.sum()),
        "old_monthly_payment_mn": mn(old_pay.sum()),
        "offer_a_monthly_payment_mn": mn(pay_a.sum()),
        "offer_b_monthly_payment_mn": mn(pay_b.sum()),
        "all_a_saving_mn": mn(hold.sum() - a.sum()),
        "all_b_saving_mn": mn(hold.sum() - b.sum()),
        "band_3_to_5_action": actions[best[2][0]],
        "band_5_to_6_action": actions[best[2][1]],
        "band_6_to_7_action": actions[best[2][2]],
        "band_7_to_10_action": actions[best[2][3]],
        "initial_upfront_cash_mn": mn(best[1]),
        "chosen_60month_cost_mn": mn(best[0]),
        "early_exit_action": early_action,
        "early_exit_payoff_mn": mn(early_payoff),
        "early_exit_forward_cost_mn": mn(early_cost),
        "restored_horizon_action": restored_action,
        "restored_upfront_cash_mn": mn(restored_cash),
        "restored_forward_cost_mn": mn(restored_cost),
    }


@lru_cache(maxsize=1)
def ground_truth():
    result = analysis()
    return tuple(result[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    precision = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[n] for n in names], [truth[n] for n in names],
        [precision[n] for n in names],
    )


def validate(outputs):
    return _validate_subset(outputs, [v.name for v in variables])


validators = {
    f"validate_turn_{i}": turn_validator(
        lambda outputs, names=names: _validate_subset(outputs, names)
    )
    for i, names in enumerate(TURN_NAMES, start=1)
}
