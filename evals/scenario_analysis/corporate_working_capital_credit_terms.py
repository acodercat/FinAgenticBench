"""Operating-cycle financing and exclusive commercial-term alternatives.

The annual filing supplies as-filed stocks and flows. The subsequent growth,
collection, inventory and supplier terms are explicitly hypothetical planning
states, priced on the governed 2025 SOFR path. The closing-balance planning
ratios are proxies and do not establish actual contract terms or credit sales.
"""

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_facts, load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


def D(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Non-finite amount")
    return result


variables = [
    Variable("receivable_days", None, "Store the closing-balance receivable planning period in days as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("inventory_days", None, "Store the closing-balance inventory planning period in days as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("payable_days", None, "Store the closing-balance payable planning period in days as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("reported_cash_bn", None, "Store as-filed closing cash in USD billions as a finite number, rounded to 6 decimals from the unrounded source."),
    Variable("plan_receivables_bn", None, "Store projected closing receivables in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("plan_inventory_bn", None, "Store projected closing inventory in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("plan_payables_bn", None, "Store projected closing accounts payable in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("initial_revolver_draw_bn", None, "Store the initial revolver draw in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("base_funding_interest_bn", None, "Store simple interest on the carried base draw in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("chosen_action", None, "Store the selected action as exactly one canonical token: COLLECTIONS or SUPPLIER."),
    Variable("chosen_total_financing_cost_bn", None, "Store the selected action's total fee and interest cost in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
]

TURN_1_NAMES = ["receivable_days", "inventory_days", "payable_days", "reported_cash_bn"]
TURN_2_NAMES = ["plan_receivables_bn", "plan_inventory_bn", "plan_payables_bn", "initial_revolver_draw_bn"]
TURN_3_NAMES = ["base_funding_interest_bn"]
TURN_4_NAMES = ["chosen_action", "chosen_total_financing_cost_bn"]
DECIMALS = [None if v.name == "chosen_action" else 6 if v.name == "reported_cash_bn" else 4 for v in variables]

ACCESSION = "0000006951-24-000044"
END_DATE = "2024-10-27"
REVENUE = "RevenueFromContractWithCustomerExcludingAssessedTax"
COST = "CostOfGoodsAndServicesSold"
AR = "AccountsReceivableNetCurrent"
INVENTORY = "InventoryNet"
AP = "AccountsPayableCurrent"
CASH = "CashAndCashEquivalentsAtCarryingValue"


def _filing_state():
    facts = load_facts()
    facts = facts.loc[
        facts.accession.eq(ACCESSION)
        & facts.cik.astype("string").str.lstrip("0").eq("6951")
        & facts.form.eq("10-K") & facts.taxonomy.eq("us-gaap")
        & facts.unit.eq("USD") & facts.end_date.eq(END_DATE)
        & facts.concept.isin([REVENUE, COST, AR, INVENTORY, AP, CASH])
    ]
    amounts = {}
    for concept in (REVENUE, COST, AR, INVENTORY, AP, CASH):
        rows = facts[facts.concept.eq(concept)]
        if concept in (REVENUE, COST):
            rows = rows[rows.start_date.eq("2023-10-30")]
        else:
            rows = rows[rows.start_date.isna()]
        values = rows.value.dropna().unique()
        if len(values) != 1:
            raise ValueError(f"Missing or conflicting same-accession {concept}")
        amounts[concept] = D(values[0]) / D(1000000000)
    if min(amounts.values()) <= 0:
        raise ValueError("Expected positive annual filing inputs")
    return amounts


def _sofr_factor():
    rates = load_expansion_table("nyfed_reference_rates")
    rates = rates.loc[rates["Rate Type"].eq("SOFR"), ["Effective Date", "Rate (%)"]]
    if rates["Effective Date"].duplicated().any():
        raise ValueError("Duplicate SOFR effective date")
    fixing = rates.set_index(pd.to_datetime(rates["Effective Date"]))["Rate (%)"].sort_index()
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    daily = fixing.reindex(days, method="ffill")
    if daily.isna().any():
        raise ValueError("Missing SOFR history for financing year")
    return sum(((D(rate) / 100 + D("0.015")) / 360 for rate in daily), D(0))


def analysis():
    state = _filing_state()
    revenue, cost = state[REVENUE], state[COST]
    ar, inventory, ap, cash = state[AR], state[INVENTORY], state[AP], state[CASH]
    if cash <= D(7):
        raise ValueError("Opening cash no longer exceeds reserve")
    dso = ar / revenue * 365
    dio = inventory / cost * 365
    dpo = ap / cost * 365
    projected_revenue, projected_cost = revenue * D("1.20"), cost * D("1.20")
    new_ar = projected_revenue * (dso + 30) / 365
    new_inventory = projected_cost * (dio + 20) / 365
    new_ap = projected_cost * (dpo - 10) / 365
    if min(new_ar, new_inventory, new_ap) <= 0:
        raise ValueError("Invalid stressed working-capital component")
    added_working_capital = new_ar + new_inventory - new_ap - (ar + inventory - ap)
    draw = max(D(0), added_working_capital - (cash - D(7)))
    if draw <= 0:
        raise ValueError("Stress no longer requires external financing")
    factor = _sofr_factor()
    alternatives = {
        "COLLECTIONS": (draw - projected_revenue * D(10) / 365 + D("0.020"), D("0.020")),
        "SUPPLIER": (draw - projected_cost * D(15) / 365 + D("0.015"), D("0.015")),
    }
    if any(principal <= 0 for principal, _ in alternatives.values()):
        raise ValueError("Action unexpectedly removes all borrowing")
    costs = {name: fee + principal * factor for name, (principal, fee) in alternatives.items()}
    chosen = min(costs, key=lambda name: (costs[name], name))
    return {
        "receivable_days": float(dso),
        "inventory_days": float(dio),
        "payable_days": float(dpo),
        "reported_cash_bn": float(cash),
        "plan_receivables_bn": float(new_ar),
        "plan_inventory_bn": float(new_inventory),
        "plan_payables_bn": float(new_ap),
        "initial_revolver_draw_bn": float(draw),
        "base_funding_interest_bn": float(draw * factor),
        "chosen_action": chosen,
        "chosen_total_financing_cost_bn": float(costs[chosen]),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answer = analysis()
    return tuple(answer[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    precision = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[name] for name in names],
        [truth[name] for name in names], [precision[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_operating_cycle": turn_validator(validate_turn_1),
    "validate_growth_plan": turn_validator(validate_turn_2),
    "validate_funding": turn_validator(validate_turn_3),
    "validate_action": turn_validator(validate_turn_4),
}
