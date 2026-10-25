"""Pension at-risk funding-target basis and valuation-to-year-end asset timing.

The selected Schedule SB has targets of 824,435,822 and 817,044,697 USD;
substituting actuarial for current assets changes the numerator rather than
the at-risk liability basis. Its Schedule H beginning/end assets differ by
-16,454,976 USD, but that annual change is not an investment return. The
one-year contribution calculation is hypothetical, not a statutory minimum.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


OUTPUTS = (
    ("at_risk_eligible_plans", 0, "number of eligible unique at-risk plans as an integer"),
    ("at_risk_selected_filing", None, "selected Form 5500 filing ID as an exact string"),
    ("at_risk_valuation_date", None, "Schedule SB valuation date as an ISO YYYY-MM-DD string"),
    ("at_risk_target_usd", 0, "reported at-risk funding target in exact USD as an integer"),
    ("at_risk_without_target_usd", 0, "reported funding target without at-risk assumptions in exact USD as an integer"),
    ("at_risk_current_assets_usd", 0, "Schedule SB current-value assets at the valuation date in exact USD as an integer"),
    ("at_risk_actuarial_assets_usd", 0, "Schedule SB actuarial-value assets at the valuation date in exact USD as an integer"),
    ("at_risk_h_begin_assets_usd", 0, "Schedule H beginning net assets in exact USD as an integer"),
    ("at_risk_h_end_assets_usd", 0, "Schedule H ending net assets in exact USD as an integer"),
    ("at_risk_repeated_expense_usd", 0, "reported annual expense repeated in the hypothetical next year, exact USD integer"),
    ("at_risk_target_contribution_usd", 0, "minimum hypothetical initial contribution to meet the reported at-risk target at next year-end, exact USD integer"),
    ("at_risk_without_contribution_usd", 0, "minimum hypothetical initial contribution to meet the without-at-risk target at next year-end, exact USD integer"),
)
variables = [Variable(name, None, f"Store {description}.")
             for name, _, description in OUTPUTS]
PRECISION = {name: precision for name, precision, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 2), (2, 7), (7, 9), (9, 12)))


def d(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite plan amount")
    return result


def integer_usd(value) -> int:
    result = d(value)
    if result != result.to_integral_value():
        raise ValueError("unexpected fractional reported USD")
    return int(result)


def analysis() -> dict[str, int | str]:
    sb = load_expansion_table("dol_form5500_schedule_sb")
    h = load_expansion_table("dol_form5500_financials")
    sb = sb.loc[
        sb.plan_year_end.eq("2024-12-31")
        & sb.filing_status.eq("FILING_RECEIVED")
        & sb.date_received.le("2025-12-31")
    ].copy()
    sb = sb.sort_values(["sponsor_ein", "plan_number", "date_received", "filing_id"])
    sb = sb.drop_duplicates(["sponsor_ein", "plan_number", "plan_year_end"], keep="last")
    numeric = ("total_funding_target_usd", "funding_target_without_at_risk_assumptions_usd",
               "current_value_assets_usd", "actuarial_value_assets_usd")
    for column in numeric:
        sb[column] = pd.to_numeric(sb[column], errors="coerce")
    sb = sb.loc[
        pd.to_numeric(sb.at_risk_indicator, errors="coerce").eq(1)
        & sb.total_funding_target_usd.gt(0)
        & sb.funding_target_without_at_risk_assumptions_usd.gt(0)
        & sb.current_value_assets_usd.gt(0)
        & sb.actuarial_value_assets_usd.gt(0)
    ].copy()
    sb["target_gap"] = (sb.total_funding_target_usd
                        - sb.funding_target_without_at_risk_assumptions_usd)
    sb = sb.loc[sb.target_gap.gt(0)].copy()
    if h.filing_id.duplicated().any():
        raise ValueError("duplicate Schedule H filing ID")
    match = sb.merge(h[["filing_id", "net_assets_beginning_usd", "net_assets_ending_usd",
                        "total_expenses_usd"]], on="filing_id", how="inner", validate="one_to_one")
    for column in ("net_assets_beginning_usd", "net_assets_ending_usd", "total_expenses_usd"):
        match[column] = pd.to_numeric(match[column], errors="coerce")
    match = match.loc[
        match.net_assets_beginning_usd.gt(0)
        & match.net_assets_ending_usd.gt(0)
        & match.total_expenses_usd.ge(0)
    ].copy()
    if match.empty:
        raise ValueError("no eligible matched at-risk plan")
    match = match.sort_values(["target_gap", "filing_id"], ascending=[False, True])
    selected = match.iloc[0]
    end = d(selected.net_assets_ending_usd)
    expense = d(selected.total_expenses_usd)
    target = d(selected.total_funding_target_usd)
    without = d(selected.funding_target_without_at_risk_assumptions_usd)
    at_risk_contribution = max(Decimal(0), target + expense - end)
    without_contribution = max(Decimal(0), without + expense - end)
    return {
        "at_risk_eligible_plans": len(match),
        "at_risk_selected_filing": str(selected.filing_id),
        "at_risk_valuation_date": str(selected.valuation_date),
        "at_risk_target_usd": integer_usd(target),
        "at_risk_without_target_usd": integer_usd(without),
        "at_risk_current_assets_usd": integer_usd(selected.current_value_assets_usd),
        "at_risk_actuarial_assets_usd": integer_usd(selected.actuarial_value_assets_usd),
        "at_risk_h_begin_assets_usd": integer_usd(selected.net_assets_beginning_usd),
        "at_risk_h_end_assets_usd": integer_usd(end),
        "at_risk_repeated_expense_usd": integer_usd(expense),
        "at_risk_target_contribution_usd": integer_usd(at_risk_contribution),
        "at_risk_without_contribution_usd": integer_usd(without_contribution),
    }


@lru_cache(maxsize=1)
def ground_truth():
    result = analysis()
    return tuple(result[v.name] for v in variables)


def _check(outputs, names):
    by = {v.name: v for v in variables}
    truth = dict(zip(by, ground_truth()))
    return validate_ordered_outputs(outputs, [by[n] for n in names],
                                    [truth[n] for n in names],
                                    [PRECISION[n] for n in names])


def validate_turn_1(outputs): return _check(outputs, TURNS[0])
def validate_turn_2(outputs): return _check(outputs, TURNS[1])
def validate_turn_3(outputs): return _check(outputs, TURNS[2])
def validate_turn_4(outputs): return _check(outputs, TURNS[3])
def validate(outputs): return _check(outputs, tuple(v.name for v in variables))


validators = {f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
              for i in range(1, 5)}
