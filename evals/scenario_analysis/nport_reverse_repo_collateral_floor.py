"""Observed reverse-repo roles with a separately stated collateral-floor counterfactual.

The reported collateral-security principal for the selected position is USD
62,900,000; substituting it for the USD 50,854,650 holding PA balance changes
the stress shortfall to USD 13,445,734.8125. The question explicitly uses the
holding's balance, so the security principal is not an accepted denominator.
The floor is not an observed contractual margin requirement.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


OUTPUTS = (
    ("repo_position_count", 0, "the repurchase-position count as an integer"),
    ("reverse_position_count", 0, "the reverse-repurchase-position count as an integer"),
    ("repo_signed_balance_m", 4, "the signed repurchase balance total in USD millions"),
    ("reverse_signed_balance_m", 4, "the signed reverse-repurchase balance total in USD millions"),
    ("reverse_covered_count", 0, "the initially covered reverse-repurchase-position count as an integer"),
    ("reverse_selected_holding_id", None, "the selected exact holding ID as a string"),
    ("reverse_balance_m", 5, "the absolute reported balance of the selected holding in USD millions"),
    ("reverse_collateral_m", 5, "the selected holding's total reported collateral value in USD millions"),
    ("reverse_contract_rate_pct", 2, "the selected agreement's annualized rate in percent"),
    ("reverse_sofr_spread_bp", 0, "the selected agreement rate less the same-effective-date published SOFR fixing, in basis points"),
    ("reverse_floor_topup_m", 4, "the hypothetical additional collateral needed after the markdown in USD millions"),
)
variables = [Variable(name, None, f"Store {description}" +
    (f", rounded to {precision} decimals from unrounded inputs." if precision is not None else "."))
    for name, precision, description in OUTPUTS]
PRECISION = {name: precision for name, precision, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 4), (4, 6), (6, 10), (10, 11)))


def d(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite repo amount")
    return result


def analysis() -> dict[str, int | str | float]:
    raw = load_expansion_table("sec_nport_repo_collateral")
    raw = raw.loc[raw.accession.eq("0000810893-25-002583") &
                  raw.report_date.eq("2025-06-30") & raw.form.eq("NPORT-P") &
                  raw.unit.eq("PA") & raw.holding_currency.eq("USD") &
                  raw.collateral_currency.eq("USD") & raw.collateral_category.eq("UST")]
    if raw.empty or raw.duplicated(["holding_id", "collateral_id"]).any():
        raise ValueError("missing or repeated collateral records")
    head = raw.drop_duplicates("holding_id")[["holding_id", "transaction_type",
        "balance", "repurchase_rate_pct"]].copy()
    if head.holding_id.duplicated().any():
        raise ValueError("duplicate agreement")
    collateral = raw.groupby("holding_id", as_index=False).collateral_value_usd.sum()
    positions = head.merge(collateral, on="holding_id", validate="one_to_one")
    repo = positions.loc[positions.transaction_type.eq("Repurchase")]
    reverse = positions.loc[positions.transaction_type.eq("Reverse repurchase")]
    if repo.empty or reverse.empty or not repo.balance.gt(0).all() or not reverse.balance.lt(0).all():
        raise ValueError("repo direction or sign inconsistent")
    covered = reverse.loc[reverse.collateral_value_usd.ge(-reverse.balance)].copy()
    covered["absolute_balance"] = -covered.balance
    covered = covered.sort_values(["absolute_balance", "holding_id"], ascending=[False, True])
    if covered.empty:
        raise ValueError("no initially covered reverse repo")
    selected = covered.iloc[0]
    rates = load_expansion_table("nyfed_reference_rates")
    sofr = rates.loc[rates["Effective Date"].eq("2025-06-30") & rates["Rate Type"].eq("SOFR")]
    if len(sofr) != 1:
        raise ValueError("SOFR effective-date fixing missing or duplicated")
    balance = d(selected.absolute_balance)
    collateral_value = d(selected.collateral_value_usd)
    agreement_rate = d(selected.repurchase_rate_pct)
    fixing = d(sofr.iloc[0]["Rate (%)"])
    topup = max(Decimal(0), balance - collateral_value * Decimal(".95"))
    scale = Decimal(1_000_000)
    return {
        "repo_position_count": len(repo),
        "reverse_position_count": len(reverse),
        "repo_signed_balance_m": float(sum((d(v) for v in repo.balance), Decimal(0)) / scale),
        "reverse_signed_balance_m": float(sum((d(v) for v in reverse.balance), Decimal(0)) / scale),
        "reverse_covered_count": len(covered),
        "reverse_selected_holding_id": str(selected.holding_id),
        "reverse_balance_m": float(balance / scale),
        "reverse_collateral_m": float(collateral_value / scale),
        "reverse_contract_rate_pct": float(agreement_rate),
        "reverse_sofr_spread_bp": float((agreement_rate - fixing) * 100),
        "reverse_floor_topup_m": float(topup / scale),
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
