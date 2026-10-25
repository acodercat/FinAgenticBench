"""Domestic time-deposit reset, contractual runoff and renewal decision.

Baseline alternatives: the RC-E mixed one-year reset-or-maturity total is not
cash maturing within one year. For the selected bank the over-USD-250,000 mixed
bucket is 804,584 thousand USD versus 372,448 thousand USD of contractual
maturity; using it as runoff overstates the 50%-nonrenewal cash need by
216,068 thousand USD. The counterfactual 150 bp change is not an observed
deposit-rate beta.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


OUTPUTS = (
    ("time_deposit_eligible_banks", 0, "eligible bank count as an integer"),
    ("time_deposit_selected_rssd", None, "selected bank's exact RSSD ID as a string"),
    ("time_deposit_small_short_m", 3, "at-most-USD-250,000 time deposits with maturity or next repricing within one year, USD millions"),
    ("time_deposit_small_maturity_m", 3, "at-most-USD-250,000 time deposits contractually maturing within one year, USD millions"),
    ("time_deposit_large_short_m", 3, "over-USD-250,000 time deposits with maturity or next repricing within one year, USD millions"),
    ("time_deposit_large_maturity_m", 3, "over-USD-250,000 time deposits contractually maturing within one year, USD millions"),
    ("time_deposit_half_renewal_cash_m", 3, "hypothetical ending cash after 50 percent of one-year maturities renew, USD millions"),
    ("time_deposit_half_renewal_extra_expense_m", 4, "hypothetical next-year annualized incremental interest expense at 50 percent renewal, USD millions"),
    ("time_deposit_floor_min_renewal_pct", 4, "smallest common renewal fraction meeting the hypothetical cash floor, percent"),
    ("time_deposit_floor_extra_expense_m", 4, "hypothetical next-year annualized incremental interest expense at the minimum floor-preserving renewal fraction, USD millions"),
)
variables = [Variable(name, None, f"Store {description}" +
    (f", rounded to {precision} decimals from unrounded inputs." if precision is not None else "."))
    for name, precision, description in OUTPUTS]
PRECISION = {name: precision for name, precision, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 2), (2, 6), (6, 8), (8, 10)))


def d(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite amount")
    return result


def analysis() -> dict[str, int | str | float]:
    rce = load_expansion_table("ffiec_call_reports_time_deposits")
    fdic = load_expansion_table("fdic_bankfind")
    fdic = fdic.loc[fdic.REPDTE.eq("2025-09-30")]
    joined = rce.merge(fdic[["CERT", "ASSET", "CHBAL"]],
                       left_on="fdic_certificate", right_on="CERT", validate="one_to_one")
    small_a = "small_within_3m_maturity_or_reset_thousand_usd"
    small_b = "small_3_to_12m_maturity_or_reset_thousand_usd"
    small_mat = "small_within_1y_maturity_thousand_usd"
    large_a = "large_within_3m_maturity_or_reset_thousand_usd"
    large_b = "large_3_to_12m_maturity_or_reset_thousand_usd"
    large_mat = "large_within_1y_maturity_thousand_usd"
    fields = [small_a, small_b, small_mat, large_a, large_b, large_mat,
              "ASSET", "CHBAL"]
    valid = joined.filing_type.eq("041") & joined[fields].notna().all(axis=1)
    valid &= joined[fields].ge(0).all(axis=1)
    valid &= joined.ASSET.gt(5_000_000) & joined.CHBAL.gt(0)
    sample = joined.loc[valid].copy()
    sample["small_short"] = sample[small_a] + sample[small_b]
    sample["large_short"] = sample[large_a] + sample[large_b]
    sample["long_reset"] = (sample.small_short - sample[small_mat]
                             + sample.large_short - sample[large_mat])
    sample["maturity"] = sample[small_mat] + sample[large_mat]
    sample = sample.loc[sample.long_reset.gt(0) & sample.maturity.gt(0)]
    sample = sample.sort_values(["long_reset", "bank_rssd_id"], ascending=[False, True])
    if len(sample) < 2:
        raise ValueError("insufficient banks with observable reset/maturity distinction")
    row = sample.iloc[0]
    small_short, small_due = d(row.small_short), d(row[small_mat])
    large_short, large_due = d(row.large_short), d(row[large_mat])
    maturity = small_due + large_due
    long_reset = small_short - small_due + large_short - large_due
    cash = d(row.CHBAL)
    half = Decimal("0.5")
    rate = Decimal("0.015")
    floor = Decimal(100_000)
    half_cash = cash - (Decimal(1) - half) * maturity
    half_expense = (long_reset + half * maturity) * rate
    min_renewal = max(Decimal(0), (maturity - (cash - floor)) / maturity)
    if not Decimal(0) < min_renewal < Decimal(1):
        raise ValueError("selected bank has no interior renewal decision")
    floor_expense = (long_reset + min_renewal * maturity) * rate
    scale = Decimal(1000)
    return {
        "time_deposit_eligible_banks": len(sample),
        "time_deposit_selected_rssd": str(row.bank_rssd_id),
        "time_deposit_small_short_m": float(small_short / scale),
        "time_deposit_small_maturity_m": float(small_due / scale),
        "time_deposit_large_short_m": float(large_short / scale),
        "time_deposit_large_maturity_m": float(large_due / scale),
        "time_deposit_half_renewal_cash_m": float(half_cash / scale),
        "time_deposit_half_renewal_extra_expense_m": float(half_expense / scale),
        "time_deposit_floor_min_renewal_pct": float(min_renewal * 100),
        "time_deposit_floor_extra_expense_m": float(floor_expense / scale),
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
