"""RC-N loan-state migration, keeping accrual status separate from charge-off.

The reported RC-N columns contain entire outstanding loan-and-lease balances, not only
missed installments. Hypothetical migration changes classification without
originating, curing principal, charging off, or realizing credit losses. The
denominator is explicitly FDIC *net* loans, so the constructed rate is not
advertised as the official gross-loan noncurrent ratio.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


OUTPUTS = (
    ("eligible_bank_count", 0, "the eligible matched-bank count as an integer"),
    ("selected_rssd_id", None, "the selected bank's exact RSSD identifier as a string"),
    ("selected_noncurrent_to_net_loans_leases_pct", 4, "the selected 90-plus-accruing plus nonaccrual balance divided by FDIC net loans and leases, in percent"),
    ("runner_up_noncurrent_to_net_loans_pct", 4, "the runner-up defined rate in percent"),
    ("opening_30_89_m", 3, "opening 30-89-days-past-due still-accruing loan-and-lease balance in USD millions"),
    ("opening_90_plus_m", 3, "opening 90-plus-days-past-due still-accruing loan-and-lease balance in USD millions"),
    ("opening_nonaccrual_m", 3, "opening nonaccrual loan-and-lease balance in USD millions"),
    ("post_migration_90_plus_m", 4, "90-plus-days-past-due still-accruing loan-and-lease balance after the first hypothetical migration in USD millions"),
    ("post_migration_nonaccrual_m", 4, "nonaccrual loan-and-lease balance after the first hypothetical migration in USD millions"),
    ("post_migration_rate_pct", 4, "the first-migration noncurrent exposure over unchanged FDIC net loans and leases in percent"),
    ("cured_new_90_plus_m", 4, "the balance cured from the newly migrated 90-plus cohort in USD millions"),
    ("post_cure_rate_pct", 4, "the post-cure noncurrent exposure over unchanged FDIC net loans and leases in percent"),
)
variables = [Variable(n, None, f"Store {d}" + (f", rounded to {p} decimals from unrounded inputs." if p is not None else ".")) for n, p, d in OUTPUTS]
PRECISION = {n: p for n, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:4]),
    tuple(n for n, _, _ in OUTPUTS[4:7]),
    tuple(n for n, _, _ in OUTPUTS[7:10]),
    tuple(n for n, _, _ in OUTPUTS[10:]),
)
M = Decimal(1000)


def dec(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite amount")
    return result


def analysis():
    delinquency = load_expansion_table("ffiec_call_reports_past_due")
    fdic = load_expansion_table("fdic_bankfind")
    fdic = fdic.loc[fdic.REPDTE.eq("2025-09-30")]
    joined = delinquency.merge(
        fdic[["CERT", "ASSET", "LNLSNET", "NCLNLS"]],
        left_on="fdic_certificate", right_on="CERT", validate="one_to_one",
    )
    cols = ["loans_30_89_accruing_thousand_usd", "loans_90_plus_accruing_thousand_usd",
            "loans_nonaccrual_thousand_usd", "LNLSNET", "NCLNLS"]
    good = joined.filing_type.eq("031") & joined.ASSET.gt(10_000_000)
    good &= joined[cols].notna().all(axis=1)
    good &= joined[cols[:4]].gt(0).all(axis=1) & joined.NCLNLS.ge(0)
    sample = joined.loc[good].copy()
    sample["ratio"] = (
        sample.loans_90_plus_accruing_thousand_usd + sample.loans_nonaccrual_thousand_usd
    ) / sample.LNLSNET * 100
    sample = sample.sort_values(["ratio", "bank_rssd_id"], ascending=[False, True])
    if len(sample) < 2:
        raise ValueError("insufficient complete banks")
    row, second = sample.iloc[0], sample.iloc[1]
    opening_30 = dec(row.loans_30_89_accruing_thousand_usd)
    opening_90 = dec(row.loans_90_plus_accruing_thousand_usd)
    opening_nonaccrual = dec(row.loans_nonaccrual_thousand_usd)
    denominator = dec(row.LNLSNET)
    if opening_90 + opening_nonaccrual != dec(row.NCLNLS):
        raise ValueError("selected FFIEC/FDIC noncurrent amount does not reconcile")
    newly_90 = opening_30 * Decimal("0.20")
    moved_nonaccrual = opening_90 * Decimal("0.40")
    ending_90 = opening_90 - moved_nonaccrual + newly_90
    ending_nonaccrual = opening_nonaccrual + moved_nonaccrual
    cure = newly_90 * Decimal("0.25")
    return {
        "eligible_bank_count": len(sample),
        "selected_rssd_id": str(row.bank_rssd_id),
        "selected_noncurrent_to_net_loans_leases_pct": float((opening_90 + opening_nonaccrual) / denominator * 100),
        "runner_up_noncurrent_to_net_loans_pct": float((dec(second.loans_90_plus_accruing_thousand_usd) + dec(second.loans_nonaccrual_thousand_usd)) / dec(second.LNLSNET) * 100),
        "opening_30_89_m": float(opening_30 / M),
        "opening_90_plus_m": float(opening_90 / M),
        "opening_nonaccrual_m": float(opening_nonaccrual / M),
        "post_migration_90_plus_m": float(ending_90 / M),
        "post_migration_nonaccrual_m": float(ending_nonaccrual / M),
        "post_migration_rate_pct": float((ending_90 + ending_nonaccrual) / denominator * 100),
        "cured_new_90_plus_m": float(cure / M),
        "post_cure_rate_pct": float((ending_90 - cure + ending_nonaccrual) / denominator * 100),
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


validators = {f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1, 5)}
