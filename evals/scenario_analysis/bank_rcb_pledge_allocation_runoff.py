"""Partial-identification runoff stress from a bank's reported securities totals.

The pledged amount is not allocated by HTM/AFS category in the governed table.
The two extremal allocations are explicit counterfactuals, not observed holdings
or an assertion that pledged securities can be sold. AFS securities are already
carried at fair value; charging their disclosed amortized-cost gap again to book
equity is the measured wrong accounting path. HTM sales here are a simplified,
tax-free hypothetical and may have classification implications beyond this case.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


M = Decimal(1_000_000)
NAMES = (
    ("eligible_bank_count", 0, "the count of eligible matched banks as an integer"),
    ("selected_rssd_id", None, "the selected bank's exact RSSD identifier as a string"),
    ("selected_bank_name", None, "the selected bank's exact source-verbatim legal name"),
    ("selected_htm_gap_to_equity_pct", 4, "the selected HTM cost-minus-fair-value gap divided by FDIC book equity, in percent"),
    ("runner_up_htm_gap_to_equity_pct", 4, "the runner-up HTM gap divided by FDIC book equity, in percent"),
    ("htm_gap_m", 3, "the selected bank's HTM amortized cost minus fair value in USD millions"),
    ("afs_gap_m", 3, "the selected bank's AFS amortized cost minus fair value in USD millions"),
    ("other_securities_book_m", 3, "FDIC total securities less HTM amortized cost and AFS fair value, in USD millions"),
    ("minimum_unpledged_afs_m", 3, "the minimum AFS fair value unpledged under a category allocation of the aggregate pledge, in USD millions"),
    ("maximum_unpledged_afs_m", 3, "the maximum AFS fair value unpledged under a category allocation of the aggregate pledge, in USD millions"),
    ("withdrawal_m", 3, "the stipulated deposit withdrawal in USD millions"),
    ("cash_shortfall_m", 3, "the withdrawal less opening cash, floored at zero, in USD millions"),
    ("afs_only_feasibility", None, "exactly one token: always, allocation-dependent, or never"),
    ("pessimistic_sale_loss_m", 3, "positive realized HTM sale loss under the pessimistic allocation, in USD millions"),
    ("pessimistic_ending_book_equity_m", 3, "ending book equity after the stipulated pessimistic HTM sale loss, in USD millions"),
)
variables = [Variable(n, None, f"Store {d}" + (f", rounded to {p} decimals from unrounded inputs." if p is not None else ".")) for n,p,d in NAMES]
PRECISION = {n:p for n,p,_ in NAMES}
DECIMALS = tuple(p for _,p,_ in NAMES)
TURNS = (
    tuple(n for n,_,_ in NAMES[:5]),
    tuple(n for n,_,_ in NAMES[5:10]),
    tuple(n for n,_,_ in NAMES[10:13]),
    tuple(n for n,_,_ in NAMES[13:]),
)


def dec(x):
    value = Decimal(str(x))
    if not value.is_finite():
        raise ValueError("non-finite source value")
    return value


def analysis():
    securities = load_expansion_table("ffiec_call_reports_securities")
    balances = load_expansion_table("ffiec_call_reports_balance")
    banks = load_expansion_table("fdic_bankfind")
    balances = balances.loc[balances.report_date.eq("2025-09-30")]
    banks = banks.loc[banks.REPDTE.eq("2025-09-30")]
    joined = securities.merge(
        balances[["bank_rssd_id", "report_date", "fdic_certificate", "bank_legal_name", "total_assets_thousand_usd", "balance_reporting_basis"]],
        on=["bank_rssd_id", "report_date"], validate="one_to_one",
    ).merge(
        banks[["CERT", "DEP", "CHBAL", "SC", "EQ", "ASSET"]],
        left_on="fdic_certificate", right_on="CERT", validate="one_to_one",
    )
    cols = ["htm_amortized_cost_thousand_usd", "htm_fair_value_thousand_usd",
            "afs_amortized_cost_thousand_usd", "afs_fair_value_thousand_usd",
            "pledged_securities_book_value_thousand_usd", "EQ", "DEP", "CHBAL", "SC"]
    good = joined.reporting_basis.eq("consolidated") & joined.balance_reporting_basis.eq("consolidated")
    good &= joined.total_assets_thousand_usd.gt(10_000_000)
    good &= joined[cols].notna().all(axis=1)
    good &= joined[["EQ", "DEP", "htm_fair_value_thousand_usd"]].gt(0).all(axis=1)
    good &= joined[["CHBAL", "SC", "pledged_securities_book_value_thousand_usd"]].ge(0).all(axis=1)
    sample = joined.loc[good].copy()
    sample["htm_gap"] = sample.htm_amortized_cost_thousand_usd - sample.htm_fair_value_thousand_usd
    sample = sample.loc[sample.htm_gap.gt(0)].copy()
    sample["ratio"] = sample.htm_gap / sample.EQ * 100
    sample = sample.sort_values(["ratio", "bank_rssd_id"], ascending=[False, True])
    if len(sample) < 2:
        raise ValueError("insufficient complete positive-gap banks")
    row, second = sample.iloc[0], sample.iloc[1]
    htm_cost = dec(row.htm_amortized_cost_thousand_usd)
    htm_fair = dec(row.htm_fair_value_thousand_usd)
    afs_cost = dec(row.afs_amortized_cost_thousand_usd)
    afs_fair = dec(row.afs_fair_value_thousand_usd)
    pledged = dec(row.pledged_securities_book_value_thousand_usd)
    securities_book = dec(row.SC)
    cash = dec(row.CHBAL)
    deposits = dec(row.DEP)
    equity = dec(row.EQ)
    other = securities_book - htm_cost - afs_fair
    free = securities_book - pledged
    if not (other >= 0 and free > 0 and securities_book >= pledged):
        raise ValueError("inconsistent RC-B/FDIC securities reconciliation")
    minimum_afs_free = max(Decimal(0), afs_fair - pledged)
    maximum_afs_free = min(afs_fair, free)
    withdrawal = deposits * Decimal("0.15")
    gap = max(Decimal(0), withdrawal - cash)
    if gap <= minimum_afs_free:
        feasibility = "always"
    elif gap <= maximum_afs_free:
        feasibility = "allocation-dependent"
    else:
        feasibility = "never"
    # Pessimistic endpoint: allocate the pledge to AFS and the residual
    # securities category first, leaving the remaining free book in HTM.
    free_htm = max(Decimal(0), htm_cost - max(Decimal(0), pledged - afs_fair - other))
    fair_capacity = free_htm * htm_fair / htm_cost
    if not (0 < gap <= fair_capacity):
        raise ValueError("selected stress cannot be funded by pessimistic free HTM")
    htm_cost_sold = gap * htm_cost / htm_fair
    sale_loss = htm_cost_sold - gap
    return {
        "eligible_bank_count": len(sample),
        "selected_rssd_id": str(row.bank_rssd_id),
        "selected_bank_name": str(row.bank_legal_name),
        "selected_htm_gap_to_equity_pct": float(dec(row.htm_gap) / equity * 100),
        "runner_up_htm_gap_to_equity_pct": float(dec(second.htm_gap) / dec(second.EQ) * 100),
        "htm_gap_m": float((htm_cost - htm_fair) / 1000),
        "afs_gap_m": float((afs_cost - afs_fair) / 1000),
        "other_securities_book_m": float(other / 1000),
        "minimum_unpledged_afs_m": float(minimum_afs_free / 1000),
        "maximum_unpledged_afs_m": float(maximum_afs_free / 1000),
        "withdrawal_m": float(withdrawal / 1000),
        "cash_shortfall_m": float(gap / 1000),
        "afs_only_feasibility": feasibility,
        "pessimistic_sale_loss_m": float(sale_loss / 1000),
        "pessimistic_ending_book_equity_m": float((equity - sale_loss) / 1000),
    }


@lru_cache(maxsize=1)
def ground_truth():
    out = analysis()
    return tuple(out[v.name] for v in variables)


def _check(outputs, names):
    by = {v.name: v for v in variables}
    truth = dict(zip(by, ground_truth()))
    return validate_ordered_outputs(outputs, [by[n] for n in names], [truth[n] for n in names], [PRECISION[n] for n in names])


def validate_turn_1(outputs): return _check(outputs, TURNS[0])
def validate_turn_2(outputs): return _check(outputs, TURNS[1])
def validate_turn_3(outputs): return _check(outputs, TURNS[2])
def validate_turn_4(outputs): return _check(outputs, TURNS[3])
def validate(outputs): return _check(outputs, tuple(v.name for v in variables))


validators = {f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1, 5)}
