"""Hypothetical branch-deposit runoff, funding and equity headroom.

SOD branch balances are deposits, not cash. Selling 40% of securities at 98%
of book creates a realized loss and lower cash proceeds; wholesale borrowing
funds the residual and only its accrued interest reduces year-end equity.
The 2025 daily SOFR path is used retrospectively. The equity floor is a board
scenario limit, not regulatory capital. Alternatives measured in author review:
using book sales as cash understates the draw by 11,142.104 thousand USD;
25% CT runoff leaves 1,093,306.447555 thousand USD of equity, below the
1,095,000 floor. These alternatives are not accepted validator paths.
"""

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


def D(value):
    amount = Decimal(str(value))
    if not amount.is_finite():
        raise ValueError("non-finite amount")
    return amount


variables = [
    Variable("ct_branch_deposits_k", None, "Store Connecticut branch deposits in thousand USD as a finite whole-number amount."),
    Variable("ma_branch_deposits_k", None, "Store Massachusetts branch deposits in thousand USD as a finite whole-number amount."),
    Variable("deposit_reconciliation", None, "Store whether the branch deposit sum equals reported domestic deposits as exactly one canonical token: MATCH or MISMATCH."),
    Variable("runoff_k", None, "Store the scenario deposit runoff in thousand USD as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("securities_loss_k", None, "Store the realized securities-sale loss in thousand USD as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("initial_borrowing_k", None, "Store the initial wholesale borrowing in thousand USD as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("year_end_interest_k", None, "Store the year-end borrowing interest in thousand USD as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("year_end_equity_k", None, "Store the scenario year-end equity in thousand USD as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("max_ct_runoff_pct", None, "Store the highest Connecticut runoff rate in percent as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("extra_stress_equity_k", None, "Store year-end equity under 25% Connecticut runoff in thousand USD as a finite number, rounded to 4 decimals from unrounded inputs."),
]
TURNS = [
    ["ct_branch_deposits_k", "ma_branch_deposits_k", "deposit_reconciliation"],
    ["runoff_k"],
    ["securities_loss_k", "initial_borrowing_k"],
    ["year_end_interest_k", "year_end_equity_k"],
    ["max_ct_runoff_pct", "extra_stress_equity_k"],
]
DECIMAL_BY_NAME = {v.name: (None if v.name == "deposit_reconciliation" else 0 if v.name in {"ct_branch_deposits_k", "ma_branch_deposits_k"} else 4) for v in variables}
DECIMALS = [DECIMAL_BY_NAME[v.name] for v in variables]


def _sofr_factor():
    rates = load_expansion_table("nyfed_reference_rates")
    rates = rates.loc[rates["Rate Type"].eq("SOFR"), ["Effective Date", "Rate (%)"]].dropna()
    if rates["Effective Date"].duplicated().any():
        raise ValueError("duplicate SOFR effective date")
    fixings = rates.set_index(pd.to_datetime(rates["Effective Date"]))["Rate (%)"].sort_index()
    days = pd.date_range("2025-10-01", "2025-12-31", freq="D")
    daily = fixings.reindex(days, method="ffill")
    if daily.isna().any() or len(daily) != 92:
        raise ValueError("incomplete SOFR financing window")
    return sum((D(rate) / 100 + D("0.015")) / 360 for rate in daily)


def analysis():
    sod = load_expansion_table("fdic_sod")
    branches = sod.loc[sod["fdic_certificate"].astype(str).eq("17943") & sod["report_date"].astype(str).eq("2025-06-30")]
    if len(branches) != 54 or branches["branch_id"].duplicated().any() or branches["branch_deposits_thousand_usd"].isna().any():
        raise ValueError("unexpected branch population")
    by_state = branches.groupby("branch_state")["branch_deposits_thousand_usd"].sum()
    ct, ma = D(by_state["CT"]), D(by_state["MA"])
    bankfind = load_expansion_table("fdic_bankfind")
    bank = bankfind.loc[bankfind["CERT"].astype(str).eq("17943") & bankfind["REPDTE"].astype(str).eq("2025-06-30")]
    if len(bank) != 1:
        raise ValueError("ambiguous bank snapshot")
    row = bank.iloc[0]
    depdom, cash, securities, equity = (D(row[column]) for column in ("DEPDOM", "CHBAL", "SC", "EQ"))
    reconciliation = "MATCH" if D(branches["branch_deposits_thousand_usd"].sum()) == depdom else "MISMATCH"
    usable_cash = cash - D(100000)
    sale_book = securities * D("0.40")
    sale_proceeds = sale_book * D("0.98")
    sale_loss = sale_book - sale_proceeds
    factor = _sofr_factor()

    def path(ct_rate):
        runoff = ct * ct_rate + ma * D("0.05")
        borrowing = runoff - usable_cash - sale_proceeds
        if borrowing < 0:
            raise ValueError("scenario no longer requires wholesale borrowing")
        interest = borrowing * factor
        return runoff, borrowing, interest, equity - sale_loss - interest

    runoff, borrowing, interest, end_equity = path(D("0.20"))
    _, _, _, extra_equity = path(D("0.25"))
    max_borrowing = (equity - sale_loss - D(1095000)) / factor
    max_rate = (max_borrowing + usable_cash + sale_proceeds - ma * D("0.05")) / ct
    if not (D("0.20") < max_rate < D("0.25")):
        raise ValueError("equity boundary moved")
    return {
        "ct_branch_deposits_k": float(ct),
        "ma_branch_deposits_k": float(ma),
        "deposit_reconciliation": reconciliation,
        "runoff_k": float(runoff),
        "securities_loss_k": float(sale_loss),
        "initial_borrowing_k": float(borrowing),
        "year_end_interest_k": float(interest),
        "year_end_equity_k": float(end_equity),
        "max_ct_runoff_pct": float(max_rate * 100),
        "extra_stress_equity_k": float(extra_equity),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[v.name] for v in variables)


def _validate(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(outputs, [by_name[name] for name in names], [truth[name] for name in names], [DECIMAL_BY_NAME[name] for name in names])


def validate(outputs):
    return _validate(outputs, [v.name for v in variables])


validators = {f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _validate(outputs, names)) for i, names in enumerate(TURNS, 1)}
