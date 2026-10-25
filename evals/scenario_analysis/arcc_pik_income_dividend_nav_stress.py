"""ARCC filing-note PIK stock/accrual, SOFR bridge and valuation stress.

The current September 2025 schedule annualizes only positions with positive
reported PIK rates, principal and fair value; it is not a quarter's collected
cash or recognized income. Q3 PIK interest is September year-to-date less June
year-to-date, USD59m, not the September year-to-date USD165m. Treating the
static annual schedule proxy USD230.57721m as Q3 income is also incorrect.
Subtracting PIK interest from NII is a partial noncash-income proxy and does
not reconstruct operating cash flow. A hypothetical mark of the selected
positions changes simplified equity/NAV, not historical Q3 investment income.
The 2025-09-30 effective SOFR fixing is 4.24%; using a future or prior fixing
changes the stipulated 30-day bridge cost. The bridge is not a reported ARCC
facility or historical funding need. Its financing amount is based only on the
partial proxy gap, not audited operating cash flow.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


Q2 = "0001287750-25-000038"
Q3 = "0001287750-25-000046"
TURNS = [
    ["pik_position_count", "pik_position_fair_value_m", "annual_pik_runrate_m"],
    ["q3_recognized_pik_interest_m", "q3_unaffiliated_pik_interest_m", "q3_after_tax_nii_m"],
    ["partial_cash_like_dividend_gap_m", "bridge_interest_m"],
    ["stressed_equity_m"],
]
variables = [
    Variable("pik_position_count", None, "Store the retained PIK-bearing investment-row count as a finite integer number."),
    Variable("pik_position_fair_value_m", None, "Store retained position fair value in USD millions, rounded to 4 decimals from unrounded source rows."),
    Variable("annual_pik_runrate_m", None, "Store the static annual PIK-interest run-rate proxy in USD millions, rounded to 4 decimals from unrounded source rows."),
    Variable("q3_recognized_pik_interest_m", None, "Store Q3 2025 reported PIK-interest income in USD millions, rounded to 4 decimals from unrounded filing facts."),
    Variable("q3_unaffiliated_pik_interest_m", None, "Store Q3 2025 PIK-interest income attributed to unaffiliated issuers in USD millions, rounded to 4 decimals."),
    Variable("q3_after_tax_nii_m", None, "Store Q3 2025 reported after-tax net investment income in USD millions, rounded to 4 decimals from its filing fact."),
    Variable("partial_cash_like_dividend_gap_m", None, "Store the positive dividend-over-partial-cash-like-income gap in USD millions, rounded to 3 decimals from unrounded filing facts."),
    Variable("bridge_interest_m", None, "Store the hypothetical 30-day bridge interest in USD millions, rounded to 4 decimals."),
    Variable("stressed_equity_m", None, "Store simplified equity after that hypothetical mark in USD millions, rounded to 4 decimals from unrounded inputs."),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 3, 4, 4]


def _note(notes, accession, tag, date, quarters, dimension="0x00000000"):
    rows = notes.loc[
        notes["adsh"].eq(accession)
        & notes["tag"].eq(tag)
        & notes["ddate"].astype(str).eq(date)
        & pd.to_numeric(notes["qtrs"], errors="coerce").eq(quarters)
        & notes["uom"].eq("USD")
        & notes["iprx"].eq(0)
        & notes["dimh"].eq(dimension),
        "value",
    ]
    if len(rows) != 1:
        raise ValueError(f"ambiguous ARCC note: {accession}/{tag}/{date}/{quarters}/{dimension}")
    value = float(rows.iloc[0])
    if not np.isfinite(value):
        raise ValueError("nonfinite ARCC note")
    return value


@lru_cache(maxsize=1)
def analysis():
    notes = load_expansion_table("sec_notes")
    position_tags = {
        "InvestmentOwnedBalancePrincipalAmount": "principal",
        "InvestmentOwnedAtFairValue": "fair_value",
        "InvestmentInterestRatePaidInKind": "pik_rate",
    }
    position_rows = notes.loc[
        notes["adsh"].eq(Q3)
        & notes["ddate"].astype(str).eq("2025-09-30")
        & pd.to_numeric(notes["qtrs"], errors="coerce").eq(0)
        & notes["iprx"].eq(0)
        & notes["tag"].isin(position_tags),
        ["dimh", "tag", "value"],
    ].copy()
    if position_rows.duplicated(["dimh", "tag"]).any():
        raise ValueError("ambiguous position-level note facts")
    rows = position_rows.pivot(index="dimh", columns="tag", values="value").rename(columns=position_tags)
    rows = rows.reindex(columns=["principal", "fair_value", "pik_rate"])
    for column in rows.columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.loc[
        np.isfinite(rows["principal"]) & rows["principal"].gt(0)
        & np.isfinite(rows["fair_value"]) & rows["fair_value"].gt(0)
        & np.isfinite(rows["pik_rate"]) & rows["pik_rate"].gt(0)
    ].copy()
    if rows.empty or rows["pik_rate"].gt(1).any():
        raise ValueError("invalid PIK-bearing schedule cohort")
    fair_value = rows["fair_value"].sum()
    runrate = (rows["principal"] * rows["pik_rate"]).sum()
    dimensions = load_expansion_table("sec_notes_dimensions")
    unaffiliated = dimensions.loc[
        dimensions["axis"].eq("InvestmentIssuerAffiliation")
        & dimensions["member"].eq("InvestmentUnaffiliatedIssuer"), "dimh"
    ].drop_duplicates()
    if len(unaffiliated) != 1:
        raise ValueError("ambiguous unaffiliated issuer dimension")
    pik_ytd9 = _note(notes, Q3, "InterestIncomeOperatingPaidInKind", "2025-09-30", 3)
    pik_ytd6 = _note(notes, Q2, "InterestIncomeOperatingPaidInKind", "2025-06-30", 2)
    pik_q3 = pik_ytd9 - pik_ytd6
    unaffiliated_pik = _note(notes, Q3, "InterestIncomeOperatingPaidInKind", "2025-09-30", 1, unaffiliated.iloc[0])
    nii = _note(notes, Q3, "InvestmentIncomeOperatingAfterExpenseAndTax", "2025-09-30", 1)
    dividend = _note(notes, Q3, "DividendsCommonStockCash", "2025-09-30", 1)
    equity = _note(notes, Q3, "StockholdersEquity", "2025-09-30", 0)
    rates = load_expansion_table("nyfed_reference_rates")
    fixing = rates.loc[
        rates["Effective Date"].astype(str).eq("2025-09-30") & rates["Rate Type"].eq("SOFR"), "Rate (%)"
    ]
    if len(fixing) != 1:
        raise ValueError("ambiguous SOFR fixing")
    sofr_pct = float(fixing.iloc[0])
    gap = max(0.0, dividend - (nii - pik_q3))
    interest = gap * (sofr_pct / 100 + 0.02) * 30 / 360
    mark = fair_value * 0.15
    if pik_q3 <= 0 or unaffiliated_pik <= 0 or unaffiliated_pik > pik_q3 or nii <= 0 or equity <= mark or not np.isfinite(sofr_pct) or gap <= 0:
        raise ValueError("scenario financial boundary changed")
    return {
        "pik_position_count": len(rows),
        "pik_position_fair_value_m": fair_value / 1_000_000,
        "annual_pik_runrate_m": runrate / 1_000_000,
        "q3_recognized_pik_interest_m": pik_q3 / 1_000_000,
        "q3_unaffiliated_pik_interest_m": unaffiliated_pik / 1_000_000,
        "q3_after_tax_nii_m": nii / 1_000_000,
        "partial_cash_like_dividend_gap_m": gap / 1_000_000,
        "bridge_interest_m": interest / 1_000_000,
        "stressed_equity_m": (equity - mark) / 1_000_000,
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[v.name] for v in variables)


def _validate(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = {v.name: d for v, d in zip(variables, DECIMALS)}
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate(outputs):
    return _validate(outputs, [v.name for v in variables])


validators = {f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _validate(outputs, names)) for i, names in enumerate(TURNS, 1)}
