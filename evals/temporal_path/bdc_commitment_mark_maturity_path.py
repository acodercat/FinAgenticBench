"""Separate a BDC's funded loan maturities from unfunded commitment marks.

The frozen BlackRock Private Credit Fund filing has 342 debt rows with positive
principal, positive reported fair value and maturity. One further positive-
principal row lacks fair value and is excluded, not valued at zero. The funded
cohort first crosses a 10% cumulative-principal maturity screen at 2028-09-30.
There are 55 zero-principal negative-fair-value rows; six mature by that same
date. The filing says negative balances represent unfunded commitments acquired
or valued at a discount, but their marks do not report undrawn notional.

Measured alternative readings: counting the one positive-principal row with
missing fair value would give 343 funded rows and USD 2,092.143582 million of
principal instead of 342 and USD 2,091.012197 million. Dropping all negative
marks would overstate the signed schedule fair value by USD 2.317280 million.
Adding schedule fair value to reported cash gives USD 2,116.228490 million,
USD 26.073568 million below total assets because other asset lines exist.
This is a baseline: the query states the mutually exclusive row cohorts and
asks for their distinct financial meanings.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, load_financial_statements
from core.validation import turn_validator, validate_ordered_outputs


CIK = "1902649"
REPORT_DATE = "2025-09-30"
CUTOFF = "2025-12-31"
MILLION = Decimal(1_000_000)

OUTPUTS = (
    ("bdc_maturity_accession", None,
     "the original 10-Q accession as an exact SEC 10-2-6 digit string"),
    ("bdc_maturity_assets_m", 4, "reported consolidated total assets in USD millions"),
    ("bdc_maturity_cash_m", 4, "reported cash and cash equivalents in USD millions"),
    ("bdc_maturity_schedule_fair_m", 4,
     "the signed sum of reported investment-schedule fair values in USD millions"),
    ("bdc_maturity_funded_count", 0, "the eligible funded debt row count as an integer"),
    ("bdc_maturity_funded_principal_m", 4,
     "eligible funded debt principal in USD millions"),
    ("bdc_maturity_negative_count", 0,
     "the zero-principal negative-fair-value row count as an integer"),
    ("bdc_maturity_negative_fair_m", 4,
     "the signed zero-principal negative fair value in USD millions"),
    ("bdc_maturity_threshold_date", None,
     "the first qualifying September 30 anniversary as an ISO YYYY-MM-DD string"),
    ("bdc_maturity_by_threshold_count", 0,
     "the funded debt row count maturing by the selected anniversary as an integer"),
    ("bdc_maturity_by_threshold_principal_m", 4,
     "funded principal maturing by the selected anniversary in USD millions"),
    ("bdc_maturity_by_threshold_fair_m", 4,
     "the report-date fair value of those funded rows in USD millions"),
    ("bdc_maturity_negative_by_threshold_count", 0,
     "the zero-principal negative-fair-value row count maturing by the selected anniversary as an integer"),
    ("bdc_maturity_negative_by_threshold_fair_m", 4,
     "the signed report-date fair value of those zero-principal rows in USD millions"),
)

variables = [
    Variable(name, None, "Store " + description +
             (f", rounded to {precision} decimals from unrounded source values."
              if precision is not None and precision > 0 else "."))
    for name, precision, description in OUTPUTS
]
PRECISION = {name: precision for name, precision, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 3), (3, 8), (8, 12), (12, 14)))


def _decimal(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("missing or non-finite reported amount")
    return result


def _sum_money(series) -> Decimal:
    return sum((_decimal(value) for value in series), Decimal(0))


def _millions(value: Decimal) -> float:
    return float(value / MILLION)


def _filing():
    schedule = load_expansion_table("sec_bdc")
    schedule = schedule.loc[
        schedule.cik.astype(str).str.lstrip("0").eq(CIK)
        & schedule.report_date.eq(REPORT_DATE)
        & schedule.form.eq("10-Q")
        & schedule.filed_date.le(CUTOFF)
    ].copy()
    if schedule.empty or schedule.accession.nunique() != 1:
        raise ValueError("expected one original BDC quarterly filing")
    if schedule.investment_id.isna().any() or schedule.investment_id.duplicated().any():
        raise ValueError("investment schedule has missing or duplicate row labels")
    accession = str(schedule.accession.iloc[0])
    statements = load_financial_statements()
    statement = statements.loc[
        statements.accession.eq(accession)
        & statements.cik.astype(str).str.lstrip("0").eq(CIK)
        & statements.report_date.eq(REPORT_DATE)
        & statements.form.eq("10-Q")
        & statements.filed_date.le(CUTOFF)
    ]
    if len(statement) != 1:
        raise ValueError("expected one matching financial statement")
    return accession, schedule, statement.iloc[0]


def analysis() -> dict[str, str | int | float]:
    accession, schedule, statement = _filing()
    maturity = pd.to_datetime(schedule.maturity_date, errors="coerce")
    if maturity.isna().sum() != schedule.maturity_date.isna().sum():
        raise ValueError("malformed maturity date")
    report_date = pd.Timestamp(REPORT_DATE)
    valid_maturity = maturity.notna() & maturity.gt(report_date)
    funded = schedule.loc[
        valid_maturity
        & schedule.principal_amount_usd.gt(0)
        & schedule.fair_value_usd.gt(0)
    ].copy()
    negative = schedule.loc[
        valid_maturity
        & schedule.principal_amount_usd.eq(0)
        & schedule.fair_value_usd.lt(0)
    ].copy()
    if funded.empty or negative.empty:
        raise ValueError("missing funded or negative-value cohort")
    funded_maturity = maturity.loc[funded.index]
    negative_maturity = maturity.loc[negative.index]
    funded_principal = _sum_money(funded.principal_amount_usd)

    max_year = int(funded_maturity.dt.year.max())
    selected_date = None
    for year in range(report_date.year + 1, max_year + 2):
        anniversary = pd.Timestamp(year=year, month=9, day=30)
        due = funded.loc[funded_maturity.le(anniversary)]
        if _sum_money(due.principal_amount_usd) * 10 >= funded_principal:
            selected_date = anniversary
            break
    if selected_date is None:
        raise ValueError("funded cohort never reaches the maturity threshold")
    due_funded = funded.loc[funded_maturity.le(selected_date)]
    due_negative = negative.loc[negative_maturity.le(selected_date)]

    return {
        "bdc_maturity_accession": accession,
        "bdc_maturity_assets_m": _millions(_decimal(statement.Assets)),
        "bdc_maturity_cash_m": _millions(_decimal(statement.CashAndCashEquivalentsAtCarryingValue)),
        "bdc_maturity_schedule_fair_m": _millions(_sum_money(
            schedule.fair_value_usd.dropna())),
        "bdc_maturity_funded_count": int(len(funded)),
        "bdc_maturity_funded_principal_m": _millions(funded_principal),
        "bdc_maturity_negative_count": int(len(negative)),
        "bdc_maturity_negative_fair_m": _millions(_sum_money(negative.fair_value_usd)),
        "bdc_maturity_threshold_date": selected_date.strftime("%Y-%m-%d"),
        "bdc_maturity_by_threshold_count": int(len(due_funded)),
        "bdc_maturity_by_threshold_principal_m": _millions(
            _sum_money(due_funded.principal_amount_usd)),
        "bdc_maturity_by_threshold_fair_m": _millions(
            _sum_money(due_funded.fair_value_usd)),
        "bdc_maturity_negative_by_threshold_count": int(len(due_negative)),
        "bdc_maturity_negative_by_threshold_fair_m": _millions(
            _sum_money(due_negative.fair_value_usd)),
    }


@lru_cache(maxsize=1)
def ground_truth():
    result = analysis()
    return tuple(result[v.name] for v in variables)


def _check(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(outputs, [by_name[name] for name in names],
                                    [truth[name] for name in names],
                                    [PRECISION[name] for name in names])


def validate_turn_1(outputs): return _check(outputs, TURNS[0])
def validate_turn_2(outputs): return _check(outputs, TURNS[1])
def validate_turn_3(outputs): return _check(outputs, TURNS[2])
def validate_turn_4(outputs): return _check(outputs, TURNS[3])
def validate(outputs): return _check(outputs, tuple(v.name for v in variables))


validators = {f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
              for i in range(1, 5)}
