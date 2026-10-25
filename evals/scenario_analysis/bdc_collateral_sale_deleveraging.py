"""Hypothetical secured-facility cure using a BDC filing and realized SOFR path.

The filing does not identify an actual pledged borrowing base, and all facility
terms here are scenario assumptions. Treating each selected investment as an
eligible unit is a contractual hypothetical, not an assertion of lien status.

Measured rejected paths: dividing the initial USD871.64m shortage by sale cash
alone gives 5.5338427587% rather than 24.9022924142%, and leaves USD677.9422222m
of immediate deficiency because sold collateral leaves the base. The minimum
immediate sale leaves USD88.6174077m of accrued year-end deficiency. Gross
reported-cost-basis sale P&L is -USD1638.9493557m, while incremental post-stress
NAV change is only -USD435.82m; using the former as the incremental NAV effect
double-counts the pre-sale stress mark. Unrounded state is carried throughout.
"""

from decimal import Decimal
from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


M = Decimal(1_000_000)
ACCESSION = "0001287750-25-000046"
REPORT_DATE = "2025-09-30"
CIK = "1287750"
COLS = ("principal_amount_usd", "cost_usd", "fair_value_usd")
OUTPUTS = (
    ("eligible_investment_count", 0, "the eligible investment-row count as an integer"),
    ("opening_borrowing_base_m", 4, "the hypothetical opening borrowing base in USD millions"),
    ("opening_facility_debt_m", 4, "the hypothetical opening facility debt in USD millions"),
    ("stressed_borrowing_deficiency_m", 4, "the post-markdown facility debt in excess of the borrowing base, in USD millions"),
    ("immediate_minimum_sale_pct", 7, "the minimum immediate pro-rata sale as a percentage of each eligible holding"),
    ("immediate_realized_pnl_m", 4, "signed sale proceeds less allocated reported adjusted cost basis, in USD millions"),
    ("immediate_incremental_nav_change_m", 4, "signed incremental NAV change relative to stressed carrying value from the sale, in USD millions"),
    ("year_end_borrowing_deficiency_m", 4, "the accrued year-end facility debt in excess of the unchanged post-sale borrowing base, in USD millions"),
    ("whole_window_minimum_sale_pct", 7, "the minimum up-front pro-rata sale covering the entire financing window as a percentage of each eligible holding"),
)
variables = [Variable(name, None, f"Store {description}, rounded to {places} decimals from unrounded source inputs.")
             for name, places, description in OUTPUTS]
PRECISION = {name: places for name, places, _ in OUTPUTS}
TURN_NAMES = (
    ("eligible_investment_count", "opening_borrowing_base_m", "opening_facility_debt_m"),
    ("stressed_borrowing_deficiency_m",),
    ("immediate_minimum_sale_pct", "immediate_realized_pnl_m", "immediate_incremental_nav_change_m"),
    ("year_end_borrowing_deficiency_m",),
    ("whole_window_minimum_sale_pct",),
)


def D(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Non-finite filing or rate amount")
    return result


def _filing_pool():
    filing = load_expansion_table("sec_bdc")
    filing = filing.loc[
        filing.accession.eq(ACCESSION) & filing.report_date.eq(REPORT_DATE)
        & filing.cik.astype("string").eq(CIK)
    ]
    values = filing[list(COLS)]
    good = values.notna().all(axis=1) & values.gt(0).all(axis=1)
    good &= pd.Series(np.isfinite(values.fillna(0).to_numpy(dtype=float)).all(axis=1), index=values.index)
    pool = filing.loc[good]
    if pool.empty or pool.investment_id.isna().any() or pool.investment_id.duplicated().any():
        raise ValueError("Missing or duplicate eligible BDC investment rows")
    return [tuple(D(x) / M for x in row) for row in pool[list(COLS)].itertuples(index=False, name=None)]


def _rate_factor():
    rates = load_expansion_table("nyfed_reference_rates")
    rates = rates.loc[rates["Rate Type"].eq("SOFR"), ["Effective Date", "Rate (%)"]]
    if rates["Effective Date"].duplicated().any():
        raise ValueError("Duplicate SOFR effective dates")
    fixes = rates.set_index(pd.to_datetime(rates["Effective Date"]))["Rate (%)"].sort_index()
    days = pd.date_range("2025-11-03", "2025-12-31", freq="D")
    daily = fixes.reindex(days, method="ffill")
    if len(daily) != 59 or daily.isna().any():
        raise ValueError("Incomplete financing-period SOFR path")
    if any(D(x) / 100 + Decimal("0.02") <= 0 for x in daily):
        raise ValueError("All-in daily rate is not strictly positive")
    factor = sum(((D(x) / 100 + Decimal("0.02")) / 360 for x in daily), Decimal(0))
    if factor <= 0:
        raise ValueError("Nonpositive financing factor")
    return factor


def analysis():
    rows = _filing_pool()
    cost = sum((r[1] for r in rows), Decimal(0))
    fair_value = sum((r[2] for r in rows), Decimal(0))
    eligible = sum((min(r[0], r[2]) for r in rows), Decimal(0))
    opening_base, debt = eligible * Decimal("0.70"), eligible * Decimal("0.60")
    stressed_value = fair_value * Decimal("0.80")
    stressed_base = sum((min(r[0], r[2] * Decimal("0.80")) for r in rows), Decimal(0)) * Decimal("0.70")
    full_sale_proceeds = stressed_value * Decimal("0.90")
    shortage = debt - stressed_base
    cure_per_unit_sold = full_sale_proceeds - stressed_base
    if not (shortage > 0 and cure_per_unit_sold > 0):
        raise ValueError("No unique immediate sale cure in selected filing")
    immediate_sale = shortage / cure_per_unit_sold
    immediate_debt = debt - immediate_sale * full_sale_proceeds
    immediate_base = stressed_base * (1 - immediate_sale)
    if not (0 < immediate_sale < 1) or abs(immediate_debt - immediate_base) > Decimal("1e-20"):
        raise ValueError("Immediate covenant cure failed")
    factor = _rate_factor()
    horizon_sale = (debt * (1 + factor) - stressed_base) / (full_sale_proceeds * (1 + factor) - stressed_base)
    if not immediate_sale < horizon_sale < 1:
        raise ValueError("Whole-window cure outside expected domain")
    if debt - horizon_sale * full_sale_proceeds <= 0:
        raise ValueError("Whole-window cure unexpectedly pays off all debt")
    terminal = immediate_debt * (1 + factor) - immediate_base
    horizon_terminal = (debt - horizon_sale * full_sale_proceeds) * (1 + factor) - stressed_base * (1 - horizon_sale)
    if terminal <= 0 or abs(horizon_terminal) > Decimal("1e-20"):
        raise ValueError("Year-end covenant check failed")
    return {
        "eligible_investment_count": len(rows),
        "opening_borrowing_base_m": float(opening_base),
        "opening_facility_debt_m": float(debt),
        "stressed_borrowing_deficiency_m": float(shortage),
        "immediate_minimum_sale_pct": float(immediate_sale * 100),
        "immediate_realized_pnl_m": float(immediate_sale * (full_sale_proceeds - cost)),
        "immediate_incremental_nav_change_m": float(immediate_sale * (full_sale_proceeds - stressed_value)),
        "year_end_borrowing_deficiency_m": float(terminal),
        "whole_window_minimum_sale_pct": float(horizon_sale * 100),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answer = analysis()
    return tuple(answer[v.name] for v in variables)


def _validate(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names],
                                    [truth[n] for n in names], [PRECISION[n] for n in names])


def validate_turn_1(outputs): return _validate(outputs, TURN_NAMES[0])
def validate_turn_2(outputs): return _validate(outputs, TURN_NAMES[1])
def validate_turn_3(outputs): return _validate(outputs, TURN_NAMES[2])
def validate_turn_4(outputs): return _validate(outputs, TURN_NAMES[3])
def validate_turn_5(outputs): return _validate(outputs, TURN_NAMES[4])
def validate(outputs): return _validate(outputs, tuple(v.name for v in variables))


validators = {f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
              for i in range(1, 6)}
