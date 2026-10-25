"""Securities-lending availability and collateral-reinvestment redemption stress.

N-PORT C.12 loan value is a portion of a reported holding's value, not an
additional holding. C.12 cash-collateral-reinvestment amounts are invested
assets, not unrestricted cash or a complete collateral-received balance.
Measured wrong paths are checked in case-local tests: excluding entire loaned
holdings understates saleable equities; adding collateral investments to cash
overstates immediate redemption resources. The borrowing is hypothetical.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


SERIES_ID = "S000011112"
REPORT_DATE = "2025-09-30"
M = Decimal(1_000_000)
OUTPUTS = (
    ("selected_accession", None, "the exact selected SEC accession identifier as a string"),
    ("net_assets_m", 3, "the selected fund's reported net assets in USD millions"),
    ("holding_count", 0, "the count of reported holding rows as an integer"),
    ("loaned_holding_count", 0, "the number of holdings with a positive amount on loan as an integer"),
    ("loan_value_m", 3, "the aggregate reported value of securities on loan in USD millions"),
    ("unloaned_equity_value_m", 3, "the common-equity market value not on loan in USD millions"),
    ("cash_collateral_reinvestment_m", 3, "the aggregate investment amount identified as cash-collateral reinvestment in USD millions"),
    ("non_cash_collateral_asset_m", 3, "the reported non-cash collateral treated as a fund asset in USD millions"),
    ("redemption_m", 3, "the stipulated redemption amount in USD millions"),
    # Subtracts two outputs already reported at 3 decimals, so the standard pins it
    # one decimal coarser; at 3 the rounded operands admit a 1.0e-3 error against a
    # 6.0e-4 tolerance. Source is USD/1e6, so the operands are not exactly representable.
    ("immediate_funding_gap_m", 2, "the remaining redemption funding gap after eligible unloaned equity and reported non-portfolio cash in USD millions"),
    ("sofr_fixing_pct", 2, "the September 30, 2025 SOFR fixing in percent"),
    ("bridge_interest_m", 4, "interest on the stipulated 30-day bridge at the all-in rate in USD millions"),
    ("reinvestment_markdown_m", 3, "the stipulated collateral-reinvestment asset markdown in USD millions"),
    ("ending_nav_m", 2, "fund net assets after the stipulated redemption, markdown and bridge interest in USD millions"),
)
variables = [Variable(n, None, f"Store {d}" + (f", rounded to {p} decimals from unrounded inputs." if p is not None else ".")) for n,p,d in OUTPUTS]
PRECISION = {n:p for n,p,_ in OUTPUTS}
DECIMALS = tuple(p for _,p,_ in OUTPUTS)
TURNS = (
    tuple(n for n,_,_ in OUTPUTS[:5]),
    tuple(n for n,_,_ in OUTPUTS[5:8]),
    tuple(n for n,_,_ in OUTPUTS[8:10]),
    tuple(n for n,_,_ in OUTPUTS[10:]),
)


def dec(x):
    value = Decimal(str(x))
    if not value.is_finite():
        raise ValueError("non-finite source amount")
    return value


def analysis():
    filings = load_expansion_table("sec_nport")
    filings = filings.loc[
        filings.series_id.eq(SERIES_ID) & filings.report_date.eq(REPORT_DATE)
        & filings.filing_date.le("2025-12-31") & filings.form.eq("NPORT-P")
    ].sort_values(["filing_date", "accession"])
    if filings.empty:
        raise ValueError("no eligible original fund filing")
    filing = filings.iloc[-1]
    holdings = load_expansion_table("sec_nport_lending_cohort")
    holdings = holdings.loc[holdings.accession.eq(filing.accession)].copy()
    if holdings.empty or holdings.holding_id.duplicated().any():
        raise ValueError("missing or duplicated holding cohort")
    # The query says "a positive value reported on loan", so filter on the value,
    # not the flag. The two coincide on the frozen cohort (11 rows, identical sums).
    loan_mask = pd.to_numeric(holdings.loan_value_usd, errors="coerce").fillna(0) > 0
    cash_mask = holdings.is_cash_collateral_reinvestment.eq("Y")
    noncash_mask = holdings.is_non_cash_collateral_asset.eq("Y")
    equities = holdings.loc[holdings.asset_category.eq("EC")]
    if not holdings.loc[loan_mask, "asset_category"].eq("EC").all():
        raise ValueError("selected fund has unexpected non-equity securities on loan")
    if not holdings.loc[cash_mask, "asset_category"].isin(["RA", "STIV"]).all():
        raise ValueError("unexpected collateral-reinvestment asset type")
    if holdings.loc[loan_mask, "loan_value_usd"].isna().any():
        raise ValueError("loaned holding missing amount")
    if holdings.loc[cash_mask, "cash_collateral_reinvestment_value_usd"].isna().any():
        raise ValueError("reinvestment holding missing amount")
    if holdings.loc[equities.index, "is_restricted_security"].ne("N").any():
        raise ValueError("equity eligibility requires nonrestricted holdings")
    equity_value = sum((dec(x) for x in equities.currency_value_usd), Decimal(0))
    loan_value = sum((dec(x) for x in holdings.loc[loan_mask, "loan_value_usd"]), Decimal(0))
    reinvestment = sum((dec(x) for x in holdings.loc[cash_mask, "cash_collateral_reinvestment_value_usd"]), Decimal(0))
    noncash = sum((dec(x) for x in holdings.loc[noncash_mask, "non_cash_collateral_asset_value_usd"]), Decimal(0))
    available_equity = equity_value - loan_value
    if not (0 < loan_value <= equity_value and reinvestment > 0 and available_equity > 0):
        raise ValueError("invalid selected lending cohort")
    nav = dec(filing.net_assets_usd)
    cash = dec(filing.cash_not_elsewhere_reported_usd)
    redemption = nav * Decimal("0.80")
    gap = max(Decimal(0), redemption - available_equity - cash)
    if not 0 < gap < loan_value:
        raise ValueError("selected redemption does not separate loaned exposure")
    rates = load_expansion_table("nyfed_reference_rates")
    rate_row = rates.loc[
        rates["Rate Type"].eq("SOFR") & rates["Effective Date"].eq("2025-09-30"),
        "Rate (%)",
    ]
    if len(rate_row) != 1:
        raise ValueError("missing or duplicate SOFR fixing")
    sofr = dec(rate_row.iloc[0])
    interest = gap * (sofr + Decimal(2)) / 100 * Decimal(30) / Decimal(360)
    markdown = reinvestment * Decimal("0.10")
    ending_nav = nav - redemption - markdown - interest
    if ending_nav <= 0:
        raise ValueError("nonpositive scenario ending NAV")
    return {
        "selected_accession": str(filing.accession),
        "net_assets_m": float(nav/M),
        "holding_count": len(holdings),
        "loaned_holding_count": int(loan_mask.sum()),
        "loan_value_m": float(loan_value/M),
        "unloaned_equity_value_m": float(available_equity/M),
        "cash_collateral_reinvestment_m": float(reinvestment/M),
        "non_cash_collateral_asset_m": float(noncash/M),
        "redemption_m": float(redemption/M),
        "immediate_funding_gap_m": float(gap/M),
        "sofr_fixing_pct": float(sofr),
        "bridge_interest_m": float(interest/M),
        "reinvestment_markdown_m": float(markdown/M),
        "ending_nav_m": float(ending_nav/M),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answer = analysis()
    return tuple(answer[v.name] for v in variables)


def _check(outputs, names):
    by = {v.name:v for v in variables}
    truth = dict(zip(by, ground_truth()))
    return validate_ordered_outputs(outputs, [by[n] for n in names], [truth[n] for n in names], [PRECISION[n] for n in names])


def validate_turn_1(outputs): return _check(outputs, TURNS[0])
def validate_turn_2(outputs): return _check(outputs, TURNS[1])
def validate_turn_3(outputs): return _check(outputs, TURNS[2])
def validate_turn_4(outputs): return _check(outputs, TURNS[3])
def validate(outputs): return _check(outputs, tuple(v.name for v in variables))


validators = {f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1,5)}
