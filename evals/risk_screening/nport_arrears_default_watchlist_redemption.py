"""N-PORT C.9 status union and a hypothetical redemption concentration path.

The selected interest-arrears/deferment flag covers 72 holdings worth
183,152,284.33 USD, while one distinct current-default holding is worth
25,000,000 USD. Counting only default understates the watchlist; summing all
flagged rows without overlap control is invalid in general even though the
frozen selected filing has zero overlap. The historical largest monthly gross
redemption is used only as a stipulated single-event stress size.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


M = Decimal(1_000_000)
CAP = Decimal("0.0041")
OUTPUTS = (
    ("watchlist_status_rows", 0, "number of selected-filing debt-security rows with both C.9 status fields populated as an integer"),
    ("watchlist_accession", None, "selected SEC accession identifier as an exact string"),
    ("watchlist_report_date", None, "selected report date as an ISO YYYY-MM-DD string"),
    ("watchlist_arrears_count", 0, "number of holdings flagged for interest arrears or deferral as an integer"),
    ("watchlist_arrears_value_m", 4, "reported market value of arrears-or-deferral flagged holdings in USD millions"),
    ("watchlist_default_count", 0, "number of holdings currently flagged for default as an integer"),
    ("watchlist_default_value_m", 4, "reported market value of currently default-flagged holdings in USD millions"),
    ("watchlist_overlap_count", 0, "number of holdings carrying both status flags as an integer"),
    ("watchlist_union_value_m", 4, "reported market value of the nonduplicated flagged-holding union in USD millions"),
    ("watchlist_original_pct", 4, "flagged union value divided by reported fund net assets, percent"),
    ("watchlist_original_breach", None, "whether the original flagged union share strictly exceeds the hypothetical 0.41 percent limit as a strict Boolean"),
    ("watchlist_stress_redemption_m", 4, "largest reported monthly gross-redemption amount used as the hypothetical outflow in USD millions"),
    ("watchlist_stress_pct", 4, "flagged union share of fund net assets after the stipulated outflow and before additional watchlist sales, percent"),
    ("watchlist_stress_breach", None, "whether the stressed pre-sale flagged union share strictly exceeds the hypothetical 0.41 percent limit as a strict Boolean"),
    ("watchlist_required_sale_m", 4, "minimum additional flagged-holding value sold at reported marks with proceeds retained as cash, USD millions"),
)
variables = [Variable(name, None, f"Store {description}" +
    (f", rounded to {precision} decimals from unrounded source amounts." if precision is not None and precision > 0 else "."))
    for name, precision, description in OUTPUTS]
PRECISION = {name: precision for name, precision, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 3), (3, 8), (8, 11), (11, 15)))


def d(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite source amount")
    return result


def analysis() -> dict[str, str | int | float | bool]:
    holdings = load_expansion_table("sec_nport_holding_details")
    filings = load_expansion_table("sec_nport")
    filings = filings.loc[
        filings.form.eq("NPORT-P") & filings.filing_date.le("2025-12-31")
        & pd.to_numeric(filings.net_assets_usd, errors="coerce").gt(0)
    ].copy()
    if filings.accession.duplicated().any():
        raise ValueError("duplicate original N-PORT accession")
    valid = holdings.loc[
        holdings.default_flag.isin(["Y", "N"])
        & holdings.interest_arrears_flag.isin(["Y", "N"])
    ].copy()
    if valid.duplicated(["accession", "holding_id"]).any():
        raise ValueError("duplicate status holding")
    valid["value"] = pd.to_numeric(valid.currency_value_usd, errors="coerce")
    if valid.value.isna().any():
        raise ValueError("missing debt-security market value")
    sizes = valid.groupby("accession").size()
    eligible = filings.loc[filings.accession.isin(sizes.index)].copy()
    if eligible.empty:
        raise ValueError("no matched filing with populated debt-status rows")
    accession = sorted(eligible.accession, key=lambda a: (-int(sizes[a]), a))[0]
    filing = eligible.loc[eligible.accession.eq(accession)].iloc[0]
    rows = valid.loc[valid.accession.eq(accession)]
    arrears = rows.interest_arrears_flag.eq("Y")
    default = rows.default_flag.eq("Y")
    union = arrears | default
    arrears_value = sum((d(v) for v in rows.loc[arrears, "value"]), Decimal(0))
    default_value = sum((d(v) for v in rows.loc[default, "value"]), Decimal(0))
    union_value = sum((d(v) for v in rows.loc[union, "value"]), Decimal(0))
    nav = d(filing.net_assets_usd)
    redemptions = [d(filing[f"redemptions_month_{i}_usd"]) for i in (1, 2, 3)]
    if min(redemptions) < 0:
        raise ValueError("negative gross redemption")
    outflow = max(redemptions)
    stressed_nav = nav - outflow
    if stressed_nav <= 0 or union_value <= 0:
        raise ValueError("infeasible selected filing")
    original_share = union_value / nav
    stressed_share = union_value / stressed_nav
    sale = max(Decimal(0), union_value - CAP * stressed_nav)
    if sale > union_value:
        raise ValueError("infeasible watchlist sale")
    return {
        "watchlist_status_rows": int(sizes[accession]),
        "watchlist_accession": str(accession),
        "watchlist_report_date": str(filing.report_date),
        "watchlist_arrears_count": int(arrears.sum()),
        "watchlist_arrears_value_m": float(arrears_value / M),
        "watchlist_default_count": int(default.sum()),
        "watchlist_default_value_m": float(default_value / M),
        "watchlist_overlap_count": int((arrears & default).sum()),
        "watchlist_union_value_m": float(union_value / M),
        "watchlist_original_pct": float(original_share * 100),
        "watchlist_original_breach": bool(original_share > CAP),
        "watchlist_stress_redemption_m": float(outflow / M),
        "watchlist_stress_pct": float(stressed_share * 100),
        "watchlist_stress_breach": bool(stressed_share > CAP),
        "watchlist_required_sale_m": float(sale / M),
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
