"""Compare government and tax-exempt money-fund class yields under a tax scenario.

The supplied April 2025 classes have asset-weighted seven-day net yields near
4.1556% (Government) and 3.2049% (Other Tax Exempt). Under the explicitly
stipulated federal-only treatment, 22% and 24% marginal rates select different
categories; a 25 bp Government-yield cut selects tax-exempt at both rates.
This does not establish actual tax treatment, investor returns or future yields.
The candidate is a baseline because the tax assumptions and class screen are
public conventions, not an undisclosed wrong-path diagnostic.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


REPORT_DATE = "2025-04-30"
CATEGORIES = ("Government", "Other Tax Exempt")
BILLION = Decimal(1_000_000_000)
OUTPUTS = (
    ("tax_yield_government_classes", 0, "number of retained Government share classes as an integer"),
    ("tax_yield_government_series", 0, "number of distinct Government series represented as an integer"),
    ("tax_yield_exempt_classes", 0, "number of retained Other Tax Exempt share classes as an integer"),
    ("tax_yield_exempt_series", 0, "number of distinct Other Tax Exempt series represented as an integer"),
    ("tax_yield_government_assets_b", 4, "sum of retained Government share-class assets in USD billions"),
    ("tax_yield_government_net_pct", 4, "asset-weighted Government seven-day net yield in percent, from unrounded class amounts"),
    ("tax_yield_exempt_assets_b", 4, "sum of retained Other Tax Exempt share-class assets in USD billions"),
    ("tax_yield_exempt_net_pct", 4, "asset-weighted Other Tax Exempt seven-day net yield in percent, from unrounded class amounts"),
    ("tax_yield_choice_22", None, "higher after-tax category at the stipulated 22 percent rate as exactly GOVERNMENT or OTHER_TAX_EXEMPT"),
    ("tax_yield_choice_24", None, "higher after-tax category at the stipulated 24 percent rate as exactly GOVERNMENT or OTHER_TAX_EXEMPT"),
    ("tax_yield_shock_choice_22", None, "higher after-tax category at 22 percent after the stipulated yield change as exactly GOVERNMENT or OTHER_TAX_EXEMPT"),
    ("tax_yield_shock_choice_24", None, "higher after-tax category at 24 percent after the stipulated yield change as exactly GOVERNMENT or OTHER_TAX_EXEMPT"),
)
variables = [Variable(name, None, "Store " + description +
                      (f", rounded to {places} decimals." if places and places > 0 else "."))
             for name, places, description in OUTPUTS]
PRECISION = {name: places for name, places, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 4), (4, 8), (8, 10), (10, 12)))


def dec(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite class yield or assets")
    return result


def _category_stats(frame: pd.DataFrame, category: str) -> tuple[int, int, Decimal, Decimal]:
    rows = frame.loc[frame.fund_category.eq(category)]
    assets = sum((dec(v) for v in rows.class_net_assets_usd), Decimal(0))
    if not len(rows) or assets <= 0:
        raise ValueError(f"no eligible {category} share classes")
    weighted = sum((dec(a) * dec(y) for a, y in
                    zip(rows.class_net_assets_usd, rows.seven_day_net_yield)), Decimal(0)) / assets
    return len(rows), rows.series_id.nunique(), assets, weighted


def analysis() -> dict[str, int | float | str]:
    series = load_expansion_table("sec_nmfp")
    classes = load_expansion_table("sec_nmfp_class_yields")
    series = series.loc[
        series.report_date.eq(REPORT_DATE) & series.filing_date.le("2025-12-31")
        & series.series_id.notna()
    ].sort_values(["series_id", "filing_date", "accession"]).drop_duplicates("series_id", keep="last")
    if series.accession.duplicated().any():
        raise ValueError("duplicate selected filing accession")
    classes = classes.loc[classes.yield_date.eq(REPORT_DATE)].copy()
    joined = classes.merge(series[["accession", "series_id", "fund_category"]],
                           on="accession", how="inner", validate="many_to_one")
    for column in ("class_net_assets_usd", "seven_day_net_yield", "seven_day_gross_yield"):
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    joined = joined.loc[
        joined.fund_category.isin(CATEGORIES)
        & joined.class_net_assets_usd.gt(0)
        & joined.seven_day_net_yield.gt(0)
        & joined.seven_day_net_yield.le(joined.seven_day_gross_yield)
    ].copy()
    if joined.duplicated(["series_id", "class_id"]).any():
        raise ValueError("duplicate retained class within series")
    gov_count, gov_series, gov_assets, gov_yield = _category_stats(joined, CATEGORIES[0])
    exempt_count, exempt_series, exempt_assets, exempt_yield = _category_stats(joined, CATEGORIES[1])

    def choice(rate: Decimal, cut: Decimal = Decimal(0)) -> str:
        gov_after_tax = (gov_yield - cut) * (Decimal(1) - rate)
        return "GOVERNMENT" if gov_after_tax >= exempt_yield else "OTHER_TAX_EXEMPT"

    return {
        "tax_yield_government_classes": gov_count,
        "tax_yield_government_series": gov_series,
        "tax_yield_exempt_classes": exempt_count,
        "tax_yield_exempt_series": exempt_series,
        "tax_yield_government_assets_b": float(gov_assets / BILLION),
        "tax_yield_government_net_pct": float(gov_yield * 100),
        "tax_yield_exempt_assets_b": float(exempt_assets / BILLION),
        "tax_yield_exempt_net_pct": float(exempt_yield * 100),
        "tax_yield_choice_22": choice(Decimal("0.22")),
        "tax_yield_choice_24": choice(Decimal("0.24")),
        "tax_yield_shock_choice_22": choice(Decimal("0.22"), Decimal("0.0025")),
        "tax_yield_shock_choice_24": choice(Decimal("0.24"), Decimal("0.0025")),
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
