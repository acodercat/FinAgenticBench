"""13F public filing lineage, additive holdings and revised concentration decision.

Baseline wrong paths: summing all four filing values produces 1,041,906,755
USD rather than the final 263,917,093 USD public information-table value.
Replacing the last restatement with the one-row NEW HOLDINGS amendment produces
4,001,782 USD. Treating the LivePerson note's 8,793,000 PRN quantity as common
shares also violates the 13F quantity type. The hypothetical concentration
decision reverses when the correct public table replaces the old version.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


OUTPUTS = (
    ("thirteen_f_chain_count", 0, "count of the original and qualifying amendments as an integer"),
    ("thirteen_f_latest_restatement", None, "latest restatement's exact 13F accession as a string"),
    ("thirteen_f_added_accession", None, "added-holdings amendment's exact 13F accession as a string"),
    ("thirteen_f_original_value_m", 3, "original public information-table value in USD millions"),
    ("thirteen_f_original_apple_weight_pct", 4, "Apple common-stock value share of the original public 13F information table, percent"),
    ("thirteen_f_final_value_m", 3, "public information-table value after the latest restatement and added-holdings amendment, USD millions"),
    ("thirteen_f_final_apple_weight_pct", 4, "Apple common-stock value share of the reconstructed later public 13F information table, percent"),
    ("thirteen_f_first_cutoff_sale_shares", 0, "minimum whole Apple common shares sold to meet the hypothetical 4.7 percent cap using the first public cutoff"),
    ("thirteen_f_later_cutoff_sale_shares", 0, "minimum whole Apple common shares sold to meet the hypothetical 4.7 percent cap using the later public cutoff"),
)
variables = [Variable(name, None, f"Store {description}" +
    (f", rounded to {precision} decimals from unrounded inputs." if precision is not None else "."))
    for name, precision, description in OUTPUTS]
PRECISION = {name: precision for name, precision, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 3), (3, 5), (5, 7), (7, 9)))


def d(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite filing value")
    return result


def analysis() -> dict[str, int | str | float]:
    filings = load_expansion_table("sec_13f_filings")
    subset = filings.loc[
        filings.cik.eq("0001532943") & filings.periodofreport.eq("2024-12-31")
        & filings.filing_date.le("2025-03-12")
        & filings.submissiontype.isin(["13F-HR", "13F-HR/A"])
    ].copy()
    originals = subset.loc[subset.isamendment.eq("N")].sort_values(
        ["filing_date", "accession_number"])
    restatements = subset.loc[subset.amendmenttype.eq("RESTATEMENT")].sort_values(
        ["filing_date", "accession_number"])
    additions = subset.loc[subset.amendmenttype.eq("NEW HOLDINGS")].sort_values(
        ["filing_date", "accession_number"])
    if len(subset) != 4 or len(originals) != 1 or len(restatements) != 2 or len(additions) != 1:
        raise ValueError("unexpected 13F amendment lineage")
    if not originals.value_reporting_convention.eq("dollars").all() or \
            subset.value_reporting_convention.dropna().ne("dollars").any():
        raise ValueError("unexpected 13F reporting unit")
    original = originals.iloc[0]
    restated = restatements.iloc[-1]
    added = additions.iloc[0]
    holdings = load_expansion_table("sec_13f_amendment_chain")

    def position(accession: str):
        rows = holdings.loc[holdings.accession.eq(accession)]
        if rows.empty or len(rows) != int(subset.loc[subset.accession_number.eq(accession),
                                                      "tableentrytotal"].iloc[0]):
            raise ValueError("13F row count differs from summary")
        total = sum((d(v) for v in rows.reported_value_usd), Decimal(0))
        reported = d(subset.loc[subset.accession_number.eq(accession),
                                "tablevaluetotal"].iloc[0])
        if total != reported:
            raise ValueError("13F information-table value differs from summary")
        apple = rows.loc[rows.cusip.eq("037833100") & rows.quantity_type.eq("SH")
                         & rows.option_side.isna() & rows.title_of_class.eq("COM")]
        return total, sum((d(v) for v in apple.reported_value_usd), Decimal(0)), \
            sum((d(v) for v in apple.reported_quantity), Decimal(0))

    original_total, original_apple_value, original_apple_shares = position(original.accession_number)
    rest_total, rest_apple_value, rest_apple_shares = position(restated.accession_number)
    added_total, added_apple_value, _ = position(added.accession_number)
    if added_apple_value != 0 or rest_apple_shares <= 0:
        raise ValueError("unexpected Apple row in additive amendment")
    final_total = rest_total + added_total
    cap = Decimal("0.047")

    def minimum_sale(total: Decimal, apple_value: Decimal, shares: Decimal) -> int:
        if total <= 0 or shares <= 0:
            raise ValueError("invalid concentration base")
        excess = apple_value - cap * total
        if excess <= 0:
            return 0
        sale = int((excess / (apple_value / shares)).to_integral_value(
            rounding=ROUND_CEILING))
        if sale > shares:
            raise ValueError("infeasible concentration sale")
        return sale

    scale = Decimal(1_000_000)
    return {
        "thirteen_f_chain_count": len(subset),
        "thirteen_f_latest_restatement": str(restated.accession_number),
        "thirteen_f_added_accession": str(added.accession_number),
        "thirteen_f_original_value_m": float(original_total / scale),
        "thirteen_f_original_apple_weight_pct": float(original_apple_value / original_total * 100),
        "thirteen_f_final_value_m": float(final_total / scale),
        "thirteen_f_final_apple_weight_pct": float(rest_apple_value / final_total * 100),
        "thirteen_f_first_cutoff_sale_shares": minimum_sale(original_total, original_apple_value, original_apple_shares),
        "thirteen_f_later_cutoff_sale_shares": minimum_sale(final_total, rest_apple_value, rest_apple_shares),
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
